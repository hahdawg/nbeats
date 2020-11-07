import math

import torch
from torch import nn

TREND_ID = "trend"
SEASONALITY_ID = "seasonality"
GENERIC_ID = "generic"


class NBeats(nn.Module):
    """
    Implement forward method.
    """

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, bcst_len))

        Returns
        -------
        bcst(shape=(batch_size, bcst_len)), fcst(shape=(batch_size, fcst_len))
        """
        y = 0.0
        mean = x.mean(axis=1).reshape(-1, 1)
        x = x - mean
        for stack in self.stacks:
            bcst, fcst = stack(x)
            x = x - bcst
            y = y + fcst
        return x, y + mean


class NBeatsInterpretable(NBeats):
    """
    Parameters
    ----------
    device: str
    bcst_len: int
    fcst_len: int
    num_trend_block_units: int
    num_seasonal_block_units: int
    trend_degree: int
    num_blocks_per_stack: int
    num_seasonal_terms: int
    seasonal_period: int
    """
    def __init__(
        self,
        device,
        bcst_len,
        fcst_len,
        num_trend_block_units=256,
        num_seasonal_block_units=2048,
        trend_degree=2,
        num_blocks_per_stack=3,
        num_seasonal_terms=None,
        seasonal_period=1,
    ):
        super().__init__()
        self.device = device
        self.bcst_len = bcst_len
        self.fcst_len = fcst_len
        self.num_trend_block_units = num_trend_block_units
        self.num_seasonal_block_units = num_seasonal_block_units
        self.trend_degree = trend_degree
        self.num_blocks_per_stack = num_blocks_per_stack
        self.num_seasonal_terms = num_seasonal_terms
        self.seasonal_period = seasonal_period

        trend_stack = Stack(
            block_type=TREND_ID,
            num_blocks=num_blocks_per_stack,
            share_weights=True,
            num_units=num_trend_block_units,
            trend_degree=trend_degree,
            bcst_len=bcst_len,
            fcst_len=fcst_len,
            device=self.device
        )

        seasonal_stack = Stack(
            block_type=SEASONALITY_ID,
            num_blocks=num_blocks_per_stack,
            share_weights=True,
            num_units=num_seasonal_block_units,
            bcst_len=bcst_len,
            fcst_len=fcst_len,
            num_seasonal_terms=num_seasonal_terms,
            period=seasonal_period,
            device=self.device
        )

        self.stacks = nn.ModuleList([trend_stack, seasonal_stack])
        self.to(self.device)


class NBeatsGeneric(NBeats):
    """
    Parameters
    ----------
    device: str
    bcst_len: int
    fcst_len: int
    num_units: int
    num_stacks: int
    num_blocks_per_stack: int
    theta_dim: int
    share_thetas: bool
    """
    def __init__(
        self,
        device,
        bcst_len,
        fcst_len,
        num_units=512,
        num_stacks=30,
        num_blocks_per_stack=1,
        theta_dim=32,
        share_thetas=True
    ):
        super().__init__()
        block_params = {
            "num_units": num_units,
            "theta_dim": theta_dim,
            "bcst_len": bcst_len,
            "fcst_len": fcst_len,
            "share_thetas": share_thetas
        }
        self.stacks = nn.ModuleList([
            Stack(
                device=device,
                block_type=GENERIC_ID,
                num_blocks=num_blocks_per_stack,
                share_weights=False,
                **block_params
            ) for _ in range(num_stacks)
        ])


class Stack(nn.Module):
    """
    Parameters
    ----------
    device: str
    block_type: str
        'trend', 'seasonality', or 'generic'
    share_weights: bool
    block_params: parameters for each block in stack.
    """
    def __init__(self, device, block_type, num_blocks, share_weights, **block_params):
        super().__init__()
        self.device = device
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.share_weights = share_weights

        if block_type == TREND_ID:
            block = TrendBlock
        elif block_type == SEASONALITY_ID:
            block = SeasonalityBlock
        elif block_type == GENERIC_ID:
            block = GenericBlock

        block_params["device"] = self.device
        self.blocks = nn.ModuleList([block(**block_params)])

        for _ in range(1, self.num_blocks):
            if self.share_weights:
                self.blocks.append(self.blocks[-1])
            else:
                self.blocks.append(block(**block_params))

        self.to(self.device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, bcst_len))

        Returns
        -------
        bcst: Tensor(shape=(batch_size, bcst_len))
        fcst: Tensor(shape=(batch_size, fcst_len))
        """
        y = 0.0
        for block in self.blocks:
            bcst, fcst = block(x)
            x = x - bcst
            y = y + fcst
        return x, y


def linspace(bcst_len, fcst_len, normalize=True):
    """
    Centered linspace.

    Parameters
    ----------
    bcst_len: int
    fcst_len: int

    Returns
    -------
    t_bcst: Tensor(shape=bcst_len)),
    t_fcst: Tensor(shape=bcst_len))
    """
    t = torch.linspace(-bcst_len, fcst_len, bcst_len + fcst_len)
    if normalize:
        t = t/t[-1]
    t_bcst = t[:bcst_len]
    t_fcst = t[bcst_len:]
    return t_bcst, t_fcst


class Block(nn.Module):
    """
    Parameters
    ----------
    device: str
    num_units: int
    theta_dim: int
    bcst_len: int
    fcst_len: int
    share_thetas: bool
    """
    def __init__(
        self,
        device,
        num_units,
        theta_dim,
        bcst_len,
        fcst_len,
        share_thetas,
        num_layers=4
    ):
        super().__init__()
        self.device = device
        self.num_units = num_units
        self.theta_dim = theta_dim
        self.bcst_len = bcst_len
        self.fcst_len = fcst_len
        self.share_thetas = share_thetas
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=self.bcst_len, out_features=num_units)] + \
            [nn.Linear(in_features=num_units, out_features=num_units)
             for _ in range(self.num_layers - 1)
            ]
        )

        self.time_bcst, self.time_fcst = linspace(bcst_len, fcst_len)

        if share_thetas:
            self.theta_bcst_fc = self.theta_fcst_fc = nn.Linear(
                in_features=num_units,
                out_features=theta_dim,
                bias=False
            )
        else:
            self.theta_bcst_fc = nn.Linear(
                in_features=num_units,
                out_features=theta_dim,
                bias=False
            )
            self.theta_fcst_fc = nn.Linear(
                in_features=num_units,
                out_features=theta_dim,
                bias=False
            )

        self.to(self.device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, bcst_len))

        Returns
        -------
        theta_bcst: Tensor(shape=(batch_size, theta_dim))
        theta_fcst: Tensor(shape=(batch_size, theta_dim))
        """
        for layer in self.layers:
            x = torch.relu(layer(x))
        theta_bcst = self.theta_bcst_fc(x)
        theta_fcst = self.theta_fcst_fc(x)
        return theta_bcst, theta_fcst


def trend_basis(theta, t, device):
    """
    Calculate polynomial trend terms.

    Parameters
    ----------
    theta: Tensor(shape=(batch_size, theta_dim])
    t: Tensor(shape=[num_steps])
    device: str

    Returns
    -------
    Tensor(shape=(batch_size, num_steps])
    """
    power = torch.arange(theta.shape[1])
    # T.shape = (theta.shape[1], num_steps)
    T = t**power.reshape(-1, 1)
    return torch.matmul(theta, T.to(device))


def seasonal_basis(theta, t, period, device):
    """
    Parameters
    ----------
    theta: Tensor(shape=(batch_size, theta_dim))
        Note: theta_dim must be odd.
    t: Tensor(shape=[num_steps])
    period: int
        Lowest perod.
    device: str

    Returns
    -------
    Tensor(shape=(batch_size, num_steps))
    """
    if theta.shape[1] % 2 == 0:
        raise ValueError("theta.shape[1] must be odd.")

    num_terms = (theta.shape[1] - 1) // 2

    frequencies = 2*math.pi*torch.arange(1, num_terms + 1)/period
    idx = t*(frequencies.reshape(-1, 1))
    constant = torch.ones(idx.shape[1]).reshape(1, -1)
    cos = torch.cos(idx)
    sin = torch.sin(idx)
    # S.shape = (2*num_terms - 1, num_steps])
    S = torch.cat((constant, cos, sin), axis=0)
    return torch.matmul(theta, S.to(device))


class TrendBlock(Block):
    """
    Parameters
    ----------
    device: str
    num_units: int
    trend_degree: int
    bcst_len: int
    fcst_len: int
    share_thetas: bool
    """
    def __init__(
        self,
        device,
        num_units,
        trend_degree,
        bcst_len,
        fcst_len,
        share_thetas=True,
    ):
        super().__init__(
            device=device,
            num_units=num_units,
            theta_dim=trend_degree + 1,
            bcst_len=bcst_len,
            fcst_len=fcst_len,
            share_thetas=share_thetas
        )
        self.to(self.device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, bcst_len))

        Returns
        -------
        bcst: Tensor(shape=(batch_size, bcst_len))
        fcst: Tensor(shape=(batch_size, fcst_len))
        """
        theta_fcst, theta_bcst = super().forward(x)
        bcst = trend_basis(theta_bcst, self.time_bcst, self.device)
        fcst = trend_basis(theta_fcst, self.time_fcst, self.device)
        return bcst, fcst


class SeasonalityBlock(Block):
    """
    Parameters
    ----------
    device: str
    num_units: int
    trend_degree: int
    bcst_len: int
    fcst_len: int
    period: int
    num_seasonal_terms: int
    share_thetas: bool
    """
    def __init__(
        self,
        device,
        num_units,
        bcst_len,
        fcst_len,
        period=1,
        num_seasonal_terms=0,
        share_thetas=True,
    ):
        if not num_seasonal_terms:
            num_seasonal_terms = fcst_len // 2 - 1

        theta_dim = 2*num_seasonal_terms + 1
        super().__init__(
            device=device,
            num_units=num_units,
            theta_dim=theta_dim,
            bcst_len=bcst_len,
            fcst_len=fcst_len,
            share_thetas=share_thetas
        )
        self.period = period
        self.num_seasonal_terms = num_seasonal_terms
        self.to(self.device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, bcst_len))

        Returns
        -------
        bcst: Tensor(shape=(batch_size, bcst_len))
        fcst: Tensor(shape=(batch_size, fcst_len))
        """
        theta_fcst, theta_bcst = super().forward(x)
        bcst = seasonal_basis(
            theta=theta_bcst,
            t=self.time_bcst,
            period=self.period,
            device=self.device
        )
        fcst = seasonal_basis(
            theta=theta_fcst,
            t=self.time_fcst,
            period=self.period,
            device=self.device
        )
        return bcst, fcst


class GenericBlock(Block):
    """
    Parameters
    ----------
    device: str
    num_units: int
    theta_dim: int
    bcst_len: int
    fcst_len: int
    share_thetas: bool
    """
    def __init__(
        self,
        device,
        num_units,
        theta_dim,
        bcst_len,
        fcst_len,
        share_thetas=True,
    ):
        super().__init__(
            device=device,
            num_units=num_units,
            theta_dim=theta_dim,
            bcst_len=bcst_len,
            fcst_len=fcst_len,
            share_thetas=share_thetas
        )

        self.bcst_fc = nn.Linear(in_features=self.theta_dim, out_features=bcst_len)
        self.fcst_fc = nn.Linear(in_features=self.theta_dim, out_features=fcst_len)
        self.to(self.device)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=(batch_size, bcst_len))

        Returns
        -------
        bcst: Tensor(shape=(batch_size, bcst_len))
        fcst: Tensor(shape=(batch_size, fcst_len))
        """
        theta_bcst, theta_fcst = super().forward(x)
        bcst = self.bcst_fc(torch.relu(theta_bcst))
        fcst = self.fcst_fc(torch.relu(theta_fcst))
        return bcst, fcst
