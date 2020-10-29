import math

import torch
from torch import nn

TREND_ID = "trend"
SEASONALITY_ID = "seasonality"
GENERIC_ID = "generic"


class NBeatsInterpretable(nn.Module):
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
        num_seasonal_block_units=1024,
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
        for stack in self.stacks:
            bcst, fcst = stack(x)
            x = x - bcst
            y = y + fcst
        return x, y


class NBeatsGeneric(nn.Module):
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
                share_weights=True,
                **block_params
            ) for _ in range(num_stacks)
        ])

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
        for stack in self.stacks:
            bcst, fcst = stack(x)
            x = x - bcst
            y = y + fcst
        return x, y


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


def linspace(bcst_len, fcst_len):
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
        share_thetas
    ):
        super().__init__()
        self.device = device
        self.num_units = num_units
        self.theta_dim = theta_dim
        self.bcst_len = bcst_len
        self.fcst_len = fcst_len
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(in_features=self.bcst_len, out_features=num_units)
        self.fc2 = nn.Linear(in_features=num_units, out_features=num_units)
        self.fc3 = nn.Linear(in_features=num_units, out_features=num_units)
        self.fc4 = nn.Linear(in_features=num_units, out_features=num_units)

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
        x: Tensor(shape=[batch_size, bcst_len])

        Returns
        -------
        theta_bcst: Tensor(shape=(batch_size, theta_dim))
        theta_fcst: Tensor(shape=(batch_size, theta_dim))
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        theta_bcst = self.theta_bcst_fc(x)
        theta_fcst = self.theta_fcst_fc(x)
        return theta_bcst, theta_fcst


def trend_basis(theta, t, device):
    """
    Calculate polynomial trend terms.

    Parameters
    ----------
    theta: Tensor(shape=[batch_size, theta_dim])
    t: Tensor(shape=[num_steps])
    device: str

    Returns
    -------
    Tensor(shape=[batch_size, num_steps])
    """
    power = torch.arange(theta.shape[1])
    T = t**power.reshape(-1, 1)
    return torch.matmul(theta, T.to(device))


def seasonal_basis(theta, t, num_terms, period, device):
    """
    Parameters
    ----------
    theta: Tensor(shape=[batch_size, theta_dim])
    t: Tensor(shape=[num_steps])
    num_terms: int
        Number of (sin, cos) pairs.
    period: int
        Lowest frequency perod.
    device: str

    Returns
    -------
    Tensor(shape=[batch_size, num_steps])
    """
    idx = (t/period)*(2*math.pi*torch.arange(0, num_terms).reshape(-1, 1))
    cos = torch.cos(idx)
    sin = torch.sin(idx[1:])
    # S.shape = (2*num_terms - 1, num_steps])
    S = torch.cat((cos, sin), axis=0)
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
        num_steps = bcst_len + fcst_len
        self.time_bcst /= num_steps
        self.time_fcst /= num_steps

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor(shape=[batch_size, bcst_len])

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
        theta_dim = 2*num_seasonal_terms - 1
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
        x: Tensor(shape=[batch_size, bcst_len])

        Returns
        -------
        bcst: Tensor(shape=(batch_size, bcst_len))
        fcst: Tensor(shape=(batch_size, fcst_len))
        """
        theta_fcst, theta_bcst = super().forward(x)
        bcst = seasonal_basis(
            theta=theta_bcst,
            t=self.time_bcst,
            num_terms=self.num_seasonal_terms,
            period=self.period,
            device=self.device
        )
        fcst = seasonal_basis(
            theta=theta_fcst,
            t=self.time_fcst,
            num_terms=self.num_seasonal_terms,
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
        x: Tensor(shape=[batch_size, bcst_len])

        Returns
        -------
        bcst: Tensor(shape=(batch_size, bcst_len))
        fcst: Tensor(shape=(batch_size, fcst_len))
        """
        theta_bcst, theta_fcst = super().forward(x)
        bcst = self.bcst_fc(torch.relu(theta_bcst))
        fcst = self.fcst_fc(torch.relu(theta_fcst))
        return bcst, fcst


def test_seasonal_block():
    bcst_len = 32
    fcst_len = 64
    batch_size = 8
    sb = SeasonalityBlock(
        device="cpu",
        num_units=32,
        bcst_len=bcst_len,
        fcst_len=fcst_len,
        period=1,
        num_seasonal_terms=5
    )
    x = torch.zeros((batch_size, bcst_len))
    bcst, fcst = sb(x)
    assert bcst.shape == x.shape
    assert fcst.shape == (batch_size, fcst_len)


def test_trend_block():
    bcst_len = 32
    fcst_len = 64
    batch_size = 8
    tb = TrendBlock(
        device="cpu",
        num_units=32,
        bcst_len=bcst_len,
        fcst_len=fcst_len,
        trend_degree=4
    )
    x = torch.zeros((batch_size, bcst_len))
    bcst, fcst = tb(x)
    assert bcst.shape == x.shape
    assert fcst.shape == (batch_size, fcst_len)


def test_generic_block():
    bcst_len = 32
    fcst_len = 64
    batch_size = 8
    gb = GenericBlock(
        device="cpu",
        num_units=32,
        bcst_len=bcst_len,
        fcst_len=fcst_len,
        theta_dim=10
    )
    x = torch.zeros((batch_size, bcst_len))
    bcst, fcst = gb(x)
    assert bcst.shape == x.shape
    assert fcst.shape == (batch_size, fcst_len)
