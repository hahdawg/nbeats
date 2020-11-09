from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as td


from nbeats.model import NBeatsInterpretable, NBeatsGeneric


def load_data():
    """
    Load M4 monthly data from disk.
    """
    res = pd.read_csv("/home/hahdawg/projects/nbeats/examples/data/Monthly-train.csv")
    return res.values


def process_data(df):
    """
    For each ts in data, get the first and last valid idx.

    Returns
    -------
    data, series_id, first_valid_idx, last_valid_idx
    """
    series_id = df[:, 0]
    data = df[:, 1:].astype(float)
    notna = (~np.isnan(data)).cumsum(axis=1)
    first_valid_idx = np.nanargmin(notna, axis=1)
    last_valid_idx = np.nanargmax(notna, axis=1)
    return data, series_id, first_valid_idx, last_valid_idx


def generate_batches(
    processed,
    batch_size,
    bcst_len,
    fcst_len,
    num_iter=1000000,
    lh=1.5,
    check_nan=False,
    seed=42
):
    """
    Batch generator written in numpy.

    Parameters
    ----------
    processed: (np.array,)
        Result of process_data.
    batch_size: int
    bcst_len: int
    fcst_len: int
    num_iter: int
    lh: float
    check_nan: bool
    seed: int

    Yields
    ------
    x_tr, x_val, y_tr, y_val
    """
    np.random.seed(seed)
    ts, series_id, _, last_valid_idx = processed
    sample_idx = np.arange(ts.shape[0])
    max_offset = int(lh*fcst_len)
    for _ in range(num_iter):
        offset = np.random.randint(low=fcst_len, high=max_offset)

        # Sample batch_size rows from ts
        idx_batch = np.random.choice(sample_idx, size=batch_size, replace=False)
        series_id_batch = series_id[idx_batch]
        ts_batch = ts[idx_batch]

        # Get columns to use. Spacing shown below.
        #
        #  |---------bcst_len----------|----offset----|--fcst_len--|
        # fvi                                                     lvi
        #
        lvi_batch = last_valid_idx[idx_batch]
        fvi_batch = lvi_batch - fcst_len - offset - bcst_len
        time_batch = lvi_batch[0] - fvi_batch[0]
        time_idx = fvi_batch.reshape(-1, 1) + np.arange(time_batch)

        ts_batch = ts_batch[np.arange(ts_batch.shape[0]).reshape(-1, 1), time_idx]
        nan_rows = np.isnan(ts_batch).any(axis=1)
        ts_batch = ts_batch[~nan_rows]
        time_idx_batch = time_idx[~nan_rows]

        if not ts_batch.shape[0]:
            continue

        # Always use the last fcst_len columns for validation
        y_val = ts_batch[:, -fcst_len:]
        x_val = ts_batch[:, -(bcst_len + fcst_len):-fcst_len]

        # Grab train cols based on offset
        y_tr = ts_batch[:, bcst_len:bcst_len + fcst_len]
        x_tr = ts_batch[:, :bcst_len]

        series_id_batch = series_id_batch[~nan_rows]

        if check_nan:
            any_nan = (
                np.isnan(x_tr).any() | np.isnan(x_val).any() |
                np.isnan(y_tr).any() | np.isnan(y_val).any()
            )
            if any_nan:
                raise ValueError()
        yield x_tr, x_val, y_tr, y_val, series_id_batch, time_idx_batch


class M4Dataset(td.Dataset):
    """
    Parameters
    ----------
    processed: (np.array,)
        Result of process_data.
    bcst_len: int
    fcst_len: int
    lh: float
    istrain: bool
    debug_mode: bool
    """
    def __init__(
        self,
        processed,
        bcst_len,
        fcst_len,
        lh=1.5,
        istrain=True,
        debug_mode=True
    ):
        self.bcst_len = bcst_len
        self.fcst_len = fcst_len
        self.lh = lh
        self.max_offset = int(self.lh*self.fcst_len)
        self.istrain = istrain
        self.debug_mode = debug_mode

        self.ts, self.series_id, _, self.last_valid_idx = self._filter_rows(processed)

    def _filter_rows(self, processed):
        """
        If a ts doesn't have enough history, then drop it.

        Parameters
        ----------
        processed: (np.array,)

        Returns
        -------
        (np.array,)
        """
        keep_rows = processed[-1] >= self.bcst_len + self.max_offset + self.fcst_len
        pct_removed = 100*(1 - keep_rows.mean())
        print(f"Filter stage removed {pct_removed:.04f}% of rows.")
        filtered = [xs[keep_rows] for xs in processed]
        return filtered

    def __len__(self):
        return self.ts.shape[0]

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: int

        Returns
        -------
        (x_tr, x_val, y_tr, y_val, series_id) if istrain else (x, series_id)
        """
        ts = self.ts[index]
        series_id = self.series_id[index]
        last_valid_idx = self.last_valid_idx[index]

        # Spacing shown below.
        #
        #  |---------bcst_len----------|----offset----|--fcst_len--|
        # fvi                                                     lvi
        #
        if self.istrain:
            offset = np.random.randint(low=self.fcst_len, high=self.max_offset)
            first_valid_idx = last_valid_idx - self.fcst_len - offset - self.bcst_len
            ts_tr = ts[first_valid_idx:last_valid_idx]
            x_tr = ts_tr[:self.bcst_len]
            y_tr = ts_tr[self.bcst_len:self.bcst_len + self.fcst_len]

            first_valid_idx = last_valid_idx - self.fcst_len - self.bcst_len
            ts_val = ts[first_valid_idx:last_valid_idx]
            x_val = ts_val[-(self.bcst_len + self.fcst_len):-self.fcst_len]
            y_val = ts_val[-self.fcst_len:]
            if self.debug_mode:
                invalid = any((
                    len(x_tr) != self.bcst_len,
                    len(x_val) != self.bcst_len,
                    len(y_tr) != self.fcst_len,
                    len(y_val) != self.fcst_len,
                    np.isnan(x_tr).any(),
                    np.isnan(x_val).any(),
                    np.isnan(y_tr).any(),
                    np.isnan(y_val).any()
                ))
                if invalid:
                    raise ValueError()
            return x_tr, x_val, y_tr, y_val, series_id

        first_valid_idx = last_valid_idx - self.bcst_len
        ts = ts[first_valid_idx:last_valid_idx]
        x = ts[-self.bcst_len:]
        return x, series_id


def plot_output(x, y, y_hat, num_plots):
    """
    Plot A&F for the first num_plots series in y.

    Parameters
    ----------
    x: Tensor(shape=(batch_size, bcst_len))
    y: Tensor(shape=(batch_size, fcst_len))
    y_hat: Tensor(shape=(batch_size, fcst_len))
    num_plots: int
    """
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    actual = np.concatenate([x, y], axis=1)
    pred = np.concatenate([x, y_hat], axis=1)
    pred[:, :x.shape[1]] = np.nan
    plot_idx = np.arange(actual.shape[1])
    for i in range(num_plots):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(plot_idx, actual[i, :], label="actual")
        ax.plot(plot_idx, pred[i, :], label="pred")
        plt.legend()
        fig.savefig(f"/home/hahdawg/tmp/fig{i}.pdf")
        plt.close()


def compute_log_loss(y, y_hat):
    """
    l1 log loss.

    Parameters
    ----------
    y: Tensor(shape=(batch_size, fcst_len))
    y_hat: Tensor(shape=(batch_size, fcst_len))

    Returns
    -------
    float
    """
    y_hat = torch.relu(y_hat)
    return torch.abs(torch.log1p(y) - torch.log1p(y_hat)).mean().item()


def train(data=None, interpretable=False):
    """
    Train nbeats on monthly m4 data.
    """
    if data is None:
        data = load_data()

    proc = process_data(data)
    fcst_len = 18
    bcst_len = int(5*fcst_len)
    num_epochs = 1000000

    dataset = M4Dataset(
        processed=proc,
        bcst_len=bcst_len,
        fcst_len=fcst_len,
        lh=1.5
    )

    num_seasonal_terms = fcst_len // 2

    device = "cuda"
    if interpretable:
        model = NBeatsInterpretable(
            device=device,
            bcst_len=bcst_len,
            fcst_len=fcst_len,
            num_seasonal_terms=num_seasonal_terms,
            seasonal_period=1
        )
    else:
        model = NBeatsGeneric(
            device=device,
            bcst_len=bcst_len,
            fcst_len=fcst_len
        )

    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())

    checkpoint_freq = 500
    l1_loss = torch.nn.L1Loss()
    running_loss = deque(maxlen=checkpoint_freq)
    running_mape_tr = deque(maxlen=checkpoint_freq)
    running_mape_val = deque(maxlen=checkpoint_freq)
    i = 0
    for _ in range(num_epochs):
        batch_generator = td.DataLoader(
            dataset=dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=1,
            drop_last=True
        )

        for x_tr, x_val, y_tr, y_val, _ in batch_generator:
            x_tr = x_tr.float().to(device)
            x_val = x_val.float().to(device)
            y_tr = y_tr.float().to(device)
            y_val = y_val.float().to(device)

            optimizer.zero_grad()
            _, y_tr_hat = model(x_tr)
            loss = l1_loss(y_tr, y_tr_hat)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss.append(loss.mean().item())

                mape_tr = compute_log_loss(y_tr, y_tr_hat)
                running_mape_tr.append(mape_tr)
                _, y_val_hat = model(x_val)

                mape_val = compute_log_loss(y_val, y_val_hat)
                running_mape_val.append(mape_val)

                if i % checkpoint_freq == 0:
                    loss_check = np.mean(running_loss)
                    mape_tr_check = np.mean(running_mape_tr)
                    mape_val_check = np.mean(running_mape_val)
                    loss_msg = (
                        f"step: {i}  l1_loss: {loss_check:.2f}  "
                        f"mape_tr: {mape_tr_check:.4f}  "
                        f"mape_val: {mape_val_check:.4f}"
                    )
                    print(loss_msg)
                    plot_output(x_val, y_val, y_val_hat, 15)
            i += 1



if __name__ == "__main__":
    train()
