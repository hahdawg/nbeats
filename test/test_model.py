import unittest

import torch
import nbeats.model as nbm


class ModelTester(unittest.TestCase):
    batch_size = 126
    device = "cpu"
    bcst_len = 32
    fcst_len = 16
    num_units = 8

    def test_nbeats_interpretable(self):
        nb = nbm.NBeatsInterpretable(
            device=self.device,
            bcst_len=self.bcst_len,
            fcst_len=self.fcst_len
        )
        x = torch.zeros((self.batch_size, self.bcst_len))
        bcst, fcst = nb(x)
        self.assertEqual(bcst.shape, x.shape)
        self.assertEqual(fcst.shape, (self.batch_size, self.fcst_len))

    def test_nbeats_generic(self):
        nb = nbm.NBeatsGeneric(
            device=self.device,
            bcst_len=self.bcst_len,
            fcst_len=self.fcst_len
        )
        x = torch.zeros((self.batch_size, self.bcst_len))
        bcst, fcst = nb(x)
        self.assertEqual(bcst.shape, x.shape)
        self.assertEqual(fcst.shape, (self.batch_size, self.fcst_len))

    def test_seasonal_block(self):
        sb = nbm.SeasonalityBlock(
            device=self.device,
            num_units=self.num_units,
            bcst_len=self.bcst_len,
            fcst_len=self.fcst_len,
            period=1,
            num_seasonal_terms=5
        )
        x = torch.zeros((self.batch_size, self.bcst_len))
        bcst, fcst = sb(x)
        self.assertEqual(bcst.shape, x.shape)
        self.assertEqual(fcst.shape, (self.batch_size, self.fcst_len))

    def test_trend_block(self):
        tb = nbm.TrendBlock(
            device=self.device,
            num_units=self.num_units,
            bcst_len=self.bcst_len,
            fcst_len=self.fcst_len,
            trend_degree=4
        )
        x = torch.zeros((self.batch_size, self.bcst_len))
        bcst, fcst = tb(x)
        self.assertEqual(bcst.shape, x.shape)
        self.assertEqual(fcst.shape, (self.batch_size, self.fcst_len))

    def test_generic_block(self):
        gb = nbm.GenericBlock(
            device="cpu",
            num_units=self.num_units,
            bcst_len=self.bcst_len,
            fcst_len=self.fcst_len,
            theta_dim=10
        )
        x = torch.zeros((self.batch_size, self.bcst_len))
        bcst, fcst = gb(x)
        self.assertEqual(bcst.shape, x.shape)
        self.assertEqual(fcst.shape, (self.batch_size, self.fcst_len))


if __name__ == "__main__":
    unittest.main()
