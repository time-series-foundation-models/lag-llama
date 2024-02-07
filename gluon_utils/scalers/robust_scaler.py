from __future__ import annotations

import torch
from gluonts.core.component import validated
from gluonts.torch.scaler import Scaler


class RobustScaler(Scaler):
    """
    Computes a scaling factor by removing the median and scaling by the
    interquartile range (IQR).

    Parameters
    ----------
    dim
        dimension along which to compute the scale
    keepdim
        controls whether to retain dimension ``dim`` (of length 1) in the
        scale tensor, or suppress it.
    minimum_scale
        minimum possible scale that is used for any item.
    """

    @validated()
    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = False,
        minimum_scale: float = 1e-10,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def __call__(
        self, data: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            data.shape == weights.shape
        ), "data and observed_indicator must have same shape"

        with torch.no_grad():
            observed_data = torch.where(weights == 1, data, torch.nan)

            med = torch.nanmedian(observed_data, dim=self.dim, keepdim=True).values
            q1 = torch.nanquantile(observed_data, 0.25, dim=self.dim, keepdim=True)
            q3 = torch.nanquantile(observed_data, 0.75, dim=self.dim, keepdim=True)
            iqr = q3 - q1

            # if observed data is all zeros, nanmedian returns nan
            loc = torch.where(torch.isnan(med), torch.zeros_like(med), med)
            scale = torch.where(torch.isnan(iqr), torch.ones_like(iqr), iqr)
            scale = torch.maximum(scale, torch.full_like(iqr, self.minimum_scale))

            scaled_data = (data - loc) / scale

            if not self.keepdim:
                loc = torch.squeeze(loc, dim=self.dim)
                scale = torch.squeeze(scale, dim=self.dim)

            # assert no nans in scaled data, loc or scale
            assert not torch.any(torch.isnan(scaled_data))
            assert not torch.any(torch.isnan(loc))
            assert not torch.any(torch.isnan(scale))
            assert not torch.any(scale == 0)

            return scaled_data, loc, scale
