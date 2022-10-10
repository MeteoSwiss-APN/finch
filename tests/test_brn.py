from typing import Tuple
import xarray as xr
import dask.array as da
import numpy as np
import finch
from finch import brn

finch.start_scheduler(debug=True)

class TestBRN:
    shape = [2, 2, 3]
    chunks = {"x" : 1}
    dims = finch.brn.input.translate_order("xyz")

    def get_thetav_test_data(self) -> Tuple[list[xr.DataArray], xr.DataArray]:
        p = 1023.4
        t = 127.3
        qv = 0.3
        tv1 = (finch.const.P0 / p) ** (finch.const.PC_R_D/finch.const.PC_CP_D) \
            * (1.+((finch.const.PC_R_V / finch.const.PC_R_D - 1.0)*qv / (1.-qv)))
        tv = tv1 * t
        inout = [
            np.full(self.shape, p),
            np.full(self.shape, t),
            np.full(self.shape, qv),
            np.full(self.shape, tv)
            ]
        # modify top level (for brn to not be zero everywhere)
        inout[1][:, :, -1] -= 1.
        inout[-1][:, :, -1] = tv - tv1
        inout = [xr.DataArray(da.from_array(x), dims=self.dims).chunk(self.chunks) for x in inout]
        return inout[:-1], inout[-1]

    def get_brn_test_data(self) -> Tuple[list[xr.DataArray], xr.DataArray]:
        input, tv = self.get_thetav_test_data()
        tv = np.array(tv.data)
        tv_top = tv[0, 0, -1]
        tv = tv[0, 0, 0]
        cs = np.full(self.shape[-1], tv)
        cs[0] = tv_top
        cs = cs.cumsum()
        hhl = 3.
        hsurf = 1.5
        u = 42.
        v = 69.
        nlevels_da = np.arange(self.shape[-1], 0, -1).astype(float)
        brn_1 = finch.const.PC_G * (hhl-hsurf)* (tv - tv_top) * nlevels_da
        brn_1[-1] = 0 # the top is always zero
        brn_2 = cs*(u*u + v*v)
        brn = brn_1 / brn_2
        brn = np.tile(brn, self.shape[:-1] + [1])
        inout = [
            np.full(self.shape, u),
            np.full(self.shape, v),
            np.full(self.shape, hhl),
            np.full(self.shape[:-1], hsurf),
            brn
        ]
        dims = [self.dims]*3 + [self.dims[:-1]] + [self.dims]
        inout = input + [xr.DataArray(da.from_array(x), dims=d).chunk(self.chunks) for x, d in zip(inout, dims)]
        return inout[:-1], inout[-1]

    def test_thetav(self):
        """Tests all thetav implementations on the test data"""
        input, output = self.get_thetav_test_data()
        for tv in brn.list_thetav_implementations():
            out = tv(*input).transpose(*output.dims)
            assert da.allclose(out.data, output.data).compute(), f"Function {tv.__name__} returned a wrong array."

    def test_brn(self):
        """Tests all brn implementations on the test data"""
        input, output = self.get_brn_test_data()
        for fbrn in brn.list_brn_implementations():
            out = fbrn(*input).transpose(*output.dims)
            assert da.allclose(out.data, output.data).compute(), f"Function {fbrn.__name__} returned a wrong array."