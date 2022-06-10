"""Slater Koster two-center integral elements, from 10.1103/PhysRev.94.1498"""
import numpy as np

class SlaterKoster:

    def __init__(self, **kwargs):
        self.unit_vector = np.array([1, 0, 0])
        self.v_sss = 0
        self.v_sps = 0
        self.v_sds = 0
        self.v_pps = 0
        self.v_ppp = 0
        self.v_pds = 0
        self.v_pdp = 0
        self.v_dds = 0
        self.v_ddp = 0
        self.v_ddd = 0
        [setattr(self, var, kwargs[var]) for var in [*kwargs]]

    def e_ss(self, u):
        return self.v_sss

    @property
    def ss(self):
        return self.e_ss(self.unit_vector)

    def e_spx(self, u):
        return u[0] * self.v_sps

    @property
    def spx(self):
        return self.e_spx(self.unit_vector)

    @property
    def pxs(self):
        return self.e_spx(self.unit_vector)

    @property
    def spy(self):
        return self.e_spx(self.unit_vector[[1,2,0]])

    @property
    def pys(self):
        return self.e_spx(self.unit_vector[[1, 2, 0]])

    @property
    def spz(self):
        return self.e_spx(self.unit_vector[[2,0,1]])

    @property
    def pzs(self):
        return self.e_spx(self.unit_vector[[2, 0, 1]])

    def e_pxpx(self, u):
        return u[0] ** 2 * self.v_pps + (1 - u[0] ** 2) * self.v_ppp

    @property
    def pxpx(self):
        return self.e_pxpx(self.unit_vector)

    @property
    def pypy(self):
        return self.e_pxpx(self.unit_vector[[1,2,0]])

    @property
    def pzpz(self):
        return self.e_pxpx(self.unit_vector[[2,0,1]])

    def e_pxpy(self, u):
        return u[0] * u[1] * (self.v_pps - self.v_ppp)

    @property
    def pxpy(self):
        return self.e_pxpy(self.unit_vector)

    @property
    def pypz(self):
        return self.e_pxpy(self.unit_vector[[1,2,0]])

    @property
    def pzpx(self):
        return self.e_pxpy(self.unit_vector[[2,0,1]])

    def e_pxpz(self, u):
        return u[0] * u[2] * (self.v_pps - self.v_ppp)

    @property
    def pxpz(self):
        return self.e_pxpz(self.unit_vector)

    @property
    def pypx(self):
        return self.e_pxpz(self.unit_vector[[1,2,0]])

    @property
    def pzpy(self):
        return self.e_pxpz(self.unit_vector[[2,0,1]])

    def e_sdxy(self, u):
        return np.sqrt(3) * u[0] * u[1] * self.v_sds

    @property
    def sdxy(self):
        return self.e_sdxy(self.unit_vector)

    @property
    def dxys(self):
        return self.e_sdxy(self.unit_vector)

    @property
    def sdyz(self):
        return self.e_sdxy(self.unit_vector[[1,2,0]])

    @property
    def dyzs(self):
        return self.e_sdxy(self.unit_vector[[1, 2, 0]])

    @property
    def sdzx(self):
        return self.e_sdxy(self.unit_vector[[2,0,1]])

    @property
    def dzxs(self):
        return self.e_sdxy(self.unit_vector[[2, 0, 1]])

    def e_sdx2(self, u):
        return 1 / 2 * np.sqrt(3) * (u[0] ** 2 - u[1] ** 2) * self.v_sds

    @property
    def sdx2(self):
        return self.e_sdx2(self.unit_vector)

    @property
    def dx2s(self):
        return self.e_sdx2(self.unit_vector)

    def e_sdz2(self, u):
        return (u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_sds

    @property
    def sdz2(self):
        return self.e_sdz2(self.unit_vector)

    @property
    def dz2s(self):
        return self.e_sdz2(self.unit_vector)

    def e_pxdxy(self, u):
        return np.sqrt(3) * u[0] ** 2 * u[1] * self.v_pds + u[1] * (1 - 2 * u[0] ** 2) * self.v_pdp

    @property
    def pxdxy(self):
        return self.e_pxdxy(self.unit_vector)

    @property
    def dxypx(self):
        return self.e_pxdxy(self.unit_vector)

    @property
    def pydyz(self):
        return self.e_pxdxy(self.unit_vector[[1,2,0]])

    @property
    def dyzpy(self):
        return self.e_pxdxy(self.unit_vector[[1, 2, 0]])

    @property
    def pzdzx(self):
        return self.e_pxdxy(self.unit_vector[[2,0,1]])

    @property
    def dzxpz(self):
        return self.e_pxdxy(self.unit_vector[[2, 0, 1]])

    def e_pxdyz(self, u):
        return  u[0] * u[1] * u[2] * (np.sqrt(3) *self.v_pds - 2 * self.v_pdp)

    @property
    def pxdyz(self):
        return self.e_pxdyz(self.unit_vector)

    @property
    def dyzpx(self):
        return self.e_pxdyz(self.unit_vector)

    @property
    def pydzx(self):
        return self.e_pxdyz(self.unit_vector[[1,2,0]])

    @property
    def dzxpy(self):
        return self.e_pxdyz(self.unit_vector[[1, 2, 0]])

    @property
    def pzdxy(self):
        return self.e_pxdyz(self.unit_vector[[2,0,1]])

    @property
    def dxypz(self):
        return self.e_pxdyz(self.unit_vector[[2, 0, 1]])

    def e_pxdzx(self, u):
        return u[2] * (np.sqrt(3) * u[0] ** 2 * self.v_pds + (1 - 2 * u[0] ** 2) * self.v_pdp)

    @property
    def pxdzx(self):
        return self.e_pxdzx(self.unit_vector)

    @property
    def dzxpx(self):
        return self.e_pxdzx(self.unit_vector)

    @property
    def pydxy(self):
        return self.e_pxdzx(self.unit_vector[[1,2,0]])

    @property
    def dxypy(self):
        return self.e_pxdzx(self.unit_vector[[1, 2, 0]])

    @property
    def pzdyz(self):
        return self.e_pxdzx(self.unit_vector[[2,0,1]])

    @property
    def dyzpz(self):
        return self.e_pxdzx(self.unit_vector[[2, 0, 1]])

    def e_pxdx2(self, u):
        return u[0] * (1 / 2 * np.sqrt(3) * (u[0] ** 2 - u[1] ** 2) * self.v_pds +
                       (1 - u[0] ** 2 + u[1] ** 2) * self.v_pdp)

    @property
    def pxdx2(self):
        return self.e_pxdx2(self.unit_vector)

    @property
    def dx2px(self):
        return self.e_pxdx2(self.unit_vector)

    def e_pydx2(self, u):
        return u[1] * (1 / 2 * np.sqrt(3) * (u[0] ** 2 - u[1] ** 2) * self.v_pds -
                       (1 + u[0] ** 2 - u[1] ** 2) * self.v_pdp)

    @property
    def pydx2(self):
        return self.e_pydx2(self.unit_vector)

    @property
    def dx2py(self):
        return self.e_pydx2(self.unit_vector)

    def e_pzdx2(self, u):
        return u[2] * (1 / 2 * np.sqrt(3) * (u[0] ** 2 - u[1] ** 2) * self.v_pds -
                       (u[0] ** 2 - u[1] ** 2) * self.v_pdp)

    @property
    def pzdx2(self):
        return self.e_pzdx2(self.unit_vector)

    @property
    def dx2pz(self):
        return self.e_pzdx2(self.unit_vector)

    def e_pxdz2(self, u):
        return u[0] * ((u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_pds -
                       np.sqrt(3) * u[2] ** 2 * self.v_pdp)

    @property
    def pxdz2(self):
        return self.e_pxdz2(self.unit_vector)

    @property
    def dz2px(self):
        return self.e_pxdz2(self.unit_vector)

    def e_pydz2(self, u):
        return u[1] * ((u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_pds -
                       np.sqrt(3) * u[2] ** 2 * self.v_pdp)

    @property
    def pydz2(self):
        return self.e_pydz2(self.unit_vector)

    @property
    def dz2py(self):
        return self.e_pydz2(self.unit_vector)

    def e_pzdz2(self, u):
        return u[2] * ((u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_pds +
                       np.sqrt(3) * (u[0] ** 2 + u[1] ** 2) * self.v_pdp)

    @property
    def pzdz2(self):
        return self.e_pzdz2(self.unit_vector)

    @property
    def dz2pz(self):
        return self.e_pzdz2(self.unit_vector)

    def e_dxydxy(self, u):
        return 3 * u[0] ** 2 * u[1] ** 2 * self.v_dds + \
               (u[0] ** 2 + u[1] ** 2 - 4 * u[0] ** 2 * u[1] ** 2) * self.v_ddp + \
               (u[2] ** 2 + u[0] ** 2 * u[1] ** 2) * self.v_ddd

    @property
    def dxydxy(self):
        return self.e_dxydxy(self.unit_vector)

    @property
    def dyzdyz(self):
        return self.e_dxydxy(self.unit_vector[[1,2,0]])

    @property
    def dzxdzx(self):
        return self.e_dxydxy(self.unit_vector[[2,0,1]])

    def e_dxydyz(self, u):
        return 3 * u[0] * u[1] ** 2 * u[2] * self.v_dds + \
               u[0] * u[2] * (1 - 4 * u[1] ** 2) * self.v_ddp + \
               u[0] * u[2] * (u[1] ** 2 - 1) * self.v_ddd

    @property
    def dxydyz(self):
        return self.e_dxydyz(self.unit_vector)

    @property
    def dyzdzx(self):
        return self.e_dxydyz(self.unit_vector[[1,2,0]])

    @property
    def dzxdxy(self):
        return self.e_dxydyz(self.unit_vector[[2,0,1]])

    def e_dxydzx(self, u):
        return 3 * u[0] ** 2 * u[1] * u[2] * self.v_dds + \
               u[1] * u[2] * (1 - 4 * u[0] ** 2) * self.v_ddp + \
               u[1] * u[2] * (u[0] ** 2 - 1) * self.v_ddd

    @property
    def dxydzx(self):
        return self.e_dxydzx(self.unit_vector)

    @property
    def dyzdxy(self):
        return self.e_dxydzx(self.unit_vector[[1,2,0]])

    @property
    def dzxdyz(self):
        return self.e_dxydzx(self.unit_vector[[2,0,1]])

    def e_dxydx2(self, u):
        return u[0] * u[1] * (u[0] ** 2 - u[1] ** 2) * (3 / 2 * self.v_dds - 2 * self.v_ddp + 1 / 2 * self.v_ddd)

    @property
    def dxydx2(self):
        return self.e_dxydx2(self.unit_vector)

    @property
    def dx2dxy(self):
        return self.e_dxydx2(self.unit_vector)

    def e_dyzdx2(self, u):
        return u[1] * u[2] * (3 / 2 * (u[0] ** 2 - u[1] ** 2) * self.v_dds
                              - (1 + 2 * (u[0] ** 2 - u[1] ** 2)) * self.v_ddp
                              + (1 + 1 / 2 * (u[0] ** 2 - u[1] ** 2)) * self.v_ddd)

    @property
    def dyzdx2(self):
        return self.e_dyzdx2(self.unit_vector)

    @property
    def dx2dyz(self):
        return self.e_dyzdx2(self.unit_vector)

    def e_dzxdx2(self, u):
        return u[0] * u[2] * (3 / 2 * (u[0] ** 2 - u[1] ** 2) * self.v_dds
                              + (1 - 2 * (u[0] ** 2 - u[1] ** 2)) * self.v_ddp
                              - (1 - 1 / 2 * (u[0] ** 2 - u[1] ** 2)) * self.v_ddd)

    @property
    def dzxdx2(self):
        return self.e_dzxdx2(self.unit_vector)

    @property
    def dx2dzx(self):
        return self.e_dzxdx2(self.unit_vector)

    def e_dxydz2(self, u):
        return np.sqrt(3) * u[0] * u[1] * ((u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_dds
                                           - 2 * u[2] ** 2 * self.v_ddp
                                           + 1 / 2 * (1 + u[2] ** 2) * sk.v_ddd)

    @property
    def dxydz2(self):
        return self.e_dxydz2(self.unit_vector)

    @property
    def dz2dxy(self):
        return self.e_dxydz2(self.unit_vector)

    def e_dyzdz2(self, u):
        return np.sqrt(3) * u[1] * u[2] * ((u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_dds
                                           + (u[0] ** 2 + u[1] ** 2 - u[2] ** 2) * self.v_ddp
                                           - 1 / 2 *  (u[0] ** 2 + u[1] ** 2) * sk.v_ddd)

    @property
    def dyzdz2(self):
        return self.e_dyzdz2(self.unit_vector)

    @property
    def dz2dyz(self):
        return self.e_dyzdz2(self.unit_vector)

    def e_dzxdz2(self, u):
        return np.sqrt(3) * u[0] * u[2] * ((u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) * self.v_dds
                                           + (u[0] ** 2 + u[1] ** 2 - u[2] ** 2) * self.v_ddp
                                           - 1 / 2 *  (u[0] ** 2 + u[1] ** 2) * sk.v_ddd)

    @property
    def dzxdz2(self):
        return self.e_dzxdz2(self.unit_vector)

    @property
    def dz2dzx(self):
        return self.e_dzxdz2(self.unit_vector)

    def e_dx2dx2(self, u):
        return 3 / 4 * (u[0] ** 2 - u[1] ** 2) ** 2 * self.v_dds + \
               (u[0] ** 2 + u[1] ** 2 -(u[0] ** 2 - u[1] ** 2) ** 2) * self.v_ddp + \
               (u[2] ** 2 + 1 / 4 * (u[0] ** 2 - u[1] ** 2) ** 2) * self.v_ddd

    @property
    def dx2dx2(self):
        return self.e_dx2dx2(self.unit_vector)

    def e_dx2dz2(self, u):
        return 1 / 2 * np.sqrt(3) * (u[0] ** 2 - u[1] ** 2) * (u[2] ** 2 - 1 / 2 * (u[0] ** 2 + u[1] ** 2)) \
                 * self.v_dds + \
               np.sqrt(3) * u[2] ** 2 * (u[1] ** 2 - u[0] ** 2) * self.v_ddp + \
               1 / 4 * np.sqrt(3) * (1 + u[2] ** 2) * (u[0] ** 2 - u[1] ** 2) * self.v_ddd

    @property
    def dx2dz2(self):
        return self.e_dx2dz2(self.unit_vector)

    @property
    def dz2dx2(self):
        return self.e_dx2dz2(self.unit_vector)

    def e_dz2dz2(self, u):
        return (u[2] ** 2 - 1/2 * (u[0] ** 2 + u[1] ** 2)) ** 2 * self.v_dds + \
               3 * u[2] ** 2 * (u[0] ** 2 + u[1] ** 2) * self.v_ddp + \
               3 / 4 * (u[0] ** 2 + u[1] ** 2) ** 2 * self.v_ddd

    @property
    def dz2dz2(self):
        return self.e_dz2dz2(self.unit_vector)

    @property
    def integrals(self):
        return np.array([
            [self.ss, self.spx, self.spy, self.spz, self.sdz2, self.sdxy, self.sdx2, self.sdyz, self.sdzx],
            [self.pxs, self.pxpx, self.pxpy, self.pxpz, self.pxdz2, self.pxdxy, self.pxdx2, self.pxdyz, self.pxdzx],
            [self.pys, self.pypx, self.pypy, self.pypz, self.pydz2, self.pydxy, self.pydx2, self.pydyz, self.pydzx],
            [self.pzs, self.pzpx, self.pzpy, self.pzpz, self.pzdz2, self.pzdxy, self.pzdx2, self.pzdyz, self.pzdzx],
            [self.dz2s, self.dz2px, self.dz2py, self.dz2pz, self.dz2dz2, self.dz2dxy, self.dz2dx2, self.dz2dyz,
             self.dz2dzx],
            [self.dxys, self.dxypx, self.dxypy, self.dxypz, self.dxydz2, self.dxydxy, self.dxydx2, self.dxydyz,
             self.dxydzx],
            [self.dx2s, self.dx2px, self.dx2py, self.dx2pz, self.dx2dz2, self.dx2dxy, self.dx2dx2, self.dx2dyz,
             self.dx2dzx],
            [self.dyzs, self.dyzpx, self.dyzpy, self.dyzpz, self.dyzdz2, self.dyzdxy, self.dyzdx2, self.dyzdyz,
             self.dyzdzx],
            [self.dzxs, self.dzxpx, self.dzxpy, self.dzxpz, self.dzxdz2, self.dzxdxy, self.dzxdx2, self.dzxdyz,
             self.dzxdzx],
        ])