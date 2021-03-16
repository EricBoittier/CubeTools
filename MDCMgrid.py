import numpy as np
from numba import jit, njit, prange
from Cube import read_charges_refined, read_cube
from VDW import *


@njit(cache=True, parallel=True)
def in_interaction_belt(positions, _xyz, _vdw):
    size_x = _xyz.shape[2]
    size_y = _xyz.shape[1]
    size_z = _xyz.shape[3]
    num_atoms = len(_vdw)

    output = np.zeros((size_x, size_y, size_z))

    for i in prange(size_x):
        for j in prange(size_y):
            for k in prange(size_z):

                distances = []
                r = _xyz[:, j, i, k]
                for n in range(num_atoms):
                    rr = distance(r, positions[n])
                    distances.append(rr / _vdw[n])

                rmin = min(distances)

                if not rmin >= 1.2:
                    output[i, j, k] = 0  # False
                elif not rmin <= 2.20:
                    output[i, j, k] = 0  # False
                else:
                    output[i, j, k] = 1  # True

    return output


@jit(nopython=True, cache=True)
def jit_columbic_np(a, b, c):
    b = np.asarray([[b[0]] * 13, [b[1]] * 13, [b[2]] * 13])
    difference = a - b
    diff_sqr = np.square(difference)
    sum_diff = np.sum(diff_sqr, axis=0)  # sum (dif x , dif y, dif z) = (sum)
    sqr_sum_diff = np.sqrt(sum_diff)
    columbic = c / sqr_sum_diff
    columbic = np.sum(columbic)
    return columbic


@njit(cache=True, parallel=True)
def calculate_coulombic_grid(_xyz, _positions_np, _charges_np):
    size_x = _xyz.shape[2]
    size_y = _xyz.shape[1]
    size_z = _xyz.shape[3]

    output = np.zeros((size_x, size_y, size_z))
    for i in prange(size_x):
        for j in prange(size_y):
            for k in prange(size_z):
                r = _xyz[:, j, i, k]
                output[i, j, k] = jit_columbic_np(_positions_np, r, _charges_np)
    return output


@jit(nopython=True, cache=True)
def RMSE_in_kcal(output, ref):
    diff = output - ref
    rmse = np.sqrt(np.square(diff.flatten()).sum() / (diff.shape[0] * diff.shape[1] * diff.shape[2]))
    return rmse * HARTREE_TO_KCAL


@jit(nopython=True, cache=True)
def distance(output, ref):
    diff = output - ref
    rmse = np.sqrt(np.square(diff.flatten()).sum())
    return rmse


@jit(nopython=True, cache=True)
def RMSE_in_kcal_in_belt(output, ref, belt):
    diff = np.multiply(output, belt) - np.multiply(ref, belt)
    rmse = np.sqrt(np.square(diff.flatten()).sum() / belt.sum())
    return rmse * HARTREE_TO_KCAL


@jit(nopython=True, cache=True)
def RMSE(output, ref):
    diff = output - ref
    rmse = np.sqrt(np.square(diff.flatten()).sum() / (diff.shape[0] * diff.shape[1] * diff.shape[2]))
    return rmse * HARTREE_TO_KCAL


class MDCM_cube_comparison():
    def __init__(self, charges_path, pcube):
        self.charges_path = charges_path
        self.pos_charges_np = read_charges_refined(self.charges_path)
        self.positions_np = self.pos_charges_np[:, 0:3]
        self.positions_np = self.positions_np.T
        self.charges_np = self.pos_charges_np[:, -1]

        self.pcube = pcube
        self.pcube_data, self.pcube_meta = read_cube(self.pcube)
        self.pcube_atoms = self.pcube_meta["atoms"]
        self.org = list(self.pcube_meta["org"])
        self.xvec = list(self.pcube_meta["xvec"])[0]
        self.yvec = list(self.pcube_meta["yvec"])[1]
        self.zvec = list(self.pcube_meta["zvec"])[2]
        self.size_x = self.pcube_data.shape[0]
        self.size_y = self.pcube_data.shape[1]
        self.size_z = self.pcube_data.shape[2]

        self.x_values = np.linspace(self.org[0], self.org[0] + self.xvec * self.size_x, num = self.size_x)
        self.y_values = np.linspace(self.org[1], self.org[1] + self.yvec * self.size_y, num = self.size_y)
        self.z_values = np.linspace(self.org[2], self.org[2] + self.zvec * self.size_z, num = self.size_z)

        assert len(self.x_values ) == self.pcube_data.shape[0]

        self.xx, self.yy, self.zz = np.meshgrid(self.x_values, self.y_values, self.z_values, indexing="ij")
        self.xyz = np.array(np.meshgrid(self.x_values, self.y_values, self.z_values, indexing="xy"))

        self.read_map = [np.fromiter(x[1], dtype=np.float) for x in self.pcube_atoms]
        self.atom_posistions = [x[1:] for x in self.read_map]
        self.atom_posistions = np.array(self.atom_posistions, dtype=np.float64)
        self.num_atoms = len(self.atom_posistions)
        self.atoms = [int(x[0]) for x in self.read_map]
        self.known_VDW = [VDWs[x] for x in self.atoms]
        self.known_VDW = np.array(self.known_VDW, dtype=np.float64)
        self.interaction_belt = in_interaction_belt(self.atom_posistions, self.xyz, self.known_VDW)

        print("Base Error: Testing RMSE() on my cube, ref. pcube", self.get_error_from_positions(self.positions_np))

    def get_error_from_positions(self, positions_np):
        output = calculate_coulombic_grid(self.xyz,
                                          positions_np, self.charges_np)
        assert output.shape == self.pcube_data.shape
        error = RMSE_in_kcal_in_belt(output, self.pcube_data, self.interaction_belt)
        return error

    def objective(self, x):
        for i in range(len(x)):
            self.positions_np[i % 3, i // 3] = x[i]
        return self.get_error_from_positions(self.positions_np)
