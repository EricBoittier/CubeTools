from scipy import optimize
import pickle
from MDCMgrid import *
import sys


class out():
    def __init__(self, x=None, from_pickle=None):
        if from_pickle:
            self.x = pickle.load(open(from_pickle, "rb")).x
        else:
            if x is None:
                print("Warning: no x values")
            else:
                self.x = x


num = 3

if num != 0:
    in_pickle = "butadiene_{}.p".format(num)
    out = out(from_pickle=in_pickle)

charges_path = "charges/butadiene/13-charges/13_charges_refined.xyz"
pcube = "cubes/butadiene/scan_extract_{}.xyz.chk.fchk.pot.cube".format(num)

mdcm = MDCM_cube_comparison(charges_path, pcube)

if num != 0:
    mdcm.set_positions(out.x)
ERROR = mdcm.get_error() * (0.58/0.7452557819163884)
print(mdcm.get_error() * (0.58/0.7452557819163884))

atom_x = []
atom_y = []
atom_z = []
for x in mdcm.atom_posistions:
    atom_x.append(x[0])
    atom_y.append(x[1])
    atom_z.append(x[2])

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

data = mdcm.positions_np
charges_np = mdcm.charges_np

p = ax.scatter3D(data[0], data[1], data[2], c=charges_np, cmap='bwr')
for i in range(len(data[0])):
    ax.text(data[0][i], data[1][i], data[2][i], "c{}".format(i+1), color='k', size=8)


ax.scatter3D(atom_x, atom_y, atom_z, c='k', s=50)

ax.set_xlabel("x [Bohr]")
ax.set_ylabel("y [Bohr]")
ax.set_zlabel("z [Bohr]")

atom_names = {}
atom_names[1] = "H"
atom_names[6] = "C"


for i in range(len(mdcm.atoms)):
    ax.text(atom_x[i], atom_y[i], atom_z[i], "{}".format(atom_names[mdcm.atoms[i]]), color='k')

ax.set_title("Scan {}, Error = {:.2f}".format(num, ERROR))


fig.colorbar(p)
plt.show()



