from scipy import optimize
import pickle
from MDCMgrid import *
import sys
import numpy as np

class out():
    def __init__(self, x=None, from_pickle=None):
        if from_pickle:
            self.x = pickle.load(open(from_pickle, "rb")).x
        else:
            if x is None:
                print("Warning: no x values")
            else:
                self.x = np.array(x).flatten()
    
    def ignore(self, indices):
        tmp_x = []
        for i in range(len(self.x)):
            if not i // 3 in indices:
                tmp_x.append(self.x[i])
        self.x = tmp_x

def optimize_and_pickle(mdcm, out_pickle, out, eps=None, maxiter=None):
    if eps is None:
        eps = 1.4901161193847656e-05
    if maxiter is None:
        maxiter = 1
    out = optimize.minimize(mdcm.objective, [*out.x], args=(), method='BFGS',
                            jac=None, tol=None, callback=None,
                            options={'gtol': 1e-06, #'norm': np.inf,
                                     'eps': eps,
                                     'maxiter': maxiter, 'disp': True,
                                     'return_all': True})
    print(out)
    pickle.dump(out, open("{}".format(out_pickle), "wb"))


if __name__ == '__main__':
    charges_path = sys.argv[1]
    pcube = sys.argv[2]
    in_pickle = sys.argv[3]
    out_pickle = sys.argv[4]
    print("Program Starting: BFGS Optimization")
    eps = None
    maxiter = None
    ignore_indices= None
    if len(sys.argv) > 5:
        eps = float(sys.argv[5])
    if len(sys.argv) > 6:
        maxiter = float(sys.argv[6])
    if len(sys.argv) > 7:
        ignore_indices = [int(x) for x in sys.argv[7].split("_")]

    mdcm = MDCM_cube_comparison(charges_path, pcube)
    if ignore_indices is not None:
        mdcm.set_ignore_indices(ignore_indices)

    if in_pickle == "False":
        print(f"Setting x as mdcm.positions_np")
        out = out(x=mdcm.positions_np)
        print("Before ignore: ")
        print(out.x)
        out.ignore(ignore_indices)
    else:
        print(f"Setting x from in_pickle {in_pickle}")
        out = out(from_pickle=in_pickle)
    
    print(out.x) 
    #sys.exit(0)
    optimize_and_pickle(mdcm, out_pickle, out, eps=eps, maxiter=maxiter)


