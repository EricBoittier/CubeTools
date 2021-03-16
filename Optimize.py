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


def optimize_and_pickle(mdcm, out_pickle, out, eps=None, maxiter=None):
    if eps is None:
        eps = 1.4901161193847656e-05
    if maxiter is None:
        maxiter = 1
    out = optimize.minimize(mdcm.objective, [*out.x], args=(), method='BFGS',
                            jac=None, tol=None, callback=None,
                            options={'gtol': 1e-05, 'norm': np.inf,
                                     'eps': eps,
                                     'maxiter': maxiter, 'disp': False,
                                     'return_all': False})
    print(out)
    pickle.dump(out, open("{}".format(out_pickle), "wb"))

if __name__ == '__main__':
    charges_path = sys.argv[1]
    pcube = sys.argv[2]
    in_pickle = sys.argv[3]
    out_pickle = sys.argv[4]
    
    eps = None
    maxiter = None
    if len(sys.argv) > 5:
    	eps=sys.argv[5]
    if len(sys.argv) > 6:
	maxiter=sys.argv[6]   
    
    out = out(from_pickle="out.p")
    mdcm = MDCM_cube_comparison(charges_path, pcube)

    optimize_and_pickle(mdcm, out_pickle, out, eps=eps, maxiter=maxiter)

