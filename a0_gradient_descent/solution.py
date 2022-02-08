import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class Solver0(NLPSolver):

    def __init__(self, alpha=0.1):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        self.alpha = alpha
        # in case you want to initialize some class members or so...


    def solve(self) :
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        # write your code here

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()

        # use the following to query the problem:
        # phi is a vector (1D np.array); use phi[0] to access the cost value (a float number). J is a Jacobian matrix (2D np.array). Use J[0] to access the gradient (1D np.array) of the cost value.
        
        # now code some loop that iteratively queries the problem and updates x till convergence
        phi, J = self.problem.evaluate(x)
        cost = phi[0]
        gradient = J[0]
        
        it_amount = 100
        cost_vals = np.arange(it_amount+1)
        cost_vals[0] = cost
        
        for i in range(1, it_amount+1):
            x -= self.alpha * gradient
            phi, J = self.problem.evaluate(x)
            cost = phi[0]
            gradient = J[0]
            cost_vals[i] = cost
        
        # finally:
        return x
