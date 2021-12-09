import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram


class Problem0( MathematicalProgram ):
    """
    """

    def __init__(self,C):

        # in case you want to initialize some class members or so...


    def evaluate(self, x) :
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # y = ...
        # J = ...

        # and return as a tuple of arrays, namely of dim (1) and (1,n)
        #return  np.array( [ y ] ) ,  J.reshape(1,-1)

    def getDimension(self) : 
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem, e.g.
        return 2

    def getFHessian(self, x) : 
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix

        # H = ...

        # and return it
        #return H

    def getInitializationSample(self) : 
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self , verbose ): 
        """
        See Also
        ------
        MathematicalProgram.report
        """
        return "Quadratic function x C^T C x "
