import numpy as np
import unittest
import sys
sys.path.append("..")



# import the test classes



from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.mathematical_program import MathematicalProgram

from solution import Problem0

class testProblem(unittest.TestCase):
    """
    test on problem A
    """

    problem = Problem0

    def testValue(self):
        C = np.ones((2,2))
        problem = self.problem(C)
        value = problem.evaluate(np.ones(2))[0][0]
        self.assertAlmostEqual(value,8)


    def testJacobian(self):
        """
        """
        C = np.ones((2,2))
        problem = self.problem(C)
        flag , _ , _= check_mathematical_program(problem.evaluate, np.array([-1,.5])  , 1e-5)
        self.assertTrue(flag)


    def testHessian(self):

        C = np.ones((2,2))
        problem = self.problem(C)
        x = np.array([-1, .1])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f,x,tol) 
        flag = np.allclose( H , Hdiff, 10*tol, 10*tol)
        self.assertTrue(flag)


# usage:
# print results in terminal
# python3 test.py
# store results in file 
# python3 test.py out.log

if __name__ == "__main__":
    if len(sys.argv) == 2 :
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
           runner = unittest.TextTestRunner(f, verbosity=2)
           unittest.main(testRunner=runner)
    else:
        unittest.main()



