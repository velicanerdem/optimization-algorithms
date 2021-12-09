import unittest
import math
import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT

from solution import AntennaPlacement


class testProblem(unittest.TestCase):
    """
    test AntennaPlacement
    """

    problem = AntennaPlacement

    def generateProblem(self):
        P = [np.array([0, 0]), np.array([1, 0])]
        w = np.array([1, 0.5])
        problem = self.problem(P, w)
        return problem

    def testConstructor(self):
        self.generateProblem()

    def testValue(self):
        problem = self.generateProblem()
        x = np.ones(2)
        value = problem.evaluate(x)[0][0]
        self.assertAlmostEqual(value, - 1 * math.exp(- 2) - .5 * math.exp(-1))

    def testJacobian(self):
        problem = self.generateProblem()
        x = np.array([-1, .5])
        flag, _, _ = check_mathematical_program(problem.evaluate, x, 1e-5)
        self.assertTrue(flag)

    def testHessian(self):
        problem = self.generateProblem()
        x = np.array([-1, .1])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        self.assertTrue(flag)


# usage:
# print results in terminal
# python3 test.py
# store results in file
# python3 test.py out.log

if __name__ == "__main__":
    if len(sys.argv) == 2:
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
            runner = unittest.TextTestRunner(f, verbosity=2)
            unittest.main(testRunner=runner)
    else:
        unittest.main()
