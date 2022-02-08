import unittest
import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT
from solution import LQR as Problem


class testProblem(unittest.TestCase):
    """
    test RobotTool
    """

    problem = Problem

    def set_(self, problem):
        self.problem = problem

    def generateProblem(self):
        K = 11
        A = np.identity(2)
        B = 1.5 * np.identity(2)
        Q = 1.8 * np.identity(2)
        R = 1.9 * np.identity(2)
        yf = 2 * np.ones(2)
        return self.problem(K, A, B, Q, R, yf) , K

    def generateProblemB(self):
        K = 4
        A = np.identity(3)
        B = np.identity(3)
        Q = np.identity(3)
        R = np.identity(3)
        yf = np.ones(3)
        return self.problem(K, A, B, Q, R, yf) , K

    def test_value_B(self):
        problem , K = self.generateProblemB()
        x = np.zeros(problem.getDimension())

        # this only modifies first component of x at each time step
        csol = 0
        for i in range(K):
            x[i * 2 * 3 + 3] = (i + 1) / K
            csol += 0.5 * x[i * 2 * 3 + 3] ** 2

        types = problem.getFeatureTypes()
        f_ind = [i for i in range(len(types)) if types[i] == OT.f]
        eq_inds = [i for i in range(len(types)) if types[i] == OT.eq]

        phi, J = problem.evaluate(x)
        phi_eqs = phi[eq_inds]
        # count the number of fulfilled constraints
        num_zeros = len([c for c in phi_eqs if abs(c - 0) < 1e-10])

        c = phi[f_ind[0]]

        eq_inds = [i for i in range(len(types)) if types[i] == OT.eq]
        cons = phi[eq_inds]
        max_abs = 0
        sum_abs = 0
        for con in cons:
            if abs(con) > max_abs:
                max_abs = abs(con)
            sum_abs += abs(con)
        self.assertAlmostEqual(c, csol)
        self.assertAlmostEqual(num_zeros, 9)
        self.assertAlmostEqual(sum_abs, .25 * 4 + 2 * 1)
        self.assertAlmostEqual(max_abs, 1)

    def test_value_C(self):
        problem, K = self.generateProblemB()
        x = np.zeros(problem.getDimension())

        csol = 0
        # we modify the first component of control and state
        for i in range(K):
            x[i * 2 * 3 + 3] = (i + 1) / K
            x[i * 2 * 3] = i
            csol += 0.5 * x[i * 2 * 3 + 3] ** 2
            csol += 0.5 * x[i * 2 * 3] ** 2

        phi, J = problem.evaluate(x)
        types = problem.getFeatureTypes()
        f_ind = [i for i in range(len(types)) if types[i] == OT.f]
        eq_inds = [i for i in range(len(types)) if types[i] == OT.eq]

        c = phi[f_ind[0]]
        cons = phi[eq_inds]
        max_abs = 0
        sum_abs = 0
        for con in cons:
            if abs(con) > max_abs:
                max_abs = abs(con)
            sum_abs += abs(con)
        self.assertAlmostEqual(c, csol)
        self.assertAlmostEqual(sum_abs, .25 + .75 + 1.75 + 2.75 + 2)
        self.assertAlmostEqual(max_abs, 2.75)

    def test_value_A(self):
        problem, K = self.generateProblem()
        x = problem.getInitializationSample()
        phi, J = problem.evaluate(x)
        types = problem.getFeatureTypes()
        f_ind = [i for i in range(len(types)) if types[i] == OT.f]
        c = phi[f_ind[0]]
        eq_inds = [i for i in range(len(types)) if types[i] == OT.eq]
        cons = phi[eq_inds]
        max_abs = 0
        sum_abs = 0
        for con in cons:
            sum_abs += abs(con)
            if abs(con) > max_abs:
                max_abs = abs(con)
        self.assertAlmostEqual(sum_abs, 4)
        self.assertAlmostEqual(c, 0)
        self.assertAlmostEqual(max_abs, 2)

    def testJacobian(self):
        problem, K = self.generateProblem()
        x = problem.getInitializationSample()
        phi, J = problem.evaluate(x)
        flag, _, _ = check_mathematical_program(problem.evaluate, x, 1e-5)
        self.assertTrue(flag)

    def testHessian(self):
        problem , K = self.generateProblem()
        x = problem.getInitializationSample()
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
