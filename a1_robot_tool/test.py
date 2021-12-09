import unittest
import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT
from solution import RobotTool


class testProblem(unittest.TestCase):
    """
    test RobotTool
    """

    problem = RobotTool

    def generateProblem(self):
        q0 = np.zeros(3)
        pr = np.array([.5, 2.0 / 3.0])
        l = .5
        problem = self.problem(q0, pr, l)
        return problem

    def testConstructor(self):
        self.generateProblem()

    def testValue1(self):
        problem = self.generateProblem()
        # in this configuration, p = pr
        # todo: test the cost
        x = np.pi / 180.0 * np.array([90, -90, -90])
        phi, J = problem.evaluate(x)
        c = problem.l * (x - problem.q0) @ (x - problem.q0)
        self.assertAlmostEqual(c, phi @phi)

    def testValue2(self):
        problem = self.generateProblem()
        # in this configuration, q = q0
        # todo: test the cost
        x = np.zeros(3)
        phi, J = problem.evaluate(x)
        e = np.array([1.5 + 1. / 3. - .5, -2. / 3.])
        c = e @ e
        self.assertAlmostEqual(c, phi @phi)

    def testJacobian(self):
        problem = self.generateProblem()
        x = np.array([-1, .5, 1])
        flag, _, _ = check_mathematical_program(problem.evaluate, x, 1e-5)
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
