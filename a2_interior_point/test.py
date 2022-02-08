import sys
import unittest
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.mathematical_programs.hs071 import Hs071
from optimization_algorithms.mathematical_programs.quadratic_program import QuadraticProgram
from optimization_algorithms.mathematical_programs.linear_program_ineq import LinearProgramIneq
from optimization_algorithms.mathematical_programs.logistic_bounds import LogisticWithBounds

# from solution import SolverInteriorPoint as Solver
from solution import SolverInteriorPoint as Solver


class testSolver(unittest.TestCase):
    """
    test SolverUnconstrained
    """
    Solver = Solver

    def set_(self, solver):
        self.Solver = solver

    def testConstructor(self):
        """
        check the constructor
        """
        solver = self.Solver()

    def testLogisticBounds(self):
        """
        """
        problem = MathematicalProgramTraced(
            LogisticWithBounds(), max_evaluate=100000000)
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        buu = np.array([1, 2, .5])
        solution = np.array([2, 2, 1.0369])
        self.assertTrue(np.linalg.norm(solution - last_trace) < .01)
        print("solution")
        print(problem.evaluate(solution))
        print("buu")
        print(problem.evaluate(buu))

    def testQuadraticIneq(self):
        """
        """
        H = np.array([[1., -1.], [-1., 2.]])
        g = np.array([-2., -6.])
        Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
        bineq = np.array([2., 2., 3.])
        problem = MathematicalProgramTraced(
            QuadraticProgram(H=H, g=g, Aineq=Aineq, bineq=bineq))

        problem.getInitializationSample = lambda: np.zeros(2)
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        solution = np.array([0.6667, 1.3333])
        self.assertTrue(np.linalg.norm(solution - last_trace) < .01)

    def testLinearProgramIneq(self):
        """
        """
        n = 4
        problem = MathematicalProgramTraced(LinearProgramIneq(n))
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        self.assertTrue(np.linalg.norm(np.zeros(n) - last_trace) < .01)


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
