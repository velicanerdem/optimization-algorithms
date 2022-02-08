import numpy as np
import sys
sys.path.append("..")


# import the test classes

import unittest


from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced

from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2


from solution import Solver0

class testSolver0(unittest.TestCase):
    """
    test on problem A
    """
    Solver = Solver0

    def testConstructor(self):
        """
        check the constructor
        """
        solver = self.Solver()

    def testConvergence(self):
        """
        check that student solver converges
        """
        problem = MathematicalProgramTraced(QuadraticIdentity2())
        solver = self.Solver()
        solver.setProblem((problem))
        output =  solver.solve()
        print(problem.trace_x)
        last_trace = problem.trace_x[-1]
        # check that we have made some progress toward the optimum
        self.assertTrue( np.linalg.norm( np.zeros(2) - last_trace  ) < .9)




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






