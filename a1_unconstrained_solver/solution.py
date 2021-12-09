import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverUnconstrained(NLPSolver):

    def __init__(self, alpha=0.1):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        self.alpha = alpha
        self.step_increase = 1.2
        self.stepsize_decrement = 0.5
        self.minimum_desired_decrease = 0.01
        # in case you want to initialize some class members or so...
        
    # Descent methods
    
    def _gradient_descent(self, x):
        phi, J = self.problem.evaluate(x)
        phi = phi[0]
        J = J[0]
        return x - self.alpha * J, [phi, J]
        
    def _newtons_method(self, x):
        phi, J = self.problem.evaluate(x)
        phi = phi[0]
        J = J[0]
        H = self.problem.getFHessian(x)  # if necessary
        return x - np.linalg.inv(H) @ J, [phi, J, H]


    def _evaluate_cost(self, x):
        cost, _ = self.problem.evaluate(x)
        cost = cost[0]
        
        return cost
    
    def _evaluate_cost_jacobian(self, x):
        cost, jacobian = self.problem.evaluate(x)
        cost, jacobian = cost[0], jacobian[0]
    
        return cost, jacobian
        
    # Line search

    def _backtracking(self, x):
        alpha = 1
        
        phi, jacobian = self._evaluate_cost_jacobian(x)
        jacobian_norm = np.linalg.norm(jacobian)
        
        if jacobian_norm == 0:
            return x
        
        delta_max = pow(2, 32)
        
        end_search = alpha * jacobian_norm
        end_search_threshold = 0.01
        
        for _ in range(10):
            delta = - jacobian / jacobian_norm
            
            next_val = x + alpha * delta
            next_cost = self._evaluate_cost(next_val)
            estimated_step_value = self.minimum_desired_decrease * alpha * np.dot(jacobian, delta)
            desired_minimum_cost = phi + estimated_step_value
            
            while next_cost > desired_minimum_cost:
                alpha *= self.stepsize_decrement
                next_val = x + alpha * delta                
                next_cost = self._evaluate_cost(next_val)
                estimated_step_value = self.minimum_desired_decrease * alpha * np.dot(jacobian, delta)
                desired_minimum_cost = phi + estimated_step_value
            
            x = x + alpha * np.dot(jacobian, delta)
            delta = min(self.step_increase * alpha, delta_max)
            
            phi, jacobian = self._evaluate_cost_jacobian(x)
            jacobian_norm = np.linalg.norm(jacobian)
            
            if jacobian_norm == 0:
                return x
            
            end_search = alpha * jacobian_norm
            
        return x

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """

        # write your code here

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()
        # get feature types
        # ot[i] indicates the type of feature i (either OT.f or OT.sos)
        # there is at most one feature of type OT.f
        ot = self.problem.getFeatureTypes()

        line_method = self._backtracking

        if 2 in ot:
            descent_method = self._gradient_descent
        else:
            descent_method = self._newtons_method
        
        # phi is a vector (1D np.array); J is a Jacobian matrix (2D np.array).
        # use the following to get an initialization:
        x = self.problem.getInitializationSample()
        x = np.array(x, dtype=np.float64)
        
        # now code some loop that iteratively queries the problem and updates x till convergence        
        it_amount = 10000
        
        for i in range(it_amount):
            x = line_method(x)
            x, _ = descent_method(x)
                
        return x