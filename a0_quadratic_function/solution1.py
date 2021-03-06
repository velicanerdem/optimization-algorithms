import numpy as np
import sys
import warnings

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT
from utility import show_data 

class SolverUnconstrained(NLPSolver):

    def __init__(self, alpha=0.1):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        self._alpha = alpha
        
        self._step_increase = 1.2
        self._stepsize_decrement = 0.5
        self._minimum_desired_decrease = 0.01
        # in case you want to initialize some class members or so...
        self._maximum_cost = 1e-7
        
    # Descent methods
    
    def _gradient_descent(self, x):
        phi, J = self._evaluate_cost_jacobian(x)
        
        return x - self._alpha * J, [phi, J]
        
    def _sos_gradient(self, x):
    
        phi, J = self.problem.evaluate(x)
        
        # print(np.sum(phi, axis=0))
        # input()

        return x - 0.01 * J.T @ phi, [phi, J]

    def _newtons_method(self, x):
        phi, J = self._evaluate_cost_jacobian(x)
        
        H = self.problem.getFHessian(x)  # if necessary
        delta = np.linalg.inv(H) @ J
        return x - delta, [phi, J, H]

    def _evaluate_cost(self, x):
        cost, _ = self.problem.evaluate(x)
        
        return cost[0]
    
    def _evaluate_cost_sos_total(self, x):
        cost, _ = self.problem.evaluate(x)
        cost = np.dot(cost, cost)

        return cost
    
    def _evaluate_cost_jacobian(self, x):
        cost, jacobian = self.problem.evaluate(x)
    
        return cost[0], jacobian[0]
    
    def _evaluate_cost_jacobian_sos(self, x):
        cost, jacobian = self.problem.evaluate(x)
        cost_summed = np.dot(cost, cost)
        sos_gradient = jacobian.T @ cost
    
        return cost_summed, sos_gradient
    
    def _get_sos_cost(self, phi):
        return np.dot(phi, phi)
    
    def _line_search(self, x, evaluate_cost, evaluate_cost_jacobian):        
        alpha = 1
        
        phi, jacobian = evaluate_cost_jacobian(x)
        jacobian_norm = np.linalg.norm(jacobian)
        
        delta_max = pow(2, 32)
        
        for _ in range(10):
            delta = - jacobian / jacobian_norm
            next_val = x + alpha * delta
            next_cost = evaluate_cost(next_val)
            estimated_step_value = self._minimum_desired_decrease * alpha * np.dot(jacobian, delta)
            desired_minimum_cost = phi + estimated_step_value
            
            while next_cost > desired_minimum_cost:
                alpha *= self._stepsize_decrement
                next_val = x + alpha * delta                
                next_cost = evaluate_cost(next_val)
                estimated_step_value = self._minimum_desired_decrease * alpha * np.dot(jacobian, delta)
                desired_minimum_cost = phi + estimated_step_value
            
            x = x + alpha * np.dot(jacobian, delta)
            delta = min(self._step_increase * alpha, delta_max)
            
            phi, jacobian = evaluate_cost_jacobian(x)
            jacobian_norm = np.linalg.norm(jacobian)
            
        return x, [phi, jacobian]
    
    # Write line search for NonLinearSOS
    
    def _identity(self, **args):
        return x

    def _try_solution(self, x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, is_sos):
        x = x_init
        x_before = x_init
        try:
            for _ in range(it_per_try):
                for i in range(len(methods)):
                    x_before = x
                    method = methods[i]
                    # Start with line search if possible
                    if methods_param_total[i] == 3:
                        x, phi_jacobian = method(x, evaluate_cost, evaluate_cost_jacobian)
                    elif methods_param_total[i] == 1:
                        x, phi_jacobian = method(x)
                    else:
                        raise Exception("Wrong number of arguments")
                    if is_sos:
                        cost = self._get_sos_cost(phi_jacobian[0])
                        if cost < self._maximum_cost:
                            return x, True
                    else:
                        if phi_jacobian[0] < self._maximum_cost:
                            return x, True
        except:
            return x_before, False
        finally:
            return x, False
    
    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        # write your code here
        warnings.filterwarnings('error')            
        # use the following to get an initialization:
        x_init = self.problem.getInitializationSample()
        x_init = np.array(x_init, dtype=np.float64)
        # get feature types
        # ot[i] indicates the type of feature i (either OT.f or OT.sos)
        # there is at most one feature of type OT.f
        ot = self.problem.getFeatureTypes()

        # print(str(x) + " " + str(self._evaluate_cost_jacobian(x)))
        print(ot)
        
        # now code some loop that iteratively queries the problem and updates x till convergence        
        it_amount = 1000
        
        if 2 not in ot:
            evaluate_cost = self._evaluate_cost
            evaluate_cost_jacobian = self._evaluate_cost_jacobian
            it_amount = 1000
            it_trys = 3
            it_per_try = int(it_amount / it_trys)
            
            methods = [self._gradient_descent]
            methods_param_total = [1]
            solved = False
            
            x, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, False)
            
            if solved == True:
                print("Gradient descent: Solved")
                show_data.cost_over_time(self.problem)
                return x
            
            methods = [self._newtons_method]
            methods_param_total = [1]

            x, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, False)
            
            if solved == True:
                print("Newtons method: Solved")
                show_data.cost_over_time(self.problem)
                return x
            
            methods = [self._line_search]
            methods_param_total = [3]

            x, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, False)

            if solved == True:
                print("Line Search: Solved")
                show_data.cost_over_time(self.problem)
                return x

            show_data.cost_over_time(self.problem)
            # show_data.gradient_over_time(self.problem)
            
            return x
        
        else:
            evaluate_cost = self._evaluate_cost_sos_total
            evaluate_cost_jacobian = self._evaluate_cost_jacobian_sos
            it_amount = 1000
            it_trys = 1
            it_per_try = int(it_amount / it_trys)
            
            methods = [self._sos_gradient]
            methods_param_total = [1]
            solved = False
            
            x, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, True)
            
            if solved == True:
                print("Gradient descent: Solved")
                show_data.cost_over_time_sos(self.problem)
                return x
            
            show_data.cost_over_time_sos(self.problem)
            
            methods = [self._line_search]
            methods_param_total = [3]

            # x, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, True)

            # if solved == True:
                # print("Line Search: Solved")
                # show_data.cost_over_time_sos(self.problem)
                # return x
            
            # show_data.cost_over_time_sos(self.problem)
            # show_data.gradient_over_time(self.problem)
            
            return x