import numpy as np
import sys
import warnings

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT
from utility import show_data 

class SolverUnconstrained(NLPSolver):

    class FeatureTypes:
        def __init__(self, program):
            self.x_shape = np.shape(program.getInitializationSample())
        
            types = program.getFeatureTypes()
            print("Types: {}".format(types))
            self.f_index = [i for i in range(len(types)) if types[i]==OT.f]
            self.sos_index = [i for i in range(len(types)) if types[i]==OT.sos]
            self.eq_index = [i for i in range(len(types)) if types[i]==OT.eq]
            self.ineq_index = [i for i in range(len(types)) if types[i]==OT.ineq]

    class Method:
        def __init__(self, method, name, alpha, iteration_amount):
            self.method = method
            self.name = name
            self.alpha = alpha
            self.iteration_amount = iteration_amount
            
            self.last_expected_cost_change = None
            
        def lower_alpha(self, lower_alpha_multiplier):
            self.alpha *= lower_alpha_multiplier
        
    def __init__(self, alpha=0.1):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        
        self._gradient_descent_string = "Gradient Descent"
        self._newtons_method_string = "Newton's Method"
        self._line_search_string = "Line Search"
        
        self._method_list = list()
        self._gradient_descent_method = self.Method(self._gradient_descent, self._gradient_descent_string, alpha, 10)
        self._newtons_method_method = self.Method(self._newtons_method, self._newtons_method_string, alpha, 10)
        self._line_search_method = self.Method(self._line_search_normalized, self._line_search_string, alpha, 1)
        
        self._method_list.append(self._gradient_descent_method)
        self._method_list.append(self._newtons_method_method)
        self._method_list.append(self._line_search_method)
        
        self._lower_alpha_multiplier = 0.9
        self._try_amount = 10
        
        self._step_increase = 1.2
        self._stepsize_decrement = 0.5
        self._minimum_desired_decrease_multiplier = 0.01
        self._stop_value = 1e-12
        
        self._iteration_current = 0
        self._iteration_total = 800        
        
        self._line_search_forward_pass = 1
        
        
        
        self._methods_list = list()
        self._methods_list.append(self._gradient_descent_string)
        self._methods_list.append(self._newtons_method_string )
        self._methods_list.append(self._line_search_string)
        
        # Should have just made a class and checked independently maybe
        # Maybe not
        self._methods_tried = dict()
        
        for method in self._methods_list:
            self._methods_tried[method] = False

    def _assign_max_alpha(self):
        self._alpha = max(self._alpha_gradient_descent, self._alpha_newtons_method, self._alpha_line_search)

    # Correct with convex assumption
    def _stop(self, expected_cost_change):
        it_remaining = self._iteration_total - self._iteration_current
        maximum_expected_change = expected_cost_change * it_remaining

        if self._stop_value > maximum_expected_change:
            return True
        else:
            return False

    # I wonder if I can add more
    def _reject_step(self, expected_cost_change, cost_new, cost_current):
        cost_diff = cost_current - cost_new
        
        if cost_diff < 0:
            return True
        else:
            return False

    # Utility functions
    
    def _evaluate(self, x):
        # Wrapper
        self._iteration_current += 1
        return self.problem.evaluate(x)

    def _hessian(self, x):
        return self.problem.getFHessian(x)

    def _cost_total(self, phi):
        cost = 0
        f_index = self._featureTypes.f_index
        sos_index = self._featureTypes.sos_index
        eq_index = self._featureTypes.eq_index
        ineq_index = self._featureTypes.ineq_index
        
        if len(f_index) > 0:
            cost += phi[f_index][0]
        if len(sos_index) > 0:
            cost += phi[sos_index].T @ phi[sos_index]
        if len(eq_index) > 0:
            cost += phi[eq_index][0]
        if len(ineq_index) > 0:
            cost += phi[ineq_index][0]
        
        return cost

    def _gradient_total(self, phi, J):        
        gradient = np.zeros(self._featureTypes.x_shape)
        
        f_index = self._featureTypes.f_index
        sos_index = self._featureTypes.sos_index
        eq_index = self._featureTypes.eq_index
        ineq_index = self._featureTypes.ineq_index
        
        if len(f_index) > 0:
            gradient += J[f_index][0]
        if len(sos_index) > 0:
            gradient += J[sos_index].T @ phi[sos_index]
        if len(eq_index) > 0:
            gradient += J[eq_index][0]
        if len(ineq_index) > 0:
            gradient += J[ineq_index][0]
        
        return gradient

    def _cost_gradient_total(self, phi, J):        
        cost = 0
        gradient = np.zeros(self._featureTypes.x_shape)
        
        f_index = self._featureTypes.f_index
        sos_index = self._featureTypes.sos_index
        eq_index = self._featureTypes.eq_index
        ineq_index = self._featureTypes.ineq_index
        
        if len(f_index) > 0:
            cost += phi[f_index][0]
            gradient += J[f_index][0]
        if len(sos_index) > 0:
            cost += phi[sos_index].T @ phi[sos_index]
            gradient += J[sos_index].T @ phi[sos_index]
        if len(eq_index) > 0:
            cost += phi[eq_index][0]
            gradient += J[eq_index][0]
        if len(ineq_index) > 0:
            cost += phi[ineq_index][0]
            gradient += J[ineq_index][0]
        
        return cost, gradient
    
    # Optimization functions

    def _gradient_descent(self, x, gradient):
        descent = self._gradient_descent_method.alpha * gradient
        
        delta = descent * gradient
        expected_cost_change = np.sum(np.abs(delta))
        
        return x - descent, expected_cost_change

    def _newtons_method(self, x, gradient, hessian):        
        gradient_val = np.linalg.inv(hessian) @ gradient
        descent = self._newtons_method_method.alpha * gradient_val
        
        delta = descent * gradient_val
        expected_cost_change = np.sum(np.abs(delta))
        
        return x - descent, expected_cost_change
        

    # Normalized because it uses normalized jacobian for stepwise movement instead of unit.
    # Only optimization function with evaluate inside, but nothing much to do about it
    # Thus it returns phi / jacobian
    def _line_search_normalized(self, x):
        # self._alpha is normally 0.1 thus this will be 1
        # Note: since this is used before gradient descent, maybe lower initial alpha?
        # TODO: Multiplicate evaluate
        solved = False
        
        x_start = x
        
        for _ in range(self._line_search_forward_pass):
            phi, J = self._evaluate(x)
            
            cost, jacobian = self._cost_gradient_total(phi, J)
            
            # For gradient search
            jacobian_norm = np.linalg.norm(jacobian)
            # For x value update
            if jacobian_norm == 0:
                solved = True
                return solved, x, [phi, jacobian] 
            delta = -jacobian / jacobian_norm

            # I really don't know how to program function in the loop correct way
            # Especially with constant values
            # I will just do it before loop first

            # Evaluate desired cost
            desired_cost_decrease_multiplier = self._minimum_desired_decrease_multiplier * -jacobian_norm
            desired_cost_decrease = self._line_search_method.alpha * desired_cost_decrease_multiplier
            desired_cost = cost + desired_cost_decrease

            # Evaluate point cost
            step = self._line_search_method.alpha * delta
            point_to_evaluate = x + step            
            point_phi, point_J = self._evaluate(point_to_evaluate)
            point_cost = self._cost_total(point_phi)
            
            while point_cost > desired_cost:
                # honestly, alpha multiplication is just here so it will be numerically stable
                # otherwise just multiply below values with stepwise decrement.
                self._line_search_method.alpha = self._line_search_method.alpha * self._stepsize_decrement
                
                # Evaluate desired cost
                desired_cost_decrease = self._line_search_method.alpha * desired_cost_decrease_multiplier
                desired_cost = cost + desired_cost_decrease

                # Evaluate point cost
                step = self._line_search_method.alpha * delta
                point_to_evaluate = x + step            
                point_phi, point_J = self._evaluate(point_to_evaluate)
                point_cost = self._cost_total(point_phi)
                # I am just gonna ignore the alpha optimization, why is it in my mind
                # even though it just get rids of one multiplication and putting one addition
                # in place.
                
            x = x + step
            self._line_search_method.alpha = min(self._step_increase * self._line_search_method.alpha, np.inf)


        delta_x = x - x_start
        delta_x_norm = np.linalg.norm(delta_x)
        
        return solved, x, point_phi, point_J
    
    # def _method_wrapper(self, x_current, method):
        # x_next = None
        # match method.name:
            # case self._gradient_descent_string:
                # for _ in range(method.iteration_amount):
                    # phi, jacobian = self._evaluate(x_current)
                    # cost_new = self._cost_total(phi)
                    # gradient_new = self._gradient_total(phi, jacobian)
                    # x_next, expected_cost_change = self._gradient_descent(x_current, gradient_new)
    
    # def _all_methods_tried(self):
        # methods_tried_booleans = self._method_tried.values()
        
        # if np.all(methods_tried_booleans):
            # return True
        # else:
            # return False
    
    # def _reset_method_usage(self):
        # for method in self._methods_list:
            # self._methods_tried[method] = False
    
    """
    Much better structure is needed. This comes as weird.
    """
    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        # write your code here
        warnings.filterwarnings('error')   


        # get feature types
        assert(self.problem != None)
        self._featureTypes = self.FeatureTypes(self.problem)
        
        x_init = self.problem.getInitializationSample()
        
        # Prevent access with None
        x_current = x_init
        x_new = None
        
        cost_current = np.inf
        cost_new = None
                
        # Start for loop
        expected_cost_change = np.inf
        
        
        while not self._stop(expected_cost_change):
            method_alphas = [m.alpha for m in self._method_list]
            method_with_max_alpha_index = np.argmax(method_alphas)
            method_with_max_alpha = self._method_list[method_with_max_alpha_index]
            
            
            if method_with_max_alpha.name == self._gradient_descent_string:
                current_it = 0
                phi_current, jacobian_current = self._evaluate(x_current)
                cost_current, gradient_current = self._cost_gradient_total(phi_current, jacobian_current)
                while current_it < method_with_max_alpha.iteration_amount:
                    current_it += 1
                    x_new, expected_cost_change = self._gradient_descent(x_current, gradient_current)
                    phi_new, jacobian_new = self._evaluate(x_new)
                    cost_new, gradient_new = self._cost_gradient_total(phi_new, jacobian_new)
                    if self._reject_step(expected_cost_change, cost_new, cost_current):
                        method_with_max_alpha.lower_alpha(self._lower_alpha_multiplier)
                    else:
                        x_current = x_new
                        cost_current = cost_new
                        gradient_current = gradient_new
            elif method_with_max_alpha.name ==  self._newtons_method_string:
                current_it = 0
                phi_current, jacobian_current = self._evaluate(x_current)
                cost_current, gradient_current = self._cost_gradient_total(phi_current, jacobian_current)
                hessian_current = self.problem.getFHessian(x_current)
                while current_it < method_with_max_alpha.iteration_amount:
                    current_it += 1
                    # TODO: all leaking reevaluations
                    x_new, expected_cost_change = self._newtons_method(x_current, gradient_current, hessian_current)
                    phi_new, jacobian_new = self._evaluate(x_new)
                    cost_new, gradient_new = self._cost_gradient_total(phi_new, jacobian_new)
                    if self._reject_step(expected_cost_change, cost_new, cost_current):
                        method_with_max_alpha.lower_alpha(self._lower_alpha_multiplier)
                    else:
                        x_current = x_new
                        cost_current = cost_new
                        gradient_current = gradient_new
                        hessian_current = self.problem.getFHessian(x_new)
                    
            elif method_with_max_alpha.name ==  self._line_search_string:
                current_it = 0
                while current_it < method_with_max_alpha.iteration_amount:
                    current_it += 1
                    solved, x_current, point_phi, point_J = self._line_search_normalized(x_current)
                cost_current, gradient_current = self._cost_gradient_total(point_phi, point_J)
            
        # show_data.cost_over_time(self.problem)
        # show_data.gradient_over_time(self.problem)
        
        
        phi, jacobian = self._evaluate(x_current)

        return x_current