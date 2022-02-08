import numpy as np
import sys
import warnings

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT
# from utility import show_data 
from optimization_algorithms.utils import finite_diff

class SolverInteriorPoint(NLPSolver):

    class _FeatureTypes():
        def __init__(self, program):
            self.x_shape = np.shape(program.getInitializationSample())
        
            self.types = program.getFeatureTypes()
            self.type_amount = len(self.types)
            print("Types: {}".format(self.types))
            self.f_index = [i for i in range(len(self.types)) if self.types[i]==OT.f]
            self.sos_index = [i for i in range(len(self.types)) if self.types[i]==OT.sos]
            self.ineq_index = [i for i in range(len(self.types)) if self.types[i]==OT.ineq]
            self.eq_index = [i for i in range(len(self.types)) if self.types[i]==OT.eq]
    
    # _AugmentedLagrangian
    class _LogBarrier():
        
        def __init__(self, featureTypes, update_cost_multipliers=False):
            self._log_multiplier = 1.
            self._log_multiplier_decrease_mult = 0.5
            self.epsilon = 1e-24
        
        def update(self):
            self._log_multiplier *= self._log_multiplier_decrease_mult

        
        # def gradient_total(self, inequality_cost, inequality_gradient):
            # print(inequality_gradient)
            # log_gradient_without_neg = inequality_gradient / inequality_cost.reshape(-1, 1)
            # gradient = self._log_multiplier * log_gradient_without_neg
            # gradient_summed = np.sum(gradient, axis=0)

            # return gradient_summed
        
        def cost_total(self, inequality_cost):
            if np.any(inequality_cost >= 0):
                return np.inf
            log_cost = np.log(-inequality_cost)
            log_cost_summed = np.sum(log_cost)
            cost = -self._log_multiplier * log_cost_summed
            return cost
            
        def gradient_total(self, inequality_cost, inequality_gradient):
            log_gradient_without_neg = inequality_gradient / inequality_cost.reshape(-1, 1)
            gradient = self._log_multiplier * log_gradient_without_neg

            gradient_summed = np.sum(gradient, axis=0)

            return gradient_summed
            
    class _Method():
        def __init__(self, _Method, name, alpha, iteration_amount):
            self._Method = _Method
            self.name = name
            self.alpha = alpha
            self.iteration_amount = iteration_amount
            
            self.last_cost_change = 1
            
        def lower_alpha(self, lower_alpha_multiplier):
            self.alpha *= lower_alpha_multiplier
        
    def __init__(self, alpha=0.1):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        
        self._gradient_descent_string = "Gradient Descent"
        self._newtons_method_string = "Newton's _Method"
        self._line_search_string = "Line Search"
        
        self._alpha_init = alpha
        
        self._method_list = list()
        self._gradient_descent_method = self._Method(self._gradient_descent, self._gradient_descent_string, self._alpha_init, 10)
        self._newtons_method_method = self._Method(self._newtons_method, self._newtons_method_string, self._alpha_init, 10)
        self._line_search_method = self._Method(self._line_search_normalized, self._line_search_string, self._alpha_init, 1)
        
        self._method_list.append(self._gradient_descent_method)
        # self._method_list.append(self._newtons_method_method)
        # self._method_list.append(self._line_search_method)
        
        self._lower_alpha_multiplier = 0.9
        self._try_amount = 10
        
        self._step_increase = 1.2
        self._stepsize_decrement = 0.5
        self._minimum_desired_decrease_multiplier = 0.01
        
        self._iteration_current = 0
        self._iteration_total = 10000
        
        self._stop_value = 1e-3 / self._iteration_total
        
        self._line_search_forward_pass = 1
        
        self._method_multiplier = 0.25
        
        self._methods_list = list()
        self._methods_list.append(self._gradient_descent_string)
        self._methods_list.append(self._newtons_method_string)
        self._methods_list.append(self._line_search_string)
        
        # Should have just made a class and checked independently maybe
        # Maybe not
        self._methods_tried = dict()
        
        for _Method in self._methods_list:
            self._methods_tried[_Method] = False

    def _assign_max_alpha(self):
        self._alpha = max(self._alpha_gradient_descent, self._alpha_newtons_method, self._alpha_line_search)

    # Correct with convex assumption
    def _stop(self, max_last_change):
        it_remaining = self._iteration_total - self._iteration_current
        maximum_expected_change = max_last_change * it_remaining

        if self._stop_value > maximum_expected_change:
            return True
        else:
            return False

    # I wonder if I can add more
    def _reject_step(self, expected_cost_change, cost_diff):
        if cost_diff <= 0:
            return True
        else:
            return False

    # Utility functions
    
    def _evaluate(self, x):
        # Wrapper
        self._iteration_current += 1
        return self.problem.evaluate(x)

    def _hessian_wrapper_cost(self, x):
        phi, _ = self._evaluate(x)
        cost = self._cost_total(phi)
        
        return cost

    # compatibility with sos hessian and vice versa
    def _hessian(self, x, J):
        return self.problem.getFHessian(x)

    def _hessian_sos(self, x, J):
        hessian_finite_diff = finite_diff.finite_diff_hess(self._hessian_wrapper_cost, x, 0.000001)
        
        return hessian_finite_diff
        
    def _hessian_sos_2(self, x, J):
        return 2 * J.T @ J

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
        if len(ineq_index) > 0:
            cost += self._log_barrier.cost_total(phi[ineq_index])
        # print(cost)
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
        if len(ineq_index) > 0:
            gradient += self._log_barrier.gradient_total(phi[ineq_index], J[ineq_index])
        
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
        if len(ineq_index) > 0:
            cost += self._log_barrier.cost_total(phi[ineq_index])
            gradient += self._log_barrier.gradient_total(phi[ineq_index], J[ineq_index])
        
        return cost, gradient
    
    def _stop_log_barrier(self, x_last_change):
        # print(x_last_change)
        if self._log_barrier.epsilon > x_last_change:
            pass
            # return True
        if self._iteration_current >= self._iteration_total:
            return True
        return False
    
    # Optimization functions

    def _gradient_descent(self, x, gradient):
        descent = self._gradient_descent_method.alpha * gradient
        
        delta = descent * gradient
        expected_cost_change = np.sum(np.abs(delta))
        
        return x - descent, expected_cost_change

    def _newtons_method(self, x, gradient, hessian):        
        gradient_val = np.linalg.pinv(hessian) @ gradient
        descent = self._newtons_method_method.alpha * gradient_val
        
        delta = descent * gradient_val
        expected_cost_change = np.sum(np.abs(delta))
        
        return x - descent, expected_cost_change
        

    # Normalized because it uses normalized jacobian for stepwise movement instead of unit.
    # Only optimization function with evaluate inside, but nothing much to do about it
    # Thus it returns phi / jacobian
    # Doesn't work on SoS, loops.
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
            
             # and not self._stop(desired_cost_decrease)
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
        
        return solved, x, point_phi, point_J, cost - point_cost
    
    # def _method_wrapper(self, x_current, _Method):
        # x_next = None
        # match _Method.name:
            # case self._gradient_descent_string:
                # for _ in range(_Method.iteration_amount):
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
        # for _Method in self._methods_list:
            # self._methods_tried[_Method] = False
    
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
        self._featureTypes = self._FeatureTypes(self.problem)
        assert(len(self._featureTypes.eq_index) == 0)
        
        self._log_barrier = self._LogBarrier(self._featureTypes)
        
        if len(self._featureTypes.sos_index) == 0:
            hessian_method = self._hessian
        else:
            hessian_method = self._hessian_sos_2
        
        x_init = self.problem.getInitializationSample()
        
        # Prevent access with None
        x_current = x_init
        x_new = None
        
        cost_current = np.inf
        cost_new = None
                
        # Start for loop
        # not the best can change
        x_change = np.inf
        max_last_change = np.inf

        """
        Take care in case of failure of changing values since whatever the values are used they will be invalid
        """
        try:
            while not self._stop_log_barrier(x_change):
                max_last_change = np.inf
                x_current_iteration = x_current
                while not self._stop(max_last_change):
                    list_expected_cost_change = [m.last_cost_change for m in self._method_list]
                    method_with_max_change_index = np.argmax(list_expected_cost_change)
                    method_with_max_change = self._method_list[method_with_max_change_index]

                    phi_current, jacobian_current = self._evaluate(x_current)
                    cost_current, gradient_current = self._cost_gradient_total(phi_current, jacobian_current)
                    x_new, expected_cost_change = self._gradient_descent(x_current, gradient_current)
                    phi_new, jacobian_new = self._evaluate(x_new)
                    cost_new, gradient_new = self._cost_gradient_total(phi_new, jacobian_new)
                    cost_diff = cost_current - cost_new
                    if self._reject_step(expected_cost_change, cost_diff):
                        method_with_max_change.lower_alpha(self._lower_alpha_multiplier)
                        method_with_max_change.last_cost_change *= self._lower_alpha_multiplier
                    else:
                        # print(self._log_barrier._log_multiplier)
                        method_with_max_change.alpha = self._alpha_init
                        method_with_max_change.last_cost_change = cost_diff
                        x_current = x_new
                        cost_current = cost_new
                        gradient_current = gradient_new
                    
                    max_last_change = np.max([m.last_cost_change for m in self._method_list])
                
                self._log_barrier.update()
                x_change = x_current - x_current_iteration
                x_change = np.sum(np.abs(x_change))
        except:
            show_data.cost_over_time(self.problem)
            self._evaluate(x_current)
            return x_current
        
        # show_data.cost_over_time(self.problem)
        # show_data.x_over_time(self.problem)
        # show_data.gradient_over_time(self.problem)
        # print(self._iteration_current)
        phi, jacobian = self._evaluate(x_current)
        # print(x_current)
        return x_current