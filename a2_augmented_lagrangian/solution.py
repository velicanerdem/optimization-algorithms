import numpy as np
import sys
import warnings

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT
# from utility import show_data 
from optimization_algorithms.utils import finite_diff

class SolverAugmentedLagrangian(NLPSolver):

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

    class _AugmentedLagrangian():
        
        _square_cost_multiplier_init = 1.0
        _lagrange_multiplier_init = 0.0
        
        _square_cost_increase_multiplier = 1.2
        
        epsilon = 0.001
        
        def __init__(self, featureTypes, update_cost_multipliers=False):
            # Writing _AugmentedLagrangian instead of self.__class__ doesn't work
            self._ineq_square_cost_multiplier = self.__class__._square_cost_multiplier_init
            self._eq_square_cost_multiplier = self.__class__._square_cost_multiplier_init

            inequality_amount = len(featureTypes.ineq_index)
            equality_amount = len(featureTypes.eq_index)
            
            self._inequality_lagrangians = np.ones((inequality_amount)) * self.__class__._lagrange_multiplier_init
            self._equality_lagrangians = np.ones((equality_amount)) * self.__class__._lagrange_multiplier_init

            if update_cost_multipliers:
                self.update_function = self._update_with_cost_multipliers
            else:
                self.update_function = self._update
        
        def _update(self, inequality_cost, equality_cost):
            inequality_lagrangian_update = self._inequality_lagrangians + (2 * self._ineq_square_cost_multiplier) * inequality_cost
            
            self._inequality_lagrangians = np.maximum(inequality_lagrangian_update, 0)
        
            self._equality_lagrangians += (2 * self._eq_square_cost_multiplier) * equality_cost
        
        def update(self, inequality_cost, equality_cost):
            return self.update_function(inequality_cost, equality_cost)
        
        def _update_with_cost_multipliers(self, inequality_cost, equality_cost):
            self._update(equality_cost, inequality_cost)
            
            self._ineq_square_cost_multiplier *= self.__class__._square_cost_increase_multiplier
            self._eq_square_cost_multiplier *= self.__class__._square_cost_increase_multiplier
        
        def inequality_cost_total(self, inequality_cost):
            active_inequalities = np.logical_or(inequality_cost >= 0, self._inequality_lagrangians > 0)
            
            active_inequality_cost = inequality_cost[active_inequalities]
            active_inequality_lagrangian = self._inequality_lagrangians[active_inequalities]
            
            square_cost = self._ineq_square_cost_multiplier * np.dot(active_inequality_cost, active_inequality_cost)
            lagrangian_cost = np.dot(active_inequality_lagrangian, active_inequality_cost)
            
            cost = square_cost + lagrangian_cost
            
            return cost
            
        def equality_cost_total(self, equality_cost):
            square_cost = self._eq_square_cost_multiplier * np.dot(equality_cost, equality_cost)
            lagrangian_cost = np.dot(self._equality_lagrangians, equality_cost)
            
            cost = square_cost + lagrangian_cost
            
            return cost
        
        def inequality_gradient_total(self, inequality_cost, inequality_gradient):
            active_inequalities = np.logical_or(inequality_cost >= 0, self._inequality_lagrangians > 0)
            
            active_inequality_cost = inequality_cost[active_inequalities]
            active_inequality_gradient = inequality_gradient[active_inequalities]
            active_inequality_lagrangian = self._inequality_lagrangians[active_inequalities]
            
            ineq_gradient_function_multiply = 2 * self._ineq_square_cost_multiplier * active_inequality_cost + active_inequality_lagrangian
            ineq_cost_function_gradient = active_inequality_gradient.T @ ineq_gradient_function_multiply

            return ineq_cost_function_gradient
        
        def equality_gradient_total(self, equality_cost, equality_gradient):
            eq_gradient_function_multiply = 2 * self._eq_square_cost_multiplier * equality_cost  + self._equality_lagrangians
            
            eq_cost_function_gradient = equality_gradient.T @ eq_gradient_function_multiply
            
            return eq_cost_function_gradient
            
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
        
        self._method_list = list()
        self._gradient_descent_method = self._Method(self._gradient_descent, self._gradient_descent_string, alpha, 10)
        self._newtons_method_method = self._Method(self._newtons_method, self._newtons_method_string, alpha, 10)
        self._line_search_method = self._Method(self._line_search_normalized, self._line_search_string, alpha, 1)
        
        self._method_list.append(self._gradient_descent_method)
        # self._method_list.append(self._newtons_method_method)
        # self._method_list.append(self._line_search_method)
        
        self._lower_alpha_multiplier = 0.9
        self._try_amount = 10
        
        self._step_increase = 1.2
        self._stepsize_decrement = 0.5
        self._minimum_desired_decrease_multiplier = 0.01
        self._stop_value = 1e-6
        
        self._iteration_current = 0
        self._iteration_total = 10000        
        
        self._line_search_forward_pass = 1
        
        self._method_multiplier = 0.25
        
        self._methods_list = list()
        self._methods_list.append(self._gradient_descent_string)
        self._methods_list.append(self._newtons_method_string )
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
            cost += self._augmented_lagrangian.inequality_cost_total(phi[ineq_index])
        if len(eq_index) > 0:
            cost += self._augmented_lagrangian.equality_cost_total(phi[eq_index])
        
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
            gradient += self._augmented_lagrangian.inequality_gradient_total(phi[ineq_index], J[ineq_index])
        if len(eq_index) > 0:
            gradient += self._augmented_lagrangian.equality_gradient_total(phi[eq_index], J[eq_index])
        
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
            cost += self._augmented_lagrangian.inequality_cost_total(phi[ineq_index])
            gradient += self._augmented_lagrangian.inequality_gradient_total(phi[ineq_index], J[ineq_index])
        if len(eq_index) > 0:
            cost += self._augmented_lagrangian.equality_cost_total(phi[eq_index])
            gradient += self._augmented_lagrangian.equality_gradient_total(phi[eq_index], J[eq_index])

        return cost, gradient
    
    def _stop_by_checking_constraint_costs(self, phi):
        # whats wrong with me that i want to optimize this
        eq_index = self._featureTypes.eq_index
        ineq_index = self._featureTypes.ineq_index
        
        len_ineq_index = len(ineq_index)
        len_eq_index = len(eq_index)
        
        cost = np.zeros((len_ineq_index + len_eq_index))
        
        if len(ineq_index) > 0:
            cost[:len_ineq_index] = phi[ineq_index]
        if len(eq_index) > 0:
            cost[len_ineq_index:] = phi[eq_index]
        
        if np.all(cost < self._augmented_lagrangian.epsilon):
            return True
        else:
            if self._iteration_current > self._iteration_total:
                return True
            else:
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
        self._augmented_lagrangian = self._AugmentedLagrangian(self._featureTypes)
        
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
        phi_current = np.ones((self._featureTypes.type_amount)) * np.inf
        max_last_change = np.inf
        
        
        while not self._stop_by_checking_constraint_costs(phi_current):
            while not self._stop(max_last_change):
                list_expected_cost_change = [m.last_cost_change for m in self._method_list]
                method_with_max_change_index = np.argmax(list_expected_cost_change)
                method_with_max_change = self._method_list[method_with_max_change_index]

                if method_with_max_change.name == self._gradient_descent_string:
                    current_it = 0
                    phi_current, jacobian_current = self._evaluate(x_current)
                    cost_current, gradient_current = self._cost_gradient_total(phi_current, jacobian_current)
                    while current_it < method_with_max_change.iteration_amount:
                        current_it += 1
                        x_new, expected_cost_change = self._gradient_descent(x_current, gradient_current)
                        phi_new, jacobian_new = self._evaluate(x_new)
                        cost_new, gradient_new = self._cost_gradient_total(phi_new, jacobian_new)
                        cost_diff = cost_current - cost_new
                        if self._reject_step(expected_cost_change, cost_diff):
                            method_with_max_change.lower_alpha(self._lower_alpha_multiplier)
                            method_with_max_change.last_cost_change *= self._lower_alpha_multiplier
                        else:
                            method_with_max_change.last_cost_change = cost_diff
                            x_current = x_new
                            cost_current = cost_new
                            gradient_current = gradient_new
                
                elif method_with_max_change.name == self._newtons_method_string:
                    current_it = 0
                    phi_current, jacobian_current = self._evaluate(x_current)
                    cost_current, gradient_current = self._cost_gradient_total(phi_current, jacobian_current)
                    hessian_current = hessian_method(x_current, jacobian_current)
                    while current_it < method_with_max_change.iteration_amount:
                        current_it += 1
                        x_new, expected_cost_change = self._newtons_method(x_current, gradient_current, hessian_current)
                        phi_new, jacobian_new = self._evaluate(x_new)
                        cost_new, gradient_new = self._cost_gradient_total(phi_new, jacobian_new)
                        cost_diff = cost_current - cost_new
                        if self._reject_step(expected_cost_change, cost_diff):
                            method_with_max_change.lower_alpha(self._lower_alpha_multiplier)
                            method_with_max_change.last_cost_change *= self._lower_alpha_multiplier
                        else:
                            method_with_max_change.last_cost_change = cost_diff
                            x_current = x_new
                            cost_current = cost_new
                            gradient_current = gradient_new
                            hessian_current = hessian_method(x_new, jacobian_current)
                
                elif method_with_max_change.name ==  self._line_search_string:
                    current_it = 0
                    while current_it < method_with_max_change.iteration_amount:
                        current_it += 1
                        solved, x_current, point_phi, point_J, cost_diff = self._line_search_normalized(x_current)
                        method_with_max_change.last_cost_change = cost_diff
                
                
                max_last_change = np.max([m.last_cost_change for m in self._method_list])
            
            self._augmented_lagrangian.update(phi_current[self._featureTypes.ineq_index], phi_current[self._featureTypes.eq_index])
            max_last_change = np.inf
        
        # show_data.cost_over_time(self.problem)
        # show_data.gradient_over_time(self.problem)
        
        phi, jacobian = self._evaluate(x_current)

        return x_current