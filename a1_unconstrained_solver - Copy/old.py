x_list = list()
        cost_list = list()
        
        if 2 not in types:
            evaluate_cost = self._evaluate_cost
            evaluate_cost_jacobian = self._evaluate_cost_jacobian
            it_trys = 3
            it_per_try = int(self._iteration_total / it_trys)
            
            methods = [self._gradient_descent]
            methods_param_total = [1]
            solved = False
            
            x, cost, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, False)
            
            x_list.append(x)
            cost_list.append(cost)
            
            if solved == True:
                print("Gradient descent: Solved")
                show_data.cost_over_time(self.problem)
                # return x
            else:
                show_data.cost_over_time(self.problem)
            
            methods = [self._newtons_method]
            methods_param_total = [1]

            x, cost, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, False)
            
            x_list.append(x)
            cost_list.append(cost)
            
            if solved == True:
                print("Newtons method: Solved")
                show_data.cost_over_time(self.problem)
                # return x
            else:
                show_data.cost_over_time(self.problem)
            
            methods = [self._line_search]
            methods_param_total = [3]

            x, cost, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, False)

            x_list.append(x)
            cost_list.append(cost)

            if solved == True:
                print("Line Search: Solved")
                show_data.cost_over_time(self.problem)
                # return x
            else:
                show_data.cost_over_time(self.problem)
            
            show_data.cost_over_time(self.problem)
        
        else:
            evaluate_cost = self._evaluate_cost_sos_total
            evaluate_cost_jacobian = self._evaluate_cost_jacobian_sos
            it_trys = 2
            it_per_try = int(self._iteration_total / it_trys)
            
            methods = [self._sos_gradient]
            methods_param_total = [1]
            solved = False
            
            x, cost, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, True)
            
            x_list.append(x)
            cost_list.append(cost)
            
            if solved == True:
                print("Gradient descent: Solved")
                show_data.cost_over_time_sos(self.problem)
                # return x
            else:
                show_data.cost_over_time_sos(self.problem)
            
            
            methods = [self._line_search]
            methods_param_total = [3]

            x, cost, solved = self._try_solution(x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, True)

            x_list.append(x)
            cost_list.append(cost)

            if solved == True:
                print("Line Search: Solved")
                show_data.cost_over_time_sos(self.problem)
                # return x
            else:
                show_data.cost_over_time_sos(self.problem)
        
        x_min_index = np.argmin(cost_list)
        
        print(x_list)
        print(cost_list)
        print(x_min_index)
        
        x_min = x_list[x_min_index]
        
        self.problem.evaluate(x_min)
        
    def _try_solution(self, x_init, it_per_try, methods, methods_param_total, evaluate_cost, evaluate_cost_jacobian, is_sos):
        x = x_init
        x_before = x_init
        cost = self._max_val
        cost_before = self._max_val
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
                        return 0, self._max_val, False
                    if is_sos:
                        cost = self._get_sos_cost(phi_jacobian[0])
                        if cost < self._maximum_cost:
                            return x, cost, True
                    else:
                        cost = phi_jacobian[0]
                        if cost < self._maximum_cost:
                            return x, cost, True
        except:
            return x_before, cost_before, False
        finally:
            return x, cost, False
            
        while not self._stop(expected_cost_change):
            phi, jacobian = self._evaluate(x_current)
            cost_new = self._cost_total(phi)
            gradient = self._gradient_total(phi, jacobian)
            x_new, expected_cost_change = self._gradient_descent(x_current, gradient)
            if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                # no need for gradient to be recalculated
                hessian = self._hessian(x_current)
                x_new, expected_cost_change = self._newtons_method(x_current, gradient, hessian)
                phi, jacobian = self._evaluate(x_new)
                cost_new = self._cost_total(phi)
                if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                    solved, x_new, phi, jacobian = self._line_search_normalized(x_current)
                    cost_new = self._cost_total(phi)
                    expected_cost_change = np.linalg.norm(self._gradient_total(phi, jacobian) * (x_new - x_current))
                    if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                        # Now try with lower alpha value
                        self._alpha *= self._lower_alpha_multiplier
                        # Try everything again with lower alpha
                        # because of the unnecessary last evaluation thingy
                        expected_cost_change = np.inf
                        continue
            # In case of successful iteration.
            x_current = x_new
            cost_current = cost_new
            
    if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                # no need for gradient to be recalculated
                hessian = self._hessian(x_current)
                x_new, expected_cost_change = self._newtons_method(x_current, gradient, hessian)
                phi, jacobian = self._evaluate(x_new)
                cost_new = self._cost_total(phi)
                if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                    solved, x_new, phi, jacobian = self._line_search_normalized(x_current)
                    cost_new = self._cost_total(phi)
                    expected_cost_change = np.linalg.norm(self._gradient_total(phi, jacobian) * (x_new - x_current))
                    if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                        # Now try with lower alpha value
                        self._alpha *= self._lower_alpha_multiplier
                        # Try everything again with lower alpha
                        # because of the unnecessary last evaluation thingy
                        expected_cost_change = np.inf
                        continue
            # In case of successful iteration.
            x_current = x_new
            cost_current = cost_new
            
            phi, jacobian = self._evaluate(x_current)
            cost_new = self._cost_total(phi)
            gradient = self._gradient_total(phi, jacobian)
            x_new, expected_cost_change = self._gradient_descent(x_current, gradient)
        
        while not self._stop(expected_cost_change):
            phi, jacobian = self._evaluate(x_current)
            cost_new = self._cost_total(phi)
            gradient = self._gradient_total(phi, jacobian)
            x_new, expected_cost_change = self._gradient_descent(x_current, gradient)
            if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                # no need for gradient to be recalculated
                hessian = self._hessian(x_current)
                x_new, expected_cost_change = self._newtons_method(x_current, gradient, hessian)
                phi, jacobian = self._evaluate(x_new)
                cost_new = self._cost_total(phi)
                if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                    solved, x_new, phi, jacobian = self._line_search_normalized(x_current)
                    cost_new = self._cost_total(phi)
                    expected_cost_change = np.linalg.norm(self._gradient_total(phi, jacobian) * (x_new - x_current))
                    if self._reject_step_convex_assumption(expected_cost_change, cost_new, cost_current):
                        # Now try with lower alpha value
                        self._alpha *= self._lower_alpha_multiplier
                        # Try everything again with lower alpha
                        # because of the unnecessary last evaluation thingy
                        expected_cost_change = np.inf
                        continue
            # In case of successful iteration.
            x_current = x_new
            cost_current = cost_new
        
        # while not self._stop(gradient):
            # phi, jacobian = self._evaluate(x_current)
            # cost_current = self._cost_total(phi)
            # if cost_current >= cost_before:
                # self._alpha *= self._lower_alpha_multiplier
                # replace x_current to before
                # x_current = x_before
                # no need for gradient / hessian to be recalculated
            # else:
                # self._alpha = alpha_init
                # x_before = x_current
                # cost_before = cost_current
                # gradient = self._gradient_total(phi, jacobian)
                # hessian = self._hessian(x_current)
            # x_current = self._newtons_method(x_current, gradient, hessian)
        
        # _, x_current, _ = self._line_search_normalized(x_current)   
        try:
            x_try = np.ones((3, 400))
            for i in range(3):
                x_try[i] = (np.arange(400) - 200) * 0.1
            for i in range(400):
                x_try_one = x_try[:, i]
                phi, J = self._evaluate(x_try_one)
        except:
            pass
        show_data.cost_over_time(self.problem)
        return x