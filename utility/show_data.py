from matplotlib import pyplot as plt
import numpy as np

def cost_over_time(problem):
    phi_trace = problem.trace_phi
   
    plt.plot(phi_trace)
    plt.show()
   
def gradient_over_time(problem):
    jacobian_trace = problem.trace_J
    
    absolute = np.abs(jacobian_trace)
    total = np.sum(absolute, axis=2)
   
    plt.plot(total)
    plt.show()
   
def gradient_over_time_sos(problem):
    jacobian_trace = problem.trace_J
    
    print(len(jacobian_trace))
    print(len(jacobian_trace[0]))
    print(len(jacobian_trace[0][0]))
    
    absolute = np.abs(jacobian_trace)
    total = np.sum(absolute, axis=-1)
   
    plt.plot(total)
    plt.show()
    
def cost_over_time_sos(problem):
    phi_trace = problem.trace_phi
    phi_trace = np.array(phi_trace)

    phi_trace_summed = np.sum(phi_trace * phi_trace, axis=1)
    
    plt.plot(phi_trace_summed)
    plt.show()