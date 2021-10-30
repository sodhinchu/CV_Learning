import numpy as np
import matplotlib.pyplot as plt

def calc_cyclic_lr(iteration, stepsize, min_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration/(2  * stepsize))
    x = np.abs(iteration/stepsize - 2 * cycle + 1)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1-x))
    return lr

def get_lr_changes(num_iterations, stepsize, min_lr, max_lr):
    lr_changes = []
    for iteration in range(num_iterations):
        lr_changes.append(calc_cyclic_lr(iteration, stepsize, min_lr, max_lr))
    return lr_changes
        
        
def plot_lr(num_iterations, stepsize, min_lr, max_lr):
    lr = get_lr_changes(num_iterations, stepsize, min_lr, max_lr)
    plt.plot(lr)
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')
        
        
