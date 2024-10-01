"""
Gradient descent algorithms, based on Lab 1.

YOU SHOULD NOT NEED TO CHANGE THIS FILE FOR LAB 3.

"""

import torch

class ParameterSpace:
    """
    Interface for a multidimensional parameter space.
    
    You can:
        - determine the current position in the parameter space.
        - determine the current status (actively searching, 
          exceeded_step_limit, found a local minimum)
        - determine the gradient at a particular position
        - step to a new position
    
    """
        
    def __init__(self, start_pos, precision, max_steps):
        """
        Initializes the parameter space.
        
        start_pos is your starting position, an order-1 torch.tensor
        precision is how small a step indicates convergence.
        max_steps is the maximum number of steps you're willing to take
          before retreating back to the starting position
        
        """
        self.curr_pos = start_pos
        self.precision = precision
        self.steps = [start_pos]        
        self.max_steps = max_steps
    
    def gradient(self, position):
        """
        Returns the gradient at a particular position.
        
        position is an order-1 torch.tensor, i.e. a vector
          
        The return value is also an order-1 torch.tensor.
        
        """
        raise NotImplementedError('Cannot call .gradient() on abstract class.')

    def current_position(self):
        """
        Returns your current position.
        
        The return value is an order-1 torch.tensor, i.e. a vector
        
        """
        return self.curr_pos
        
    def step_to(self, position):
        """
        Changes your current position to the new position.
        
        position is an order-1 torch.tensor, i.e. a vector.
        
        """
        self.steps.append(position)
        self.curr_pos = position
        return self.status()

    ACTIVELY_SEARCHING = 0
    EXCEEDED_STEP_LIMIT = 1
    CONVERGED = 2

    def status(self):
        """
        Returns the current status of your search.
        
        - ParameterSpace.ACTIVELY_SEARCHING means that the search is still active.
        - ParameterSpace.EXCEEDED_STEP_LIMIT means that you have exceeded the
          maximum number of steps that you are willing to take
        - ParameterSpace.CONVERGED means that the gradient descent
          has converged.
        
        """
        if len(self.steps) > self.max_steps + 1:
            return ParameterSpace.EXCEEDED_STEP_LIMIT
        elif (len(self.steps) >= 2 and 
              torch.max(torch.abs(self.steps[-1] - self.steps[-2])) < self.precision):
            return ParameterSpace.CONVERGED
        else:
            return ParameterSpace.ACTIVELY_SEARCHING
    

def grad_descent(step_fn, param_space):
    """
    A general-purpose gradient descent algorithm.
    
    step_fn is a function that takes a position as input (expressed as
    a torch.tensor), and returns the relative step to take
    (also expressed as a torch.tensor).
    
    param_space is the ParameterSpace.
    
    The return value should be a list of the positions (including the starting
    positions) visited during the gradient descent. 
    
    """
    steps = [param_space.current_position()]
    while param_space.status() == ParameterSpace.ACTIVELY_SEARCHING:
        step = step_fn(param_space.current_position())
        next_pos = param_space.current_position() + step
        param_space.step_to(next_pos)
        steps.append(next_pos)
    return steps


class AdagradStepFunction:
    """
    Computes the next step for adagrad.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that adagrad would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, init_learning_rate, delta = 0.0000001):
        self.sum_squared_gradients = 0.0
        self.delta = delta
        self.loss_gradient = loss_gradient
        self.init_learning_rate = init_learning_rate
        
    def __call__(self, pos):
        squared_gradient = self.loss_gradient(pos) * self.loss_gradient(pos)
        self.sum_squared_gradients += squared_gradient
        learning_rate = (self.init_learning_rate / 
                         (self.delta + 
                          torch.sqrt(self.sum_squared_gradients)))
        return -learning_rate * self.loss_gradient(pos)

def adagrad(rate, param_space):
    return grad_descent(AdagradStepFunction(param_space.gradient, rate), param_space)

