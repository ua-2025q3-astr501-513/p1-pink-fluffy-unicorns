import numpy as np

def grad_descent(model, xdata, ydata, params, step_size):
    """
    Calculates the gradient of the sum of squared errors using the central difference 
    method to compute derivaties.

    Parameters
    --------
    - model:  function
        Function f(x, params) that returns the expected ydata for given xdata and params.
    - xdata:  np.ndarray
        Independent variable data
    - ydata:  np.ndarray
        Dependent variable data
    - params: np.ndarray
        Initial guess for the model parameters

    - step_size:  float
        Value that the params are perturbed by

    Returns
    --------
    - grad: np.ndarray
        Calculated gradient of the sum of squared errors
    """
    grad = np.zeros_like(params)

    for i in range(len(params)):
        # change each parameter individually by the step size and update the gradient accordingly
        dparams = np.zeros_like(params)
        dparams[i] = step_size

        # calculate the left and right components of the central difference
        y_plus  = model(xdata, *(params + dparams))
        y_minus = model(xdata, *(params - dparams))

        # Calculate the summed square errors
        sq_err_plus  = np.sum((ydata - y_plus )**2)
        sq_err_minus = np.sum((ydata - y_minus)**2)

        # Apply the central difference formula to update the gradient for the specific parameter
        grad[i] = (sq_err_plus - sq_err_minus)/(2 * step_size)

    return grad

def least_sq_fit(model, xdata, ydata, init_params, show_history = False, max_iterations = 5000, tolerance = 1e-6, step_size = 1e-3, learn_rate = 1e-3):
    """
    Least square fitter that uses the central difference gradient descent algorithm.
    
    Parameters
    --------
    - model:  function
        Function f(x, params) that returns the expected ydata for given xdata and params.
    - xdata:  np.ndarray
        Independent variable data
    - ydata:  np.ndarray
        Dependent variable data
    - init_params: np.ndarray
        Initial guess for the model parameters
    
    - show_history: Bool
        Choice to return a numpy array containing the sum of square errors at each step.
    - max_iterations: int
        Maximum number of iterations before the function outputs the parameters, 
        even if the tolerance has not been achieved. Defaults to 5000.
    - tolerance: float
        Stopping tolerance based on magnitude/norm of the gradient, Defaults to 1e-6
    - step_size:  float
        Value that the params are perturbed by. Defaults to 1e-7
    - learn_rate: float
        Learning rate for gradient descent. Defaults to 1e-3

    Returns
    --------
    - params: np.ndarray
        Optimized parameters
    - history: np.ndarray
        Stored sum of square errors at each step. Useful for checking if the algorithm is 
        converging or diverging for the given learning rate.
    """
    # confirm that `init_params` are given as a numpy array
    assert(isinstance(init_params, np.ndarray))

    history = []
    params = init_params

    for _ in range(max_iterations):
        # calculate the predicted ydata using the model and the inital param guess
        y_predicted = model(xdata, *params)
        # compute and store the sum of square errors for the current prediction
        residuals   = ydata - y_predicted
        sum_sq_errs = np.sum(residuals**2)
        history.append(sum_sq_errs)

        # use the gradient descent algorithm to calculate new params
        grad = grad_descent(model=model, xdata=xdata, ydata=ydata, params=params, step_size=step_size, learn_rate=learn_rate)

        # calculate the new params based on the computed gradient and the learning rate
        params = params - learn_rate * grad

        # if the tolerance is exceeded, break the for-loop and return immediately
        if np.linalg.norm(grad) < tolerance:
            break

    if show_history:
        return params, np.array(history)
    else:
        return params