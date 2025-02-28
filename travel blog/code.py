# main
def main():
    # Generating training data
    data = []
    for i in range(100):
        x = np.random.uniform(3., 12.)
        eps = np.random. normal(0., 0.1)
        y = 1.477 * x + 0.089 + eps
        data.append([x, y])

    data = np.array(data)

    lr = 0.01
    initial_b = 0   # Initialize b
    initial_w = 0   #Initialize w
    num_iterations = 1000

    #  Train model
    b, w = gradient_descent(data, initial_b, initial_w, lr, num_iterations)

    #calculating 
    loss = mse(b, w, data)
    print(f'Final loss: {loss}, w: {w}, b: {b}')

# Call the main function to execute the program
main()
#  2. step gradient , partial derivative
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points)) #total number of samples

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2 / M) * ((w_current * x + b_current) - y)
        w_gradient += (2 / M) * x * ((w_current * x + b_current) - y)
    
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]#  1. gradient descent
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w

    for step in range(num_iterations):
        b, w = step_gradient(b, w, points, lr)
        if step % 50 == 0:
            loss = mse(b, w, points)
            print(f"Iteration {step}, Loss: {loss}, w: {w}, b: {b}")
    return [b,w]# Linear Regression in Action
import numpy as np
# 3. MSE
def mse(b, w, points):
  totalError = 0;
  for i in range(len(points)):
    x = points[i, 0];
    y = points[i, 1];
    totalError += (y - (w * x + b)) ** 2
  return totalError / float(len(points))