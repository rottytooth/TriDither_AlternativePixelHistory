import numpy as np

start_x = 1
cols = 3
rows = 3
initial_load = True

kernel_list = []
kernel = np.array([1, 0, 0, 1, 0, 0, 14])
kernel_length = cols * rows - start_x - 1
sum = np.sum(kernel)

def load_kernel(initial_kernel, kernel, sum, digit, kernel_length): 
    if initial_kernel != None:
        if digit == kernel_length:
            kernel_list.append(kernel)
            initial_kernel = None
            return
        np.append(kernel, initial_kernel[digit])
    elif digit == kernel_length - 1:
        remaining_value = sum - np.sum(kernel)
        if remaining_value < 0:
            raise Exception("impossible kernel rejected")
        np.append(kernel, remaining_value)
        kernel_list.append(kernel)
        return



load_kernel(kernel, np.array([]), sum, 0, kernel_length)