import numpy as np

# Create a NumPy array
my_array = np.array([1, 2, 3, 4, 5])

# Create an array of coordinates using np.meshgrid
x, y = np.meshgrid(my_array, my_array)

# Stack the x and y arrays along the last axis to create a 2D array of coordinates
coordinates = np.stack((x, y), axis=-1)

# Print the array of coordinates
print(coordinates)
breakpoint()