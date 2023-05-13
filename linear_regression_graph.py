import numpy as np
import random
import matplotlib.pyplot as plt

# Generate some random data
n = 20  # number of data points
x = np.array([random.uniform(0, 10) for i in range(n)])
y = np.array([x[i] + random.gauss(0, 1) for i in range(n)])  # linear function with noise

# Fit a linear regression line
m, b = np.polyfit(x, y, 1)

# Calculate the predicted values of y
y_pred = m*x + b

# Calculate the errors
errors = y - y_pred

# Create the figure and axis objects
fig, ax = plt.subplots()

# Plot the scatter plot
ax.scatter(x, y, color='black', marker='o', facecolor='white')

# Plot the linear regression line
ax.plot(x, y_pred, color='black')

# Plot the error bars
ax.vlines(x, y_pred, y, color='0.4')

# Add labels and title
ax.set_xlabel('X Values')
ax.set_ylabel('Y Values')
ax.set_title('Linear Regression Example')

# Show the plot
plt.show()