print(__doc__)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = datasets.load_diabetes()

# Use only one feature
data_X = data.data[:, np.newaxis, 2]

# Split the data into training/testing sets
data_X_train = data_X[:-20]
data_X_test = data_X[-20:]

# Split the targets into training/testing sets
data_y_train = data.target[:-20]
data_y_test = data.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_X_train, data_y_train)

# Make predictions using the testing set
data_y_pred = regr.predict(data_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(data_y_test, data_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(data_y_test, data_y_pred))

# Plot outputs
plt.scatter(data_X_test, data_y_test, color='black')
plt.plot(data_X_test, data_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
