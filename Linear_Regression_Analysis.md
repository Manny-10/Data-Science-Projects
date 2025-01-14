from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:, 0:11]
y = BosData.iloc[:, 13]
Xtrain, X_test, ytrain, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

reg = LinearRegression()
reg.fit(Xtrain, ytrain)

ytrainpredict = reg.predict(Xtrain)
mse_train = mean_squared_error(ytrain, ytrainpredict)
r2_train = r2_score(ytrain, ytrainpredict)
print('Train MSE =', mse_train)
print('Train R2 score =', r2_train)
print("\n")
ytestpredict = reg.predict(X_test)
mse_test = mean_squared_error(y_test, ytestpredict)
r2_test = r2_score(y_test, ytestpredict)
print('Test MSE =', mse_test)
print('Test R2 score =', r2_test)
plt.figure()
plt.scatter(y_test, ytestpredict, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.grid()
plt.show()
