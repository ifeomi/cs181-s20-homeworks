import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import linear_model, neural_network, neighbors
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from matplotlib import pyplot as plt


boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=181)
NUM_SPLITS = 5

# print this to get a description of data
# print(boston.DESCR)

# Helper function to do n-fold cross validation
def cross_val(regression_model):
	kf = KFold(n_splits=NUM_SPLITS, shuffle=True)
	avg_mse = 0
	for train, test in kf.split(X_train):
		reg = regression_model.fit(X_train[train], y_train[train])
		preds = reg.predict(X_train[test])
		avg_mse += mean_squared_error(y_train[test], preds) / NUM_SPLITS
	return avg_mse

def test(model):
	y_test_pred = model.predict(X_test)
	return mean_squared_error(y_test, y_test_pred)

lin_reg = linear_model.LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_train)

print("Linear regression mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))
# print("Linear regression average cross-validation MSE: ", cross_val(linear_model.LinearRegression()))

print("Test mean squared error: %.2f"
      % test(lin_reg))

ridge_reg = linear_model.Ridge().fit(X_train, y_train)
y_pred = ridge_reg.predict(X_train)

print("Ridge regression mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))

print("Ridge regression average cross-validation MSE: ", cross_val(linear_model.Ridge()))

print("Test mean squared error: %.2f"
      % test(ridge_reg))

mlp = neural_network.MLPRegressor(hidden_layer_sizes=(75, 75),max_iter=3000, random_state=181, solver='lbfgs').fit(X_train, y_train)
y_pred = mlp.predict(X_train)

print("MLP mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))
print("MLP average cross-validation MSE: ", cross_val(mlp))
print("MLE test MSE: ", test(mlp))

parameters = {'hidden_layer_sizes':[(75, 75)], 'max_iter': [x for x in range(1000, 9000, 2000)], 'solver':['lbfgs'], 'activation':['relu'], 'alpha':[0.0001, 0.001, 0.01]}
# parameters = {'hidden_layer_sizes':((75, 75), (200, 200)), 'activation':['tanh', 'relu']}
mlp = neural_network.MLPRegressor()
clf = GridSearchCV(estimator=mlp, scoring='neg_mean_squared_error', param_grid=parameters, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_params_)

print("TUNED-----")
mlp = neural_network.MLPRegressor(activation='relu', alpha=0.001, hidden_layer_sizes=(75, 75), solver='lbfgs', max_iter=5000, random_state=181).fit(X_train, y_train)
y_pred = mlp.predict(X_train)

print("MLP mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))
print("MLP average cross-validation MSE: ", cross_val(mlp))
print("MLE test MSE: ", test(mlp))

rf = RandomForestRegressor(random_state=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)
print("Random forest mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))
print("Random forest average cross-validation MSE: ", cross_val(rf))
print(test(rf))

knn = neighbors.KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train)
print("KNN mean squared error: %.2f"
      % mean_squared_error(y_train, y_pred))
print("KNN average cross-validation MSE: ", cross_val(knn))
print(test(knn))

