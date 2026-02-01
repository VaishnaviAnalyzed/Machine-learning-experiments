import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\PC World\OneDrive\Documents\Practice Machine Learning\Regression model (SLR)\Poly Explanation\emp_sal.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#LinearRegression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred 

#PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5) 
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicton 
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


# SVR model 
from sklearn.svm import SVR
svr_reg = SVR()
svr_reg.fit(X, y)

svr_model_pred = svr_reg.predict([[6.5]])
svr_model_pred 

# knn regression model 
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=4)
knn_reg.fit(X,y)

knn_model_pred = knn_reg.predict([[6.5]])
knn_model_pred

# dtr model 
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X,y)
 
dt_model_pred = dt_reg.predict([[6.5]])
dt_model_pred


# rfr model 
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X,y)

rf_model_pred = rf_reg.predict([[6.5]])
rf_model_pred
















