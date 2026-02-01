import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\PC World\OneDrive\Documents\Practice Machine Learning\Regression model (SLR)\SLR - Salary Prediction\Salary_Data.csv")

x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values  

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=0) 


from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(x_train, y_train)  

y_pred =  regressor.predict(x_test)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison) 

plt.scatter(x_test, y_test, color = 'red')   
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


print(regressor)

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

emp_15 = m_slope* 15 + c_intercept
print(emp_15)

emp_20 = m_slope* 20 + c_intercept
print(emp_20)


bias = regressor.score(x_train, y_train)
print(bias) 

variance = regressor.score(x_test, y_test)
print(variance) 

# statstics topics 

#STATISTICS FOR MACHINE LEARNING

dataset.mean()

dataset['Salary'].mean() 

dataset.median() 

dataset['Salary'].mode()

dataset.describe()

dataset.var() 

dataset.std()

dataset.corr()

#Coefficient of variation(cv)
from scipy.stats import variation
variation(dataset.values)

#Correlation
dataset.corr() 
dataset['Salary'].corr(dataset['YearsExperience']) 

#Skewness
dataset.skew()
dataset['Salary'].skew() 

#Standard Error
dataset.sem() 
dataset['Salary'].sem()

#Z-score
import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

#Degree of Freedom
a = dataset.shape[0] # this will gives us no.of rows
b = dataset.shape[1] # this will give us no.of columns
degree_of_freedom = a-b
print(degree_of_freedom) 


#ssr 
y_mean = np.mean(y) 
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#sse 
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#sst 
mean_total = np.mean(dataset.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2 
r_square = 1 - SSR/SST
print(r_square)


# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)

# MODEL PERFORMANCE SECTION

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

# R2 Scores
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# MSE
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# RMSE
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

# MAE
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Output
print(f"Training R² Score: {train_r2:.2f}")
print(f"Testing R² Score: {test_r2:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training MAE: {train_mae:.2f}")
print(f"Testing MAE: {test_mae:.2f}")

