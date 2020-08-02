import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from pathlib import Path

Datapath = Path(__file__).parent / "../data_list/Rig_Count_vs._Spot_Price.csv"
data=pd.read_csv(Datapath)

#############Old data model containing all data axes
# print(data.head(2))
#Oil_Price_y = data["Oil_Price"][:, np.newaxis]
#Rig_count_x= data["Rig_count"][:, np.newaxis]
# model = LinearRegression().fit(Rig_count_x, Oil_Price_y)
# r_sq = model.score(Rig_count_x, Oil_Price_y)
# print('coefficient of determination:', r_sq)
# print('intercept:', model.intercept_)
# print('slope:', model.coef_)
# Oil_Price_y_pred = model.predict(Rig_count_x)
# data.plot.scatter("Rig_count","Oil_Price",label='Data')
# plt.plot(Rig_count_x, Oil_Price_y_pred, color='red', label='Linear regression')
# plt.title("Oil price vs Rig count")
# plt.grid()
# plt.legend()
# plt.show()

#data selected from an specific date， to make it
desire_date=pd.Timestamp('01/01/2010')
data['Date'] = pd.to_datetime(data['Date'])
condition = data['Date'] > desire_date
data_2010_2020=data[condition]


#lower cluster
data1 = data_2010_2020[data_2010_2020["Oil_Price"]<71]
data2 = data_2010_2020[data_2010_2020["Oil_Price"]>=71]

Oil_Price_x = data1["Oil_Price"][:, np.newaxis]
Rig_count_y= data1["Rig_count"][:, np.newaxis]
model = LinearRegression().fit(Oil_Price_x,Rig_count_y)
r_sq = model.score(Oil_Price_x,Rig_count_y )
print("----------Lower Cluster------------")
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
Rig_count_y_pred = model.predict(Oil_Price_x)
data1.plot.scatter("Oil_Price","Rig_count",label='Data')
plt.plot(Oil_Price_x, Rig_count_y_pred, color='red', label='Linear regression')
plt.title("Oil price vs Rig count")
plt.grid()
plt.legend()
plt.show()
print("Stat. Summary","\n")
print(data1.describe())


#Upper Closter (data2)
Oil_Price_x = data2["Oil_Price"][:, np.newaxis]
Rig_count_y= data2["Rig_count"][:, np.newaxis]
model = LinearRegression().fit(Oil_Price_x,Rig_count_y)
r_sq = model.score(Oil_Price_x,Rig_count_y )
print("----------Upper Cluster------------")
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
Rig_count_y_pred = model.predict(Oil_Price_x)
data2.plot.scatter("Oil_Price","Rig_count",label='Data')
plt.plot(Oil_Price_x, Rig_count_y_pred, color='red', label='Linear regression')
plt.title("Oil price vs Rig count")
plt.grid()
plt.legend()
plt.show()
print("Stat. Summary","\n")
print(data2.describe())


#Lower Cluster Prediction
# Oil Price training/testing sets
Rig_count_y_train = data1["Rig_count"][:-30][:, np.newaxis]
Rig_count_y_test = data1["Rig_count"][-30:][:, np.newaxis]

# Rig count training/testing sets
Oil_Price_x_train = data1["Oil_Price"][:-30][:, np.newaxis]
Oil_Price_x_test = data1["Oil_Price"][-30:][:, np.newaxis]

# linear regression object
regr = linear_model.LinearRegression()

# Train the model
regr.fit(Oil_Price_x_train,Rig_count_y_train)

# Make predictions
Rig_count_y_predic = regr.predict(Oil_Price_x_test)

# The coefficients
print('Coefficients:', regr.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(Rig_count_y_test, Rig_count_y_predic))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (y test vs y predict): %.2f'
      % r2_score(Rig_count_y_test, Rig_count_y_predic))
r_sq = model.score(Oil_Price_x_test, Rig_count_y_predic)
print('coefficient of determination (x test vs y predict):', r_sq)
# Plot outputs
plt.scatter(Oil_Price_x_test,Rig_count_y_test,  color='skyblue',label='Data')
plt.plot(Oil_Price_x_test, Rig_count_y_predic, color='red', linewidth=2,label='Linear regression')
plt.ylabel('Rig count')
plt.xlabel('Oil Price')
plt.title("Predic. Oil price vs Rig count")
plt.legend()
plt.show()

#Upper Cluster Prediction
# Oil Price training/testing sets
Rig_count_y_train = data2["Rig_count"][:-30][:, np.newaxis]
Rig_count_y_test = data2["Rig_count"][-30:][:, np.newaxis]

# Rig count training/testing sets
Oil_Price_x_train = data2["Oil_Price"][:-30][:, np.newaxis]
Oil_Price_x_test = data2["Oil_Price"][-30:][:, np.newaxis]

# linear regression object
regr = linear_model.LinearRegression()

# Train the model
regr.fit(Oil_Price_x_train,Rig_count_y_train)

# Make predictions
Rig_count_y_predic = regr.predict(Oil_Price_x_test)

# The coefficients
print('Coefficients:', regr.coef_)

print('Mean squared error: %.2f'
      % mean_squared_error(Rig_count_y_test, Rig_count_y_predic))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (y test vs y predict): %.2f'
      % r2_score(Rig_count_y_test, Rig_count_y_predic))
r_sq = model.score(Oil_Price_x_test, Rig_count_y_predic)
print('coefficient of determination (x test vs y predict):', r_sq)
# Plot outputs
plt.scatter(Oil_Price_x_test,Rig_count_y_test,  color='skyblue',label='Data')
plt.plot(Oil_Price_x_test, Rig_count_y_predic, color='red', linewidth=2,label='Linear regression')
plt.ylabel('Rig count')
plt.xlabel('Oil Price')
plt.title("Predic. Oil price vs Rig count")
plt.legend()
plt.show()




# Oil_Price_y = data_2010_2020["Oil_Price"][:, np.newaxis]
# Rig_count_x= data_2010_2020["Rig_count"][:, np.newaxis]
# model = LinearRegression().fit(Rig_count_x, Oil_Price_y)
# r_sq = model.score(Rig_count_x, Oil_Price_y)
# print('coefficient of determination:', r_sq)
# print('intercept:', model.intercept_)
# print('slope:', model.coef_)
# Oil_Price_y_pred = model.predict(Rig_count_x)
# data_2010_2020.plot.scatter("Rig_count","Oil_Price",label='Data')
# plt.plot(Rig_count_x, Oil_Price_y_pred, color='red',label='Linear regression')
# plt.title("Oil price vs Rig count 2010-2020")
# plt.grid()
# plt.legend()
# plt.show()
#
# Oil_Price_y_train = data_2010_2020["Oil_Price"][:-30][:, np.newaxis]
# Oil_Price_y_test = data_2010_2020["Oil_Price"][-30:][:, np.newaxis]
#
# # Rig count training/testing sets
# Rig_count_x_train = data_2010_2020["Rig_count"][:-30][:, np.newaxis]
# Rig_count_x_test = data_2010_2020["Rig_count"][-30:][:, np.newaxis]
#
# # Create linear regression object
# regr = linear_model.LinearRegression()
#
# # Train the model
# regr.fit(Rig_count_x_train, Oil_Price_y_train)
#
# # Make predictions
# Oil_Price_y_predic = regr.predict(Rig_count_x_test)
#
# # The coefficients
# print('Coefficients:', regr.coef_)
#
# print('Mean squared error: %.2f'
#       % mean_squared_error(Oil_Price_y_test, Oil_Price_y_predic))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(Oil_Price_y_test, Oil_Price_y_predic))
#
# # Plot outputs
# plt.scatter(Rig_count_x_test, Oil_Price_y_test,  color='skyblue',label='Data')
# plt.plot(Rig_count_x_test, Oil_Price_y_predic, color='red', linewidth=2,label='Linear regression')
# plt.xlabel('Rig count')
# plt.ylabel('Oil Price')
# plt.legend()
# plt.show()