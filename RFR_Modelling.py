# In this part, I will use the Random Forest to do regression modelling
# Import some relevant packages such as sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

# Read in the dataset
# The 'Results.csv' file is the output file of the program 'Generate_grids.py' 
data = pd.read_csv("Results.csv")

# Take a look at the structure of the dataset
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data.head()

# Pick 54 molecular emission lines from LAMDA database based on J values
col=['CH3CN_202.355517_Flux(K*km/s)', 'CH3CN_202.32039_Flux(K*km/s)', 'CH3CN_220.708949_Flux(K*km/s)',
'CH3CN_239.137925_Flux(K*km/s)', 'CH3CN_257.527393_Flux(K*km/s)', 'CH3CN_275.867681_Flux(K*km/s)',
'CH3CN_294.302399_Flux(K*km/s)', 'CH3CN_312.687755_Flux(K*km/s)', 'CO_230.538_Flux(K*km/s)', 'CO_345.7959899_Flux(K*km/s)',
'CO_461.0407682_Flux(K*km/s)', 'CO_576.2679305_Flux(K*km/s)', 'CO_691.4730763_Flux(K*km/s)', 'CS_244.9355565_Flux(K*km/s)',
'CS_293.9120865_Flux(K*km/s)', 'CS_342.8828503_Flux(K*km/s)', 'CS_391.8468898_Flux(K*km/s)', 'CS_440.803232_Flux(K*km/s)',
'CS_489.750921_Flux(K*km/s)', 'CS_538.6889972_Flux(K*km/s)', 'CS_587.616485_Flux(K*km/s)', 'HCN_265.8864339_Flux(K*km/s)',
'HCN_354.5054773_Flux(K*km/s)', 'HCN_443.1161485_Flux(K*km/s)', 'HCN_531.7163479_Flux(K*km/s)', 'HCN_620.3040022_Flux(K*km/s)',
'HCN_708.8770051_Flux(K*km/s)', 'HCN_797.4332623_Flux(K*km/s)', 'NO_249.635646_Flux(K*km/s)', 'NO_257.852746_Flux(K*km/s)',
'NO_360.953456_Flux(K*km/s)', 'NO_401.6858953_Flux(K*km/s)', 'NO_450.93954_Flux(K*km/s)', 'NO_464.017124_Flux(K*km/s)',
'NO_551.187462_Flux(K*km/s)', 'NO_602.5609564_Flux(K*km/s)', 'SO_316.341693_Flux(K*km/s)', 'SO_329.385477_Flux(K*km/s)',
'SO_286.340152_Flux(K*km/s)', 'SO_408.6361383_Flux(K*km/s)', 'SO_236.4522934_Flux(K*km/s)', 'SO_504.6762856_Flux(K*km/s)',
'SO_611.552412_Flux(K*km/s)', 'SO_725.1995176_Flux(K*km/s)', 'H2S_736.0341_Flux(K*km/s)', 'H2S_505.56523_Flux(K*km/s)',
'H2S_369.10145_Flux(K*km/s)', 'H2S_228.55627_Flux(K*km/s)', 'H2S_452.39033_Flux(K*km/s)', 'H2S_687.30347_Flux(K*km/s)',
'H2S_216.71044_Flux(K*km/s)', 'H2S_747.30189_Flux(K*km/s)', 'H2S_369.12691_Flux(K*km/s)', 'NH3_572.4980678_Flux(K*km/s)']

# # First, try Linear regression model to have a rough feeling about the regression
# # It turns out the linear regression does not perform well for our project, so we don't consider it
# X_linear=data[col].values
# Y_linear=data.loc[:,'gasTemp'].values
# # Split the dataset into train set and test set
# x_train,x_test,y_train,y_test = train_test_split(X_linear,Y_linear,test_size=0.3,random_state=42)
# linear_reg=LinearRegression()
# linear_reg.fit(x_train,y_train)
# score_linear=linear_reg.score(x_test,y_test)  #-1.2433990065156434e+85
# y_pred_linear=linear_reg.predict(x_test)


# Then, do the Random Forest regression (RFR) modelling for six parameters in turn
# Divide the parameter ranges into several sub-ranges
# Modelling on the full range first, then on the sub-intervals in turn, in order to compare the difference

# RFR model for parameter - gas temperature
df1=data.copy(deep=True)  # copy the dataset to avoid changing the original values
# df1=data[(data['gasTemp'] >= 10)& (data['gasTemp'] < 80)]
# df1=data[(data['gasTemp'] >= 80)& (data['gasTemp'] < 150)]
# df1=data[(data['gasTemp'] >= 150)& (data['gasTemp'] < 220)]
# df1=data[(data['gasTemp'] >= 220)& (data['gasTemp'] <= 300)]
X_temp=df1[col].values
Y_temp=df1.loc[:,'gasTemp'].values
# Split the dataset into training set and testing set
x_train_temp,x_test_temp,y_train_temp,y_test_temp = train_test_split(X_temp,Y_temp,test_size=0.3,random_state=42)
# Standardize the dataset
sc=StandardScaler()
x_train_temp=sc.fit_transform(x_train_temp)
x_test_temp=sc.fit_transform(x_test_temp)
# Fit the RFR model with 500 trees in the forest
forest_temp = RandomForestRegressor(n_estimators=500, random_state=0, criterion='mse',
                                    oob_score=True, n_jobs=-1, bootstrap=True)
forest_temp.fit(x_train_temp, y_train_temp)
score_temp = forest_temp.score(x_test_temp, y_test_temp)  # score indicates the performance of the model
y_pred_temp = forest_temp.predict(x_test_temp)

# RFR model for parameter - density
df2=data.copy(deep=True)
# df2=data[(data['Density'] >= 1e3)& (data['Density'] < 1e4)]
# df2=data[(data['Density'] >= 1e4)& (data['Density'] < 1e5)]
# df2=data[(data['Density'] >= 1e5)& (data['Density'] < 1e6)]
# df2=data[(data['Density'] >= 1e6)& (data['Density'] <= 1e7)]
X_dense=df2[col].values
Y_dense=df2.loc[:,'Density'].values
x_train_dense,x_test_dense,y_train_dense,y_test_dense = train_test_split(X_dense,Y_dense,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train_dense=sc.fit_transform(x_train_dense)
x_test_dense=sc.fit_transform(x_test_dense)
forest_dense = RandomForestRegressor(n_estimators=500,random_state=0,criterion='mse',
                                     oob_score=True, n_jobs=-1,bootstrap=True)
forest_dense.fit(x_train_dense, y_train_dense)
score_dense = forest_dense.score(x_test_dense, y_test_dense)
y_pred_dense = forest_dense.predict(x_test_dense)

# RFR model for parameter - Oxygen abundance
df3=data.copy(deep=True)
# df3=data[(data['fo'] >= 1e-5)& (data['fo'] < 1e-4)]
# df3=data[(data['fo'] >= 1e-4)& (data['fo'] <= 1e-3)]
X_oxg=df3[col].values
Y_oxg=df3.loc[:,'fo'].values
x_train_oxg,x_test_oxg,y_train_oxg,y_test_oxg = train_test_split(X_oxg,Y_oxg,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train_oxg=sc.fit_transform(x_train_oxg)
x_test_oxg=sc.fit_transform(x_test_oxg)
forest_oxg = RandomForestRegressor(n_estimators=500, random_state=0, criterion='mse',
                                   oob_score=True, n_jobs=-1, bootstrap=True)
forest_oxg.fit(x_train_oxg,y_train_oxg)
score_oxg = forest_oxg.score(x_test_oxg,y_test_oxg)
y_pred_oxg = forest_oxg.predict(x_test_oxg)

# RFR model for parameter - Carbon abundance
df4=data.copy(deep=True)
# df4=data[(data['fc'] >= 1e-5)& (data['fc'] < 1e-4)]
# df4=data[(data['fc'] >= 1e-4)& (data['fc'] <= 1e-3)]
X_carbon=df4[col].values
Y_carbon=df4.loc[:,'fc'].values
x_train_carbon,x_test_carbon,y_train_carbon,y_test_carbon = train_test_split(X_carbon,Y_carbon,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train_carbon=sc.fit_transform(x_train_carbon)
x_test_carbon=sc.fit_transform(x_test_carbon)
forest_carbon = RandomForestRegressor(n_estimators=500, random_state=0, criterion='mse',
                                      oob_score=True, n_jobs=-1, bootstrap=True)
forest_carbon.fit(x_train_carbon, y_train_carbon)
score_carbon = forest_carbon.score(x_test_carbon, y_test_carbon)
y_pred_carbon = forest_carbon.predict(x_test_carbon)

# RFR model for parameter - Nitrogen abundance
df5=data.copy(deep=True)
# df5=data[(data['fn'] >= 1e-6)& (data['fn'] < 1e-5)]
# df5=data[(data['fn'] >= 1e-5)& (data['fn'] <= 1e-4)]
X_Nit=df5[col].values
Y_Nit=df5.loc[:,'fn'].values
x_train_Nit,x_test_Nit,y_train_Nit,y_test_Nit = train_test_split(X_Nit,Y_Nit,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train_Nit=sc.fit_transform(x_train_Nit)
x_test_Nit=sc.fit_transform(x_test_Nit)
forest_Nit = RandomForestRegressor(n_estimators=500, random_state=0, criterion='mse',
                                   oob_score=True, n_jobs=-1, bootstrap=True)
forest_Nit.fit(x_train_Nit, y_train_Nit)
score_Nit = forest_Nit.score(x_test_Nit, y_test_Nit)
y_pred_Nit = forest_Nit.predict(x_test_Nit)

# RFR model for parameter - Sulfur abundance
df6=data.copy(deep=True)
# df6=data[(data['fs'] >= 1e-7)& (data['fs'] < 1e-6)]
# df6=data[(data['fs'] >= 1e-6)& (data['fs'] <= 1e-5)]
X_sul=df6[col].values
Y_sul=df6.loc[:,'fs'].values
x_train_sul,x_test_sul,y_train_sul,y_test_sul = train_test_split(X_sul,Y_sul,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train_sul=sc.fit_transform(x_train_sul)
x_test_sul=sc.fit_transform(x_test_sul)
forest_sul = RandomForestRegressor(n_estimators=500, random_state=0, criterion='mse',
                                   oob_score=True, n_jobs=-1, bootstrap=True)
forest_sul.fit(x_train_sul, y_train_sul)
score_sul=forest_sul.score(x_test_sul, y_test_sul)
y_pred_sul=forest_sul.predict(x_test_sul)

# Show the score and error, which are the measure of model performance
# This is an example for temperature parameter, and variable name can be changed to show other parameters
print('Score is:', score_temp)
print('Mean squared error is:', metrics.mean_squared_error(y_test_temp,y_pred_temp))
print('Mean absolute error is:', metrics.mean_absolute_error(y_test_temp,y_pred_temp))

# Plot the predicted results of the regression model
# Also, this is an example for temperature, and variable name can be changed to show other parameters
plt.figure()
plt.plot(np.arange(100),y_test_temp[:100],"go-",label="true value")  # for brevity, we only plot 100 data
plt.plot(np.arange(100),y_pred_temp[:100],"ro-",label="predict value")
plt.title("Predicted vs. True value (temperature)")  # remember to change the title when plotting other parameters
plt.legend(loc="best")

# Analysis of the feature importance
# Produce the importance of each line and rank them by the importance evaluation
# Again, an example for temperature, and change variable name to show other parameters
importances = forest_temp.feature_importances_
x_columns=col
indices=np.argsort(importances)[::-1]
for f in range(x_train_temp.shape[1]):
    print("%2d) %-*s %f" % (f+1,30,col[indices[f]],importances[indices[f]]))
threshold=0.05  # Set the threshold to be 0.05
x_selected_temp = x_train_temp[:,importances > threshold]  # we only focus on the lines whose importance is greater than the threshold
# Plot the importance in a descending order
plt.figure(figsize=(10,6))
plt.title("Importance of line intensity", fontsize=15)
plt.ylabel("Importance", fontsize=12, rotation=90)
x_columns1 = [x_columns[i] for i in indices]
for i in range(len(x_columns)):
    plt.bar(i, importances[indices[i]], color='blue', align='center')
    plt.xticks(np.arange(len(x_columns)), x_columns1, fontsize=8, rotation=90)
plt.show()