# In this part, I will do two tests to test the accuracy of line selection implemented by Random Forest Regression
# The procedure for the two tests are same, the only difference is:
# In test 1, the 25 lines were generated randomly from the 54 lines in the previous part,
# while in test 2, the 25 lines were picked manually from the 54 lines based on some empirical knowledge
# First, rank lines by importance feature of RFR, then use rate of change to rank, and compare these results
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import csv
from RFR_Modelling import col

# Read in the dataset
# The 'Results.csv' file is the output file of the program 'Generate_grids.py'
data = pd.read_csv("Results.csv")

## Test 1: 25 random lines

# Pick 25 lines randomly from col and store them into a list, where 'col' is from the previous part
# slice_rand=random.sample(col,25)
# The lines generated randomly are as follow:
slice_rand=['CO_691.4730763_Flux(K*km/s)','H2S_452.39033_Flux(K*km/s)','CO_230.538_Flux(K*km/s)','H2S_736.0341_Flux(K*km/s)',
'NH3_572.4980678_Flux(K*km/s)','CS_293.9120865_Flux(K*km/s)','HCN_443.1161485_Flux(K*km/s)','CH3CN_312.687755_Flux(K*km/s)',
'CH3CN_220.708949_Flux(K*km/s)','HCN_708.8770051_Flux(K*km/s)','H2S_369.12691_Flux(K*km/s)','NO_602.5609564_Flux(K*km/s)',
'CS_391.8468898_Flux(K*km/s)','NO_464.017124_Flux(K*km/s)','SO_408.6361383_Flux(K*km/s)','SO_611.552412_Flux(K*km/s)',
'NO_360.953456_Flux(K*km/s)','SO_725.1995176_Flux(K*km/s)','CH3CN_239.137925_Flux(K*km/s)','CO_461.0407682_Flux(K*km/s)',
'NO_551.187462_Flux(K*km/s)','CH3CN_275.867681_Flux(K*km/s)','CS_440.803232_Flux(K*km/s)','CH3CN_202.32039_Flux(K*km/s)',
'CH3CN_257.527393_Flux(K*km/s)']

# Use these 25 lines to fit the RFR model first
# First, look at gasTemp parameter, then the same process for other parameters: Density,fc,fo,fn,fs
rand25=data.copy(deep=True)  #copy the dataset to avoid changing the original data
X=rand25[slice_rand].values
Y=rand25.loc[:,'gasTemp'].values  # remember to switch to other parameters
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
test_rand25 = RandomForestRegressor(n_estimators=100,random_state=0,criterion='mse',
                                    oob_score=True, n_jobs=-1,bootstrap=True)
test_rand25.fit(x_train,y_train)
score_rand25=test_rand25.score(x_test,y_test)
y_pred_rand25=test_rand25.predict(x_test)

# Show the results of the RFR model
# Print score and error
print('Score is:', score_rand25)
print('Mean squared error is:', metrics.mean_squared_error(y_test,y_pred_rand25))
print('Mean absolute error is:', metrics.mean_absolute_error(y_test,y_pred_rand25))
# Plot the predicted results vs. true values
plt.figure()
plt.plot(np.arange(100),y_test[:100],"go-",label="true value")  # for simplicity, we only plot the first 100 data
plt.plot(np.arange(100),y_pred_rand25[:100],"ro-",label="predict value")
plt.title("Predict vs. True value (temperature)") # remember to change the title when plotting other parameters
plt.legend(loc="best")

# Rank lines by feature importance
# Produce the importance of each line and rank them
importances = test_rand25.feature_importances_
x_columns=slice_rand
indices=np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f+1,30,slice_rand[indices[f]],importances[indices[f]]))
threshold=0.05
x_selected = x_train[:,importances > threshold]
# Plot the importance in a descending order
plt.figure(figsize=(10,6))
plt.title("Importance of line intensity",fontsize=15)
plt.ylabel("Importance",fontsize=12,rotation=90)
x_columns1 = [x_columns[i] for i in indices]
for i in range(len(x_columns)):
    plt.bar(i,importances[indices[i]],color='blue',align='center')
    plt.xticks(np.arange(len(x_columns)),x_columns1,fontsize=8,rotation=90)
plt.show()

# Calaulate the mean values of each line in the slice_rand
mean_each_randline=[]  #store the mean values in a list
for i in range(len(slice_rand)):
    m=np.mean(data[slice_rand[i]].values)
    mean_each_randline.append(m)

# Define the function of testing regression
# Vary the values of 25 lines one by one, that is, 
# each time only change one line (use the mean value) and keep the others unchanged
def test_regression(i,parameter):
    global score_l
    global y_pred_l
    global rc_l
    global rc_l_abs
    global r_l
    r_l=data.copy(deep=True)
    r_l[slice_rand[i]]=mean_each_randline[i]+1e-5  #for consistency of comparison, each line will be varied by the same increment (1e-5)
    X_l=r_l[slice_rand].values
    Y_l=r_l.loc[:,parameter].values
    x_train_l,x_test_l,y_train_l,y_test_l = train_test_split(X_l,Y_l,test_size=0.3,random_state=42)
    sc_l=StandardScaler()
    x_train_l=sc_l.fit_transform(x_train_l)
    x_test_l=sc_l.fit_transform(x_test_l)
    test_l = RandomForestRegressor(n_estimators=100,random_state=0,criterion='mse',
                                   oob_score=True, n_jobs=-1,bootstrap=True)
    test_l.fit(x_train_l,y_train_l)
    score_l=test_l.score(x_test_l,y_test_l)
    y_pred_l=test_l.predict(x_test_l)
    rc_l=(np.mean(y_pred_l)-np.mean(y_pred_rand25))/1e-5  # calculate the rate of change of each line
    rc_l_abs=np.abs(rc_l)  # take the absolute value of the rate of change

# Test for the six parameters in turn
# Remember to change the index in 'parameter[]' when switch the parameter
parameter=['gasTemp','Density','fc','fo','fn','fs']
rate_of_change=[]  #store the absolute values of rate of change to a list
# Use for loop to change the values of 25 lines one by one
for i in range(len(slice_rand)):
    test_regression(i,parameter[0])  # do gasTemp first, then repeat for the other parameters
    print('Score is:', score_l)
    print('The predicted values for parameter',parameter[0],'(with line',slice_rand[i],'changed)','are:',y_pred_l)
    print('The rate of change for parameter',parameter[0],'(with line',slice_rand[i],'changed)','is:',rc_l)
    rate_of_change.append(rc_l_abs)

# Rank lines by rate of change (absolute values)
roc_des=sorted(enumerate(rate_of_change), reverse=True, key=lambda x:x[1])  # in descending order
print('Based on the descending order of absolute value of the rate of change, in terms of',parameter[0],'the 25 lines are ranked as follows:')
for i in range(len(roc_des)):
    print(slice_rand[roc_des[i][0]])


## Test 2: 25 manual lines

# Pick the lowest (in terms of energy level) 5 transitions for 5 molecules (CO,CH3CN,CS,SO,HCN) from 54 lines
# The lines picked manually are as follow:
slice_manual=['CO_230.538_Flux(K*km/s)','CO_345.7959899_Flux(K*km/s)','CO_461.0407682_Flux(K*km/s)','CO_576.2679305_Flux(K*km/s)',
'CO_691.4730763_Flux(K*km/s)','CH3CN_202.355517_Flux(K*km/s)','CH3CN_202.32039_Flux(K*km/s)','CH3CN_239.137925_Flux(K*km/s)',
'CH3CN_257.527393_Flux(K*km/s)','CH3CN_294.302399_Flux(K*km/s)','CS_244.9355565_Flux(K*km/s)','CS_293.9120865_Flux(K*km/s)',
'CS_342.8828503_Flux(K*km/s)','CS_391.8468898_Flux(K*km/s)','CS_440.803232_Flux(K*km/s)','SO_316.341693_Flux(K*km/s)',
'SO_329.385477_Flux(K*km/s)','SO_286.340152_Flux(K*km/s)','SO_408.6361383_Flux(K*km/s)','SO_236.4522934_Flux(K*km/s)',
'HCN_265.8864339_Flux(K*km/s)','HCN_354.5054773_Flux(K*km/s)','HCN_443.1161485_Flux(K*km/s)','HCN_531.7163479_Flux(K*km/s)',
'HCN_620.3040022_Flux(K*km/s)']

# Since the procedure for the two tests are same, so here I don't replicate the code
# When taking 25 manual lines to test the regression, just replace 'slice_rand' with 'slice_manual', and repeat the process did before
# The word 'rand' in variable names can also be replaced with word 'manual' to avoid confusion