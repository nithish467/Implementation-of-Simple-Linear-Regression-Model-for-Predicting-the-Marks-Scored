# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
## Program to implement the simple linear regression model for predicting the marks scored.
## Developed by: NITHISH KUMAR S
## RegisterNumber:  212223240109
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset:

![Screenshot 2024-03-05 092220](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/d8ecbf46-71ad-47b6-84c0-24932c772e7a)

Head values:

![Screenshot 2024-03-05 092310](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/c8457fe4-f459-48be-b669-12c8dbd263f8)


Tail values:

![Screenshot 2024-03-05 092333](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/8d74acd6-683d-426d-ad65-edc04bd52e5e)


X and Y values:

![Screenshot 2024-03-05 092429](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/88e97cf5-27b7-4774-be4e-a48253134571)


Predication values of X and Y:

![Screenshot 2024-03-05 092449](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/fe6b1876-2529-4977-b0ac-4c991616b2d0)



MSE,MAE and RMSE:

![Screenshot 2024-03-05 092516](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/25da824f-6971-4cc8-8e74-52af2e4e2093)


Training Set:

![Screenshot 2024-03-05 092536](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/a7d1ed44-a762-471c-b8c0-3713b901caa1)


Testing Set: 

![Screenshot 2024-03-05 092600](https://github.com/nithish467/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150232274/f000b8ed-517a-4f75-8902-b7f5e84aef39)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
