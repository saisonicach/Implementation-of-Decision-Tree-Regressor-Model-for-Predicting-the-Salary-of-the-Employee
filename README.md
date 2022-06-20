# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.



## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sai sonica CH
RegisterNumber: 212219040130 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Data Head:
![image](https://user-images.githubusercontent.com/79306169/174666993-51ad8fd9-b8f3-4558-bc4e-07e0b9893718.png)
## Data Info:
![image](https://user-images.githubusercontent.com/79306169/174667048-c6b0cfd0-3638-4423-b5e7-443a570edb02.png)
## Data Isnull:
![image](https://user-images.githubusercontent.com/79306169/174667110-7fbb1b1e-0f49-4dad-9ad6-dfef3c7bc5c0.png)
## Data Head:
![image](https://user-images.githubusercontent.com/79306169/174667149-90712e6a-acda-4872-9e06-3fca02ded971.png)
## MSE:
![image](https://user-images.githubusercontent.com/79306169/174667216-641473d9-6500-4a4a-a558-ac79d3d2e609.png)
## R2:
![image](https://user-images.githubusercontent.com/79306169/174667269-a55d7bb6-1ed6-43c4-be0b-34a6598c7b0d.png)
## Predicted Value:
![image](https://user-images.githubusercontent.com/79306169/174667315-9ea40dfe-e943-4f45-9ebe-065b096e8d75.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
