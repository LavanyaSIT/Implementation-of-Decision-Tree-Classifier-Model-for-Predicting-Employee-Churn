# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
# AIM:

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
# Equipments Required:

    1. Hardware – PCs
    2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.
# Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Aagalya R 
RegisterNumber: 212222040003

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```
# Output:
# data.head()
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/7621b9e5-6d85-4960-bd38-60dbcf753a0e)

# data.info()
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/4fdd68c2-bbc8-46d3-8eeb-ee4760f95d32)

# isnull() and sum ()
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/8fc9ceca-f352-411d-831f-1a9918db2cb3)

# data value counts()
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/adb1bb93-3d19-4980-b3bb-b34a6d3eecfb)

# data.head() for salary
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/843a2433-da0a-4f8a-816e-bf06d1ef2f85)

# x.head()
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/8e9b219d-e3a6-4d67-bd32-bf223aee3a30)

# accuracy value
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/fed9d18c-794c-4a6d-a62d-a1d83a6eb76d)

# data prediction
![image](https://github.com/LavanyaSIT/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/130207418/d327903e-d958-443b-a4bf-91b0e92a8317)

# Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
