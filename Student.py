from numpy.random import logistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.metrics import  confusion_matrix, classification_report

# Load the dataset

df = pd.read_csv("student_success_dataset.csv")
print("The Data is as follows ")
print(df.head())


print("Data shape")
print("Column :",df.shape[1])
print("Rows :",df.shape[0])
 

print("Dataset information")
print(df.info())


print("Summary of the data")
print(df.describe())


print("Missing value in data ")
print(df.isnull().sum())


#Preprosessing of data
le = LabelEncoder()
df["Internet"] = le.fit_transform(df["Internet"])
df["Passed"] = le.fit_transform(df["Passed"])
print("Lable encoding done ")

print(df)

fectures=["StudyHours","Attendance","PastScore","Internet","SleepHours"]
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[fectures]=scaler.fit_transform(df[fectures])


X = df_scaled[fectures]
y=df["Passed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)