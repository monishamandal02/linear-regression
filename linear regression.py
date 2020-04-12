import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

plt.title("Indian Women")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.scatter(df.Height,df.Weight,color='blue')
plt.show()

x = df[['Height']]
y = df['Weight']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print('x_train shape:', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape', y_test.shape)
print('percent in x_train:', x_train.shape[0]/(x_train.shape[0] + x_test.shape[0]))
print('percent in x_test:', x_test.shape[0]/(x_train.shape[0] + x_test.shape[0]))

model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)
print(model.predict([[1.68]]))