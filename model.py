import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("F:\Documents\Salary.csv")

print(df)

X=df.drop(['Salary'],axis=1)
y=df['Salary']

from sklearn.linear_model import LinearRegression

lg=LinearRegression()

lg.fit(X,y)

pickle.dump(lg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6, 76]]))