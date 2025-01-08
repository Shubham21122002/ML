import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("birthwt.csv")
df.head()


age = df["age"]
age = age.to_numpy()
birthwt = df["bwt"]
birthwt = birthwt.to_numpy()
plt.scatter(age,birthwt/1000)

motherwt = df["lwt"]
lr = LinearRegression()
age = age.reshape(-1,1)
lr.fit(age,birthwt)
y = lr.predict(age)
print("Coefficient: ",lr.coef_[0])
print("Intercept: ",lr.intercept_)

plt.scatter(age,motherwt)

age = age.reshape(-1,1)
lr.fit(age,motherwt)
y = lr.predict(age)
print("Coefficient: ",lr.coef_[0])
print("Intercept: ",lr.intercept_)

birthwt = birthwt.reshape(-1,1)
lr.fit(birthwt,motherwt)
y = lr.predict(birthwt)
print("Coefficient: ",lr.coef_[0])
print("Intercept: ",lr.intercept_)

def correlation(dataset1,dataset2):
    cov = calc_covariance(dataset1,dataset2)
    sd1 = np.std(dataset1)
    sd2 = np.std(dataset2)
    return cov/(sd1*sd2)

plt.scatter(age,birthwt/1000,c='r')
plt.xlabel("Age")
plt.ylabel("Birth_Weight")
