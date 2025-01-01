import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

#print(type(data))

X = data.data  

#print(type(X))

y = data.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in kernels:
    print(f"\nTraining SVM with {k} kernel :")

    model = SVC(kernel=k, gamma=0.5, C=1.0)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy of the SVM with {k} kernel: {accuracy * 100:.2f}%")


