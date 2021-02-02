# using physicochemical properties of wine to predict their quality
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Read the data
df = pd.read_csv('winequality-red.csv')
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
y = df['goodquality']
x = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2)

model1 = LogisticRegression().fit(x_train, y_train)
train_score1 = model1.score(x_train, y_train)
test_score1 = model1.score(x_test, y_test)
pred1 = model1.predict(x_test)
print("Logistic Regression train accuracy: " + str(train_score1))
print("Logistic Regression test accuracy: " + str(test_score1))
print(classification_report(y_test, pred1))


model2 = DecisionTreeClassifier().fit(x_train, y_train)
train_score2 = model2.score(x_train, y_train)
test_score2 = model2.score(x_test, y_test)
pred2 = model2.predict(x_test)

print("Decision tree train accuracy: " + str(train_score2))
print("Decision tree  test accuracy: " + str(test_score2))
print(classification_report(y_test, pred2))


model3 = RandomForestClassifier().fit(x_train, y_train)
train_score3= model3.score(x_train, y_train)
test_score3 = model3.score(x_test, y_test)
pred3 = model3.predict(x_test)

print("Decision tree train accuracy: " + str(train_score3))
print("Decision tree  test accuracy: " + str(test_score3))
print(classification_report(y_test, pred3))




