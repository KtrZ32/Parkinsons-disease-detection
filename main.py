import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

df = pd.read_csv('parkinsons.data.csv')
signs=df.loc[:,df.columns!='status'].values[:,1:]
status=df.loc[:,'status'].values

scaler=MinMaxScaler((0,1))
Scaled_signs=scaler.fit_transform(signs)

X_train, X_test, y_train, y_test = train_test_split(Scaled_signs, status, test_size=0.2, random_state=8, stratify=status)

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(f'Точность: {round(score*100,2)}%')

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Матрица ошибок', pad=15)
plt.show()