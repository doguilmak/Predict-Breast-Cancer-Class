# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:40:53 2021

@author: doguilmak

dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

"""
#%%
# 1. Importing Libraries

from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

#%%
# 2. Data Preprocessing

# 2.1. Importing Data
start = time.time()
data = pd.read_csv('breast-cancer-wisconsin.data', header=None)
data.replace('?', -999999, inplace=True)
data = data.drop([0], axis=1)

# 2.2. Looking For Anomalies
print("\n", data.head())
print("\n", data.describe().T)
print("\n{} duplicated".format(data.duplicated().sum()))

# 2.3. Looking for Duplicated Datas
dp = data[data.duplicated(keep=False)]
dp.head(2)
data.drop_duplicates(inplace= True)
print("{} duplicated\n".format(data.duplicated().sum()))

imputer = SimpleImputer(missing_values= -999999, strategy='mean')
newData = imputer.fit_transform(data)

# 2.4. Determination of Dependent and Independent Variables
X = newData[:, 1:9]
y = newData[:, 9]

#%%
# 3. Artificial Neural Network

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
# 3.1. Creating input layer:
model.add(Dense(64, input_dim=8))
model.add(Activation('relu'))
# 3.2. Creating first hidden layer:
model.add(Dense(64))
model.add(Activation('relu'))
# 3.3. Creating second hidden layer:
model.add(Dense(64))
model.add(Activation('relu'))
# 3.4. Creating third hidden layer:
model.add(Dense(32))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_history = model.fit(X, y, epochs=64, batch_size=32, validation_split=0.13)

# 3.5. Plot accuracy and val_accuracy
print(model_history.history.keys())
plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 3.6. Predicting Class From Model
print('\nANN Prediction')
predict = np.array([8, 8, 5, 4, 5, 10, 4, 1]).reshape(1, 8)
if model.predict_classes(predict) == 2:
    print('Model predicted as benign.')
    print(f'Model predicted class as {model.predict_classes(predict)}.')
else:
    print('Model predicted as maligant.')    
    print(f'Model predicted class as {model.predict_classes(predict)}.')

#%%
# 4 XGBoost

# 4.1. Split Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 4.2. Scaling Datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test) 

from xgboost import XGBClassifier
classifier= XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 4.3. Prediction
print('\nXGBoost Prediction')
predict_model_XGBoost = np.array([8, 8, 5, 4, 5, 10, 4, 1]).reshape(1, 8)
if classifier.predict(predict_model_XGBoost) == 2:
    print('Model predicted as benign.')
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')
else:
    print('Model predicted as maligant.')    
    print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')

# 4.4. Creating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test)  # Comparing results
print("\nConfusion Matrix(XGBoost):\n", cm2)

# 4.5. Accuracy of XGBoost
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
