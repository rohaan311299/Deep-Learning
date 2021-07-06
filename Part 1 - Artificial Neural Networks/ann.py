# Importing the libraries
import numpy as np 
import pandas as pd 
import tensorflow as tf 
tf.__version__

# Data Preprocessing

# importing the dataset
dataset=pd.read_csv("./Churn_Modelling.csv")
X=dataset.iloc[:, 3:-1].values
y=dataset.iloc[:, :-1].values

# Encoding categorical data

# Label encoding the "Gender" variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:, 2]=le.fit_transform(X[:,2])

# One hot encoding the geography problem
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X=np.array(ct.fit_transform(X))

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Building the ANN

# initilaizing the ANN
ann=tf.keras.models.Sequential()

# adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# training the ANN

# compiling the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# making the prediction and evaluating the model
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))>0.5)

# predicting the test set results
y_pred=ann.predict(X_test)
y_pred=y_pred >0.5
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))