# Importing necessary modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Reading preprocessed test and train csv files through pandas Dataframe
X_train = pd.read_csv("/content/drive/MyDrive/dataset/DataSet/preprocessed_train_data.csv")
X_test = pd.read_csv("/content/drive/MyDrive/dataset/DataSet/preprocessed_test_data.csv")

# Separating Y data from train dataset and removing extra index columns
X_train = X_train.drop(['Unnamed: 0'], axis=1)
X_test = X_test.drop(['Unnamed: 0'], axis=1)
Y_train = X_train.pop('isFraud')
X_train.head()

# Parameters used for the Neural Network model
lr = 0.5
hidden_layer_act = 'relu'
output_layer_act = 'sigmoid'
no_epochs = 100

# Scaling down the features for better perfomance of the model
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Sequential neural network model
model = Sequential()

# Adding multiple layers to the Sequential model
model.add(Dense(152, input_dim=228, activation=hidden_layer_act))
model.add(Dense(64, activation=hidden_layer_act))
model.add(Dense(32, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))
print(model.summary())

# Training the model with train dataset
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train_sc, Y_train, epochs=no_epochs, batch_size = 2048,  verbose = 2)
predictions = model.predict(X_train_sc)
rounded = [int(round(x[0])) for x in predictions]
predictions.flatten()

# Calculating train accuracy
my_accuracy = accuracy_score(Y_train, predictions.round())
print(my_accuracy)

# Predicting Y for test dataset and storing it in a csv file
predictions = model.predict(X_test_sc)
predictions.flatten()
predictions = np.round(predictions)
Y_test = pd.DataFrame(predictions, columns = ['isFraud'])
Y_test.index.name = "Id"
Y_test.to_csv("/content/NN_fraud.csv")
