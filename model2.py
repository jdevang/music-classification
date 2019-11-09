import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from keras import models
from keras import layers

data = pd.read_csv('features.csv')
data = data.drop(['filename'],axis=1)
print(data.head())
print(data.shape)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

# model.add(layers.Dense(128, activation='relu'))

# model.add(layers.Dense(64, activation='relu'))

# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=20, batch_size=128)

# test_loss, test_acc = model.evaluate(X_test,y_test)

# print('test_acc: ',test_acc)

x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=35,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(results)

# predictions = model.predict(X_test)
# print(accuracy_score(y_test, predictions))
# print(predictions[0].shape)
# print(np.sum(predictions[0]))
# print(np.argmax(predictions[0]))
# print(predictions)