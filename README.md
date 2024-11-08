# CIFAR-10-Testing-With-Different-Hyperparameters


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import time

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
import time
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0    #Normalization
y_train, y_test = to_categorical(y_train), to_categorical(y_test) # one hot encoding
basemodel = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu'),  # Hidden Layer 1
    Dense(128, activation='relu'),  # Hidden Layer 2
    Dense(64, activation='relu'),   # Hidden Layer 3
    Dense(10, activation='softmax') # Output Layer
])
basemodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st=time.time()
history = basemodel.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et=time.time()
basetest_loss, basetest_acc = basemodel.evaluate(x_test, y_test)
duration=et-st
print(f"Duration:{duration:.4f}")
print(f"Test accuracy: {basetest_acc:.4f}")
print(f"Test loss: {basetest_loss:.4f}")


xaviermodel = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu',kernel_initializer='glorot_uniform'),  # Hidden Layer 1
    Dense(128, activation='relu',kernel_initializer='glorot_uniform'),  # Hidden Layer 2
    Dense(64, activation='relu',kernel_initializer='glorot_uniform'),   # Hidden Layer 3
    Dense(10, activation='softmax') # Output Layer
])

# Compile the model
xaviermodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st=time.time()

# Train the model
history = xaviermodel.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et=time.time()

# Evaluate the model
xaviertest_loss, xaviertest_acc = xaviermodel.evaluate(x_test, y_test)

duration=et-st

print(f"Duration:{duration:.4f}")
print(f"Test accuracy: {xaviertest_acc:.4f}")
print(f"Test loss: {xaviertest_loss:.4f}")


hemodel = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu',kernel_initializer='he_normal'),  # Hidden Layer 1
    Dense(128, activation='relu',kernel_initializer='he_normal'),  # Hidden Layer 2
    Dense(64, activation='relu',kernel_initializer='he_normal'),   # Hidden Layer 3
    Dense(10, activation='softmax') # Output Layer
])

# Compile the model
hemodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st=time.time()

# Train the model
history = hemodel.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et=time.time()

# Evaluate the model
hetest_loss, hetest_acc = hemodel.evaluate(x_test, y_test)

duration=et-st

print(f"Duration:{duration:.4f}")
print(f"Test accuracy: {hetest_acc:.4f}")
print(f"Test loss: {hetest_loss:.4f}")


dropmodel = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu'),
    layers.Dropout(0.2),
    Dense(128, activation='relu'),
    layers.Dropout(0.2),
    Dense(64, activation='relu'),
    layers.Dropout(0.2),
    Dense(10, activation='softmax') # Output Layer
])

# Compile the model
dropmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st=time.time()

# Train the model
history = dropmodel.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et=time.time()

# Evaluate the model
droptest_loss, droptest_acc = dropmodel.evaluate(x_test, y_test)

duration=et-st

print(f"Duration:{duration:.4f}")
print(f"Test accuracy: {droptest_acc:.4f}")
print(f"Test loss: {droptest_loss:.4f}")



l2model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)),  # Hidden Layer 1
    Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)),  # Hidden Layer 2
    Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)),   # Hidden Layer 3
    Dense(10, activation='softmax') # Output Layer
])

# Compile the model
l2model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st=time.time()

# Train the model
history = l2model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et=time.time()

# Evaluate the model
l2test_loss, l2test_acc = l2model.evaluate(x_test, y_test)

duration=et-st

print(f"Duration:{duration:.4f}")
print(f"Test accuracy: {l2test_acc:.4f}")
print(f"Test loss: {l2test_loss:.4f}")


result={'Base_model':basetest_acc,
        'Xavier_model':xaviertest_acc,
        'He_Model':hetest_acc,
        'Drop_model':droptest_acc,
        'L2_model':l2test_acc}

best_acc = max(result, key=result.get)
print(f"The model with the maximum performance is: {best_acc} with accuracy: {result[best_acc]:.4f}")
