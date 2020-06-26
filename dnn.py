# for colab users
'''
from google.colab import auth
auth.authenticate_user()

from google.colab import drive
drive.mount('/content/drive', force_remount=False)
'''
# 1. 패키지 로드
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import check_util.checker as checker 
from IPython.display import clear_output

import os
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow.keras import layers

# 2. 하이퍼 파라미터 세팅
batch_size = 128
max_epochs = 5
learning_rate = 0.001
num_classes = 10

# 3. Dataset load 및 tf.data.Dataset 구축
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_data = train_data / 255.
train_data = train_data.reshape([-1, 28 * 28])
train_data = train_data.astype(np.float32)
train_labels = train_labels.astype(np.int32)

test_data = test_data / 255.
test_data = test_data.reshape([-1, 28 * 28])
test_data = test_data.astype(np.float32)
test_labels = test_labels.astype(np.int32)

# for train
N = len(train_data)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=100000)
train_dataset = train_dataset.batch(batch_size)
print(train_dataset)

# for test
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
test_dataset = test_dataset.batch(batch_size)
print(test_dataset)

# 4. 데이터 샘플 시각화
'''
labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}
columns = 5
rows = 5
fig = plt.figure(figsize=(8, 8))

for i in range(1, columns*rows+1):
    data_idx = np.random.randint(len(train_data))
    img = train_data[data_idx].reshape([28, 28])
    label = labels_map[train_labels[data_idx]]

    fig.add_subplot(rows, columns, i)
    plt.title(label)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.show()
'''

# 5. 모델(네트워크) 만들기
from keras.layers import Dense, Activation, BatchNormalization, ReLU
from keras.models import Sequential

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(num_classes, input_dim=512, activation='softmax'))

# 데이터의 일부를 넣어 model 체크
'''
for images, labels in train_dataset.take(1):
    print("predictions: ", model(images[0:3]))
model.summary()
'''

# 6. Loss function 및 Optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. Training
model.fit(train_data, train_labels, batch_size=batch_size, epochs=max_epochs)

# 8. Evaluate on test dataset
loss,accuracy = model.evaluate(test_data, test_labels)
print('test loss is {}'.format(loss))
print('test accuracy is {}'.format(accuracy))

# output 확인하기
model.predict(tf.reshape(images[0], (1,-1))) #  input type: (batch_size, input_shape)
model(tf.reshape(images[0], (1,-1)), training = False)