# for colab users
'''
from google.colab import auth
auth.authenticate_user()

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os
from pathlib import Path

# folder 변수에 구글드라이브에 프로젝트를 저장한 디렉토리를 입력하세요!
# My Drive 밑에 저장했다면 그대로 두시면 됩니다.
folder = "colab_notebook"
project_dir = "02_cnn_tf"

base_path = Path("/content/drive/My Drive/")
project_path = base_path / folder / project_dir
os.chdir(project_path)
for x in list(project_path.glob("*")):
    if x.is_dir():
        dir_name = str(x.relative_to(project_path))
        os.rename(dir_name, dir_name.split(" ", 1)[0])
print(f"현재 디렉토리 위치: {os.getcwd()}")

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
print_steps = 1
val_epoch = 1

batch_size = 20
max_epochs = 20
learning_rate = 1e-4
IMG_SIZE = 150

data_dir = './data/my_cat_dog'  # 압축 해제된 데이터셋의 디렉토리 경로
PATH = data_dir

# 3. Dataset load 및 tf.data.Dataset 구축
import zipfile
from pathlib import Path

current_path = Path().absolute()
data_path = current_path / "data"
print("현재 디렉토리 위치: {}".format(current_path))
if (data_path / "my_cat_dog").exists():
    print("이미 'data/my_cat_dog' 폴더에 압축이 풀려있습니다. 확인해보세요!")
else:
    with zipfile.ZipFile(str(data_path / "my_cat_dog.zip"), "r") as zip_ref:
        zip_ref.extractall(str(data_path / "my_cat_dog"))
    print("Done!")

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

# 4. tf.data.Dataset을 이용하여 input pipeline 구성
def load(image_file, label):
    # 해당경로의 파일을 읽어서 float 타입으로 변환합니다.
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)

    return image, label

def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width])
    return input_image

def random_rotation(input_image):
    angles = np.random.randint(0, 3)
    rotated_image = tf.image.rot90(input_image, angles)
    return rotated_image

def random_crop(input_image):
    crop_size = [150, 150, 3]
    cropped_image = tf.image.random_crop(input_image, crop_size)
    return cropped_image

def normalize(input_image):
    # [0, 255] -> [-1, 1]
    input_image = input_image/127.5 - 1
    return input_image

# augmentation    
def random_jitter(input_image):
    # resizing to 176 x 176 x 3
    input_image = resize(input_image, 176, 176)
    # randomly cropping to 150 x 150 x 3
    input_image = random_crop(input_image)
    # randomly rotation
    input_image = random_rotation(input_image) 
    # randomly mirroring
    input_image = tf.image.random_flip_left_right(input_image)
    ## 코드 종료 ##
    return input_image

# 확인
image, label = load(os.path.join(PATH, 'train/cat/cat.100.jpg'), 0)
plt.figure()
plt.subplot(121)
plt.title("Central Cropped Image")
plt.imshow(central_crop(image)/255.0)
plt.subplot(122)
plt.title("Original Image")
plt.imshow(image/255.0)
plt.show()

# load pipeline
def load_image_val_and_test(image_file, label):
    input_image, label = load(image_file, label)
    input_image = central_crop(input_image)
    input_image = normalize(input_image)

    return input_image, label

def add_label(image_file, label):
    return image_file, label

# train folder에 있는 폴더 이름을 list로 나타냅니다.
# 즉 학습에 사용할 category의 이름을 list로 나타내는 것입니다.
folder_list = [f for f in os.listdir(os.path.join(PATH, 'train')) if not f.startswith('.')]

train_dataset = tf.data.Dataset.list_files(                            # 1번
    os.path.join(PATH, 'train', folder_list[0], '*.jpg'))
train_dataset = train_dataset.map(lambda x: add_label(x, 0))          # 2번
for label, category_name in enumerate(folder_list[1:], 1):            # 3번
    temp_dataset = tf.data.Dataset.list_files(                         # 4번
        os.path.join(PATH, 'train', category_name, '*.jpg'))
    temp_dataset = temp_dataset.map(lambda x: add_label(x, label))    # 5번
    train_dataset = train_dataset.concatenate(temp_dataset)            # 6번

N = BUFFER_SIZE = len(list(train_dataset)) # number of samples in train_dataset
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=16)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()

folder_list = [f for f in os.listdir(os.path.join(PATH, 'val')) if not f.startswith('.')]
val_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'val', folder_list[0], '*.jpg'))
val_dataset = val_dataset.map(lambda x: add_label(x, 0))
for label, category_name in enumerate(folder_list[1:], 1):
    temp_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'val', category_name, '*.jpg'))
    temp_dataset = temp_dataset.map(lambda x: add_label(x, label))
    val_dataset = val_dataset.concatenate(temp_dataset)

val_dataset = val_dataset.map(load_image_val_and_test)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()

folder_list = [f for f in os.listdir(os.path.join(PATH, 'test')) if not f.startswith('.')]
test_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'test', folder_list[0], '*.jpg'))
test_dataset = test_dataset.map(lambda x: add_label(x, 0))
for label, category_name in enumerate(folder_list[1:], 1):
    temp_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'test', category_name, '*.jpg'))
    temp_dataset = temp_dataset.map(lambda x: add_label(x, label))
    test_dataset = test_dataset.concatenate(temp_dataset)

test_dataset = test_dataset.map(load_image_val_and_test)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.repeat()

# 5. 모델(네트워크) 만들기
model = tf.keras.Sequential()

# class 모델(미완)
'''
class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(Conv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU
        self.pool = tf.keras.layers.MaxPool2D()

    def call(self, inputs, training=True):
        x = self.conv(inputs)    # self.conv forward
        x = self.bn(x)    # self.bn   forward
        x = self.relu(x)    # self.relu forward
        x = self.pool(x)    # self.pool forward
        return x

model.add(Conv(filters=32, kernel_size=3))
model.add(Conv(filters=64, kernel_size=3))
model.add(Conv(filters=128, kernel_size=3))
model.add(Conv(filters=128, kernel_size=3))
'''

# sequential 모델
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape(IMG_SIZE, IMG_SIZE, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())

# dense layer
model.add(layers.Flatten())  # flatten
model.add(layers.Dense(512, activation='relu'))  # relu
model.add(layers.Dense(2, activation='softmax'))  # softmax

# 모델 요약 보기
model(images[:1])
model.summary()

# 모델 저장하기
checkpoint_path = "./train/exp_cnn/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# 6. Loss function 및 Optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. Training
train_len = len(glob(os.path.join(PATH, 'train', folder_list[0], '*.jpg'))) * 2
val_len = len(glob(os.path.join(PATH, 'val', folder_list[0], '*.jpg'))) * 2
test_len = len(glob(os.path.join(PATH, 'test', folder_list[0], '*.jpg'))) * 2

model.fit(train_dataset, steps_per_epoch = train_len/batch_size,
          validation_data=val_dataset, 
          validation_steps= val_len/batch_size,
         epochs= max_epochs,
         callbacks= [cp_callback])

# 8. 저장 모델 불러오기 및 Evaluation
# 5 + 6 code
for images, labels in train_dataset.take(1):
    outputs = model(images, training=False)

latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)

model.evaluate(test_dataset, steps=test_len/batch_size)

# 9. 전이 학습
# if include=False -> only conv 
'''
conv_block1
conv_block2
conv_block3
conv_block4
conv_block5
dense1 (4096)
dense2 (4096)
dense3 (1000)
'''
conv_base = tf.keras.applications.VGG16(weights='imagenet',
                                        include_top=False,
                                        input_shape=(IMG_SIZE, IMG_SIZE, 3))

model = tf.keras.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# training variable 확인
# for var in model.trainable_variables:
#     print(var.name)

# fine tuning with only new_dense layer
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# loss & optimizer
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ["accuracy"])

# checkpoint
checkpoint_path = "./train/exp_pre_trained/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

# 처음 모델 20 epoch vs 전이 학습 5 epoch
model.fit(train_dataset, steps_per_epoch = train_len/batch_size,
          validation_data=val_dataset, 
          validation_steps=val_len/batch_size,
          epochs= 5,
          callbacks= [cp_callback])