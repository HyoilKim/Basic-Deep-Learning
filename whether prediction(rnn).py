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
batch_size = 128
max_epochs = 30
learning_rate = 3e-5
hidden_sizes = [100, 100] # hidden_sizes must be a list

# 3. 데이터 전처리 함수 정의
def preprocess(all_files):
    data_0 = []  # 기온
    data_1 = []  # 강수량
    data_2 = []  # 풍속
    data_3 = []  # 습도
    data_4 = []  # 증기압
    data_5 = []  # 이슬점 온도
    data_6 = []  # 현지 기압
    data_7 = []  # 해면 기압
    data_8 = []  # 지면 온도
    for f in all_files:
        with open(f, encoding='euc-kr') as c:
            csv_reader = csv.reader(c, delimiter=',')
            header = True
            for col in csv_reader:
                if header:
                    header = False
                    continue
                data_0.append(
                    float(col[2])) if col[2] != '' else data_0.append(0.0)
                data_1.append(
                    float(col[3])) if col[3] != '' else data_1.append(0.0)
                data_2.append(
                    float(col[4])) if col[4] != '' else data_2.append(0.0)
                data_3.append(
                    float(col[6])) if col[6] != '' else data_3.append(0.0)
                data_4.append(
                    float(col[7])) if col[7] != '' else data_4.append(0.0)
                data_5.append(
                    float(col[8])) if col[8] != '' else data_5.append(0.0)
                data_6.append(
                    float(col[9])) if col[9] != '' else data_6.append(0.0)
                data_7.append(
                    float(col[10])) if col[10] != '' else data_7.append(0.0)
                data_8.append(
                    float(col[22])) if col[22] != '' else data_8.append(0.0)

    data = np.zeros((len(data_0), 9))
    for i, d in enumerate(data):
        data[i, 0] = data_0[i]
        data[i, 1] = data_1[i]
        data[i, 2] = data_2[i]
        data[i, 3] = data_3[i]
        data[i, 4] = data_4[i]
        data[i, 5] = data_5[i]
        data[i, 6] = data_6[i]
        data[i, 7] = data_7[i]
        data[i, 8] = data_8[i]

    return data.astype(np.float32)

data_dir = './data/climate_seoul'

# glob 모듈의 glob 함수는 사용자가 제시한 조건(*)에 맞는 파일명을 리스트 형식으로 반환한다. 
train_data = preprocess(sorted(glob.glob(os.path.join(data_dir, 'train', '*'))))
val_data = preprocess(sorted(glob.glob(os.path.join(data_dir, 'val', '*'))))
test_data = preprocess(sorted(glob.glob(os.path.join(data_dir, 'test', '*'))))

# shape of data
print("shape of train data: {}".format(train_data.shape))
print("shape of val data: {}".format(val_data.shape))
print("shape of test data: {}".format(test_data.shape))

# 4. 데이터 샘플 시각화
start = 0
dt = 240 # 240시간
plt.plot(train_data[start:start+dt, 0])
plt.ylabel("temperature")
plt.xlabel("time-delta")
plt.show()

# 5. Dataset 만들기
# seq_length: 과거 480시간의 데이터를 기반으로 
# target_delay: 24시간 후의 기온을 예측
# strides: training data 조절
def make_dataset(data, seq_length=480, target_delay=24, strides=5,
                 mode='train', train_mean=None, train_std=None):

    assert mode in ['train', 'val', 'test']
    if mode is not 'train':
        if train_mean is None or train_std is None:
            print('Current mode is {}'.format(mode))
            print('This mode needs mean and std of train data')
            assert False

    # 정규화
    if mode is 'train':
        mean = np.mean(data, axis=0)    
        std = np.std(data, axis=0)    
    else: 
        mean = train_mean   
        std = train_std    
    data = data-mean/std 
    
    # 입력, 타겟 데이터 생성
    sequence = []
    target = []
    for index in range(len(data) - seq_length - target_delay):
        if index % strides == 0:
            sequence.append(data[index:index+seq_length])    
            target.append(data[index+seq_length+target_delay][0])     

    if mode is 'train':
        return np.array(sequence), np.array(target), mean, std
    else:
        return np.array(sequence), np.array(target)

train_sequences, train_labels, train_mean, train_std = make_dataset(train_data, mode='train')
val_sequences, val_labels = make_dataset(val_data, mode='val', train_mean=train_mean, train_std=train_std)
test_sequences, test_labels = make_dataset(test_data, mode='test', train_mean=train_mean, train_std=train_std)

# input pipeline
N = BUFFER_SIZE = len(train_sequences) # number of samples in train_dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE) # mini-batch randomly
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
test_dataset = test_dataset.batch(batch_size)

# 미완 : prediction(128,480,9)을 (,128) 로 만들기, slicing 실패
# 6. 베이스라인 성능 측정(오늘 기온 == 내일 기온) 
def eval_baseline(dataset, loss_fn):
    mean_loss = tf.keras.metrics.Mean()
    for sequences, targets in dataset:
        predictions = (np.sum(sequences, axis=0)/len(sequences))[-1][0] # 마지막 날의 첫 번째 인자(기온)  
        loss = loss_fn(predictions, targets)           
        mean_loss(loss)
        
    print('Baseline Average Loss: {:.4f}'.format(mean_loss.result()))
    return mean_loss.result()

# 7. 네트워크 설계
# sequential
model = tf.keras.Sequential()
num_layers = len(hidden_sizes)

for i in range(num_layers - 1):
    model.add(tf.keras.layers.LSTM(hidden_sizes[0], activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True))
    model.add(tf.keras.layers.LSTM(hidden_sizes[-1], activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
    model.add(tf.keras.layers.Dense(1))

# class
'''
class my_model(tf.keras.Model):
    def __init__(self, hidden_sizes):
        super(my_model, self).__init__(name = '')
        self.lstm_a = layers.LSTM(hidden_sizes[0], return_sequences = True)
        self.lstm_b = layers.LSTM(hidden_sizes[-1], return_sequences = False)
        self.dense = layers.Dense(1)
        
    def call(self, input_tensor, training = False):
        x = self.lstm_a(input_tensor)
        x = self.lstm_b(x)
        x = self.dense(x)
        
        return x
'''
model.summary()

# Loss function, Optimizer
model.compile(loss='mse', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# 모델 저장
checkpoint_path = "./train/exp_rnn/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# 8. Training
model.fit(train_dataset, steps_per_epoch = len(train_data)/batch_size, 
          validation_data = val_dataset, 
          validation_steps= len(val_data)/batch_size,
          epochs= max_epochs,
          callbacks= [cp_callback])

# 9. 저장된 모델 불러오기 및 test
# 7 + 8
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
_, test_loss = model.evaluate(test_dataset, steps=len(test_sequences)/batch_size)