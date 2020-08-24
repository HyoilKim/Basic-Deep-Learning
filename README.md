# Fashion Mnist Classifier

Skills
- Python(3.6.9) 
- tensorflow(2.2.0) / pytorch(1.5.1+cu101)
- colab

### Struggle to increse performance(test accuracy)
1. epochs

| epochs | accuracy |
|:--------:|:--------:|
| 5 | 85% |
| 10 | 86% |
| 30 | 87% |
| 50 | 88% |
| 70 | 88% |

2. kernel initializer(he_uniform)

| epochs | origin(random) | he |
|:--------:|:--------:|:--------:|
| 30 | 87.6% | 87.7% |

3. model
- Add layer

| epochs | origin | add layer |
|:--------:|:--------:|:--------:|
| 30 | 87.6% | 87.9% |

- Change model(Dense->Conv2D)

| epochs | Dense | Conv2D | dropout | 
|:--------:|:--------:|:--------:|:--------:|
| 10 | 86% | 90% | |
| 20 | 87% | 90% | 91% |
| 30 | 87% | 88%(overfitting) | 91% |

**=> 성능에 가장 크게 영향을 미치는 것은 모델이다**

# Cat&Dog Classifier

Skills
- Python(3.6.9) 
- tensorflow(2.2.0) / pytorch(1.5.1+cu101)
- colab

### Struggle to increse performance(test accuracy)
1. epochs

| epochs | accuracy |
|:--------:|:--------:|
| 10 | 90% |
| 20 | 90% |
| 30 | 88% |

2. dropout

| epochs | origin | dropout |
|:--------:|:--------:|:--------:|
| 30 | 88% | 91% |

3. model

| epochs | origin | more layer |
|:--------:|:--------:|:--------:|
| 20 | 90% | 91% |
| 30 | 88% | 90% |

| epochs | origin | ResNet |
|:--------:|:--------:|:--------:|
| 10 | 76% | 98% |

# Whether Prediction

Skills
- Python(3.6.9) 
- tensorflow(2.2.0) / pytorch(1.5.1+cu101)
- colab

Data
- 기상자료개방포털(2009~2018)

### Struggle to increse performance(test loss)
1. model </br>
BaseLine: 어제 같은 시간과 동일하다고 예측

| epochs | BaseLine | LSTM | LSTM(bidirectional) | GRU | GRU(bidirectional) | 
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 30 | 0.1058 | 0.0861 | 0.0939 | 0.0756 | 0.0741 |

**=> GRU 모델에서 양방향 모델의 성능이 좋은 것으로 보아 특정 날씨는 전 날과 다음 날의 영향을 동시에 받음**

2. epochs </br>
LSTM(bidirectional)이 LSTM보다 낮은 성능이 의아함</br>
train loss가 급격히 적은 것으로 보아 overfitting 예상</br>

| epochs | LSTM | LSTM(bidirectional) |
|:--------:|:--------:|:--------:|
| 10 |  | 0.8253 |
| 15 |  | 0.0790 |
| 20 |  | 0.0876 |
| 30 | 0.0861 | 0.0939 |

**=> 신기하게도 양방향 LSTM은 단방향 LSTM에 비해 training 적은 epochs으로 빠르게 학습됨**
