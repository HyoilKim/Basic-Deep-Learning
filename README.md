# Fashion Mnist Classifier

Skills
- Python(3.6.9) 
- tensorflow(2.2.0)
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

# Cat&Dog Classifier

Skills
- Python(3.6.9) 
- tensorflow(2.2.0)
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

3. model(add layer)

| epochs | origin | add |
|:--------:|:--------:|:--------:|
| 20 | 90% | 91% |
| 30 | 88% | 90% |

# Whether Prediction

Skills
- Python(3.6.9) 
- tensorflow(2.0.0)
- colab

Data
- 기상자료개방포털(2009~2018)

### Struggle to increse performance(test loss)
1. model

| LSTM | LSTM(bidirectional) | GRU | GRU(bidirectional |
|:--------:|:--------:|:--------:|:--------:|
| 11.91 | 13.77 | 11.77 | 11.66 |
