# Fashion Mnist Classifier

Skills
- Python(3.6.9) 
- tensorflow(2.2.0)
- colab

Struggle to increse performance
1. epochs

| epochs | test accuracy |
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

- Change model(dnn->cnn)

| epochs | origin | cnn | dropout | 
|:--------:|:--------:|:--------:|:--------:|
| 10 | 86% | 90% | |
| 20 | 87% | 90% | 91% |
| 30 | 87% | 88%(overfitting) | 91% |

# Cat&Dog Classifier

Skills
- Python(3.6.9) 
- tensorflow(2.2.0)
- colab

# Whether Prediction

Skills
- Python(3.6.9) 
- tensorflow(2.0.0)
- colab

Data
- 기상자료개방포털(2009~2018)
