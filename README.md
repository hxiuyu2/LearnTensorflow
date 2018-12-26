# LearnTensorflow
Learning Tensorflow with [Ucademy (Complete Guide to TensorFlow for Deep Learning with Python)](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/)

## Classification:
Test and compare LinearClassification and DNNClassification with census_data.csv.
  
  Linear Classification:

>              precision    recall  f1-score   support
> 
>           0       0.88      0.88      0.88      7436
>           1       0.61      0.61      0.61      2333
>     avg / total   0.81      0.81      0.81      9769

NN Classification:

>              precision    recall  f1-score   support
> 
>           0       0.85      0.95      0.89      7436
>           1       0.73      0.46      0.56      2333
>     avg / total   0.82      0.83      0.81      9769

## CNN:
Two convolutional layers with filter size 4\*4 +
  
  One normal neuro network (fully connected) +
  
  Dropout to deal with overfitting
  
  The accyracy increases fast at first and slow down after 20 steps and finally flucutates around 55%
![](/acc.png)
  
  There are 10 classes in the dataset, 55% is OK for 500 steps.
