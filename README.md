# Classifier-Models-Implementation-via-Supervised-Learning-for-Image-Classification 🤖 

This repository focuses on implementing and comparing four different classifier models for image classification using supervised learning. The models are trained on the BloodMNIST dataset, which is a part of the MedMNIST dataset containing health-related image data designed to match the original MNIST dataset shape.


## Project Overview 📊 
- The core objective of this project is to implement and evaluate four distinct classifier architectures on the BloodMNIST dataset. The BloodMNIST dataset is a collection of health-related images designed to mimic the structure of the original MNIST digit dataset.Specifically, the project utilizes the BloodMNIST portion of this dataset.
  
-  The project investigates how traditional machine learning models (SVM and Logistic Regression) perform on multi-class image data compared to neural networks (Feed-Forward Neural Network and Convolutional Neural Network) which are designed to learn complex spatial features.

## Dataset 📋

The BloodMNIST dataset consists of 28x28 RGB images, which are not normalized initially. The dataset is loaded as a dictionary containing images and labels already split into training, validation, and test sets.


- train_images: (11959, 28, 28, 3) uint8 
- train_labels: (11959, 1) uint8 
- val_images: (1712, 28, 28, 3) uint8 
- val_labels: (1712, 1) uint8 
- test_images: (3421, 28, 28, 3) uint8 
- test_labels: (3421, 1) uint8

## Pre-processing 🔄
Necessary pre-processing steps have been implemented as the images are not normalized. 
- For all models, pixel values are normalized to the range [0, 1] by dividing by 255.0.
- For Support Vector Machines and Logistic Regression, the images are also reshaped into a 1D array (flattened).
- Similarly, for the Feed-Forward Neural Network, images are flattened into a 1D array.

## Model Selection 🧠
Four different classifier architectures are used

- 1: Support Vector Machine (SVM)
- 2: Logistic Regression
- 3: Feed-Forward Neural Network (FNN)
- 4: Convolutional Neural Network (CNN)


## Model Implementation and Training 🗺️

### Support Vector Machine (SVM):

- Initialized with probability=True, C=1.0, and an 'rbf' kernel with degree=3.
- Trained iteratively by incrementing max_iter from 1 to 9 in steps of 1.
- Accuracy and log loss are recorded for training and test datasets at each increment.

### Logistic Regression:

- Configured with a regularization parameter C=1.0, fit_intercept=True, and the 'lbfgs' solver.
- Trained by varying max_iter from 100 to 500 in steps of 50.
- Accuracy and log loss are calculated and stored for both training and test sets for each max_iter value.

### Feed-Forward Neural Network (FNN):

- Architecture includes two hidden Dense layers (128 and 64 neurons) with 'tanh' activation, and an output layer (8 neurons) with 'softmax' activation.
- Compiled with 'sgd' optimizer and 'sparse_categorical_crossentropy' loss.
- Trained for 10 epochs, monitoring performance on training and validation sets.

### Convolutional Neural Network (CNN):

- Architecture features two Conv2D layers (32 and 64 filters, 3x3 kernel, 'relu' activation) with a MaxPooling2D layer in between.
- Output is flattened, followed by a Dense layer (64 neurons) with 'relu' activation, and a final output layer (8 neurons) with 'softmax' activation.
- Compiled with 'adam' optimizer and 'sparse_categorical_crossentropy' loss.
- Trained for 10 epochs, tracking performance on training and validation datasets.


## Evaluation and Comparison 💻

- Performance metrics (accuracy and loss) are collected for all models on training, validation, and test sets.
- Visualizations (plots of accuracy and log loss over iterations/epochs) are generated to show training dynamics and model performance which are shown in report file.
  

## Result 🎯

| Model                          | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss | Test Accuracy | Test Loss |
|---------------------------------|:-------------:|:----------:|:------------------:|:--------------:|:------------:|:---------:|
| Support Vector Machine (SVM)    |    ~0.47      |   ~1.36    |        -           |       -        |    ~0.47     |   ~1.36   |
| Logistic Regression             |    ~0.89      |   ~0.3     |        -           |       -        |    ~0.82     |   ~0.5    |
| Feed-Forward Neural Network (FNN)|   0.7535     |   0.6914   |      0.7593        |    0.6729      |   0.7422     |  0.7097   |
| Convolutional Neural Network (CNN)| 0.9120      |   0.2503   |      0.8908        |    0.3147      |   0.8755     |  0.3329   |


## Conclusion ✅
The project demonstrates the implementation and comparative analysis of four distinct classification models on the BloodMNIST dataset. The results indicate that:

### SVM and Logistic Regression: 
- These models, primarily designed for binary classification, were trained on multi-class data. Their performance was evaluated based on accuracy and log loss across varying max_iter values.
- SVM showed moderate accuracy increases from ~36% to ~47%. Logistic Regression showed better performance, with training accuracy reaching ~89% and test accuracy ~82%.

### Feed-Forward Neural Network (FNN): 
The FNN, with 'tanh' activation functions and 'sgd' optimizer, achieved a test accuracy of approximately 74.22%.

### Convolutional Neural Network (CNN): 
The CNN, leveraging its ability to learn spatial hierarchies with 'relu' activation and 'adam' optimizer, demonstrated the highest performance among the tested models, achieving a test accuracy of 87.55%.

## Summary 📖
- The choice of network depth, activation functions, and optimizers played a significant role in the performance of the neural networks. 
- The CNN clearly outperformed the other models, highlighting its suitability for image classification tasks due to its inherent ability to process spatial features effectively. 
- This project provides insights into the capabilities of different classifier architectures for medical image classification.




## Dependencies 🧩
The project relies on the following Python libraries:

- numpy 
- sklearn.svm (for SVC) 
- sklearn.metrics (for accuracy_score, log_loss) 
- sklearn.linear_model (for LogisticRegression) 
- matplotlib.pyplot 
- tensorflow 
- tensorflow.keras.layers 
- tensorflow.keras.models

## Contact 🤝

- Name : Ketan Kulkarni
- LinkedIn: https://www.linkedin.com/in/ketan-b-kulkarni/

  



  
