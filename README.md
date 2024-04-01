# Facial-Emotion-Detection-Using-CNN-Model

This repository contains code for a deep learning model that performs emotion recognition using Convolutional Neural Networks (CNNs). The model is trained on a dataset consisting of facial images labeled with different emotions.

Overview
Emotion recognition is a crucial aspect of human-computer interaction and affective computing. In this project, we leverage CNNs to classify facial images into six emotion categories: Anger, Disgust, Fear, Happiness, Sadness, and Surprise.

Dataset
The dataset used in this project is the eINTERFACE Image Dataset, which contains facial images labeled with six different emotions. The dataset is divided into training and testing sets, with images resized to 128x128 pixels for model training. https://www.kaggle.com/datasets/ameyamote030/einterface-image-dataset?rvi=1

Model Architecture
The CNN model architecture comprises two convolutional layers followed by max-pooling layers, flattening, and dense layers. The final output layer utilizes the softmax activation function to predict the probability distribution over the six emotion classes.

Training and Evaluation
The model is trained using the training set and evaluated on the testing set. We employ the Adam optimizer and categorical cross-entropy loss function during training. Performance metrics such as loss and accuracy are monitored using validation data.

Usage
To utilize the trained model for emotion recognition on new images, load the model and preprocess the input image before making predictions. The model predicts the most probable emotion label for the given input image.

Results
The trained model achieves an accuracy of approximately 84% on the testing set. Sample predictions and corresponding images are visualized to showcase the model's performance.

Dependencies
Python 3,
TensorFlow,
Keras,
NumPy
OpenCV,
Matplotlib,
Pandas.
How to Use
Clone this repository to your local machine.
Install the dependencies listed in the requirements.txt file.
Utilize the provided Jupyter Notebook or Python scripts to train the model or make predictions on new images.
