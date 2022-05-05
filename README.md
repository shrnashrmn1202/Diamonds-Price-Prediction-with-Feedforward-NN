# Diamonds-Price-Prediction-with-Feedforward-NN
Using Feedforward Neural Network model created, the price of diamond will be predicted based on features such as cut, color, clarity, price, and other attributes.

## IDE and Framework
This project is created using Sypder as the main IDE. The main frameworks used in this project are Pandas, Numpy, Scikit-learn and TensorFlow Keras.

## Methodology

### 1. Data Preprocessing
The datasets are loaded and cleaned first by removing unwanted features. Categorical features are encoded ordinally. 

### 2. Data Pipeline
Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

### 3. Model Pipeline
A feedforward neural network is constructed that is catered for regression problem. The structure of the model is fairly simple with three types of layers:
- Input layer
- Hidden layer
- Output layer

The model is trained with a batch size of 64 and for 100 epochs. Early stopping is applied in this training. The training stops at epoch 26, with a training MAE of 765 and validation MAE of 414 as shown in the figure below:

![image](https://user-images.githubusercontent.com/100325884/166849535-981ccceb-926f-4c26-b77a-d36bcf74f119.png)


## Results
Upon evaluating the model with test data, the model obtain the following test results:

![image](https://user-images.githubusercontent.com/100325884/166849632-9cf7c136-2d41-4337-b654-b07950cac2a0.png)

The model is also used to made prediction with test data. A graph of prediction vs label is plotted, as shown in the image below.

![image](https://user-images.githubusercontent.com/100325884/166849683-e846502f-f9b5-4bc9-bee7-60af21fb672d.png)

Based on the graph, a clear trendline of y=x can be seen, indicating the predictions are fairly similar as labels. However, several outliers can be seen in the graph.

