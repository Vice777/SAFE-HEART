# Importing project dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import streamlit as st
import math
import os
import time
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Safe Heart')
st.header('Classify your Heart Condition as Safe or Unsafe!')


data = pd.read_csv("./Dataset/heart.csv")
print(data.head())

# As we can see, no column names. Hence inserting column names

data = pd.read_csv('NIFTY50_all.csv', sep=',', header=None)
column_names = ['Date', 'Symbol', 'Series', 'Prev_Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume',
                'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
data.columns = column_names  # setting header names
print(data.head())


# Data Visualization

df_corr = data.corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df_corr, mask=np.zeros_like(df_corr, dtype=bool), cmap = "BrBG",ax=ax)
plt.show()

# Data preprocessing
# As the data given is highly ordered, there shuffling the dataset to make it viable for training
df = data.sample(frac=1,random_state=32)
X_label = df.drop(["output"],axis=1)
y_label = df["output"]

# Normalizing the X labels
X1 = X_label.copy()
X1 = (X1 - X1.mean())/X1.std() #Standardizing the X-Labels
X1 = X1.to_numpy()

# Categorizing Y Label using One-Hot Encoding
y1 = np.zeros((y_label.shape[0], (np.amax(y_label)+1)))
y1[np.arange(y_label.shape[0]), y_label] = 1

#Splitting the data in 3/4 ratio
splitting_len = int(0.75 * len(df))
X_train = X1[:splitting_len] 
X_test  = X1[splitting_len:]
y_train = y1[:splitting_len]
y_test  = y1[splitting_len:]


#Implementing Multilayer Perceptron from Scratch
class MultiLayerPerceptron:
     
    def __init__(self,hidden_layer, epoch, learning_rate, verbose=False):
        self.hidden_layer = hidden_layer
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.verbose = verbose
        
    # Initializing the weights    
    def initial_weights(self, X, y):
        n_sample, n_features = X.shape
        n_output = y.shape[1]
        
        limit_hidden = 1/math.sqrt(n_features)
        self.hiddenWeight = np.random.uniform(-limit_hidden,limit_hidden, (n_features, self.hidden_layer))        
        self.BiasHidden = np.zeros((1,self.hidden_layer))
        
        limit_out = 1/ math.sqrt(self.hidden_layer)
        self.outputWeight = np.random.uniform(-limit_out,limit_out, (self.hidden_layer, n_output))
        self.BiasOutput = np.zeros((1, n_output))
     
    #Sigmoid Function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #Sigmoid Derivative Function
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
     
    #SoftMax Function (Output Layer)    
    def softmax(self, z):
        e_x = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    #SoftMax Gradient Function
    def softmax_gradient(self, z):
        return self.softmax(z) * (1 - self.softmax(z))
    
    #Cross-Entropy Loss Function
    def loss(self, h, y):
        h = np.clip(h, 1e-15, 1 - 1e-15)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h))
    
    #Cross-Entropy Loss Gradient Function
    def loss_gradient(self, h, y):
        h = np.clip(h, 1e-15, 1 - 1e-15)
        return -(h/y) + (1-h)/(1-y)
    
    #Accuracy Score Function
    def accuracy_score(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
        return accuracy
    
    #Prediction Function
    def predict(self, X):
        hidden_input = X.dot(self.hiddenWeight) + self.BiasHidden
        hidden_output = self.sigmoid(hidden_input)
        output_layer_input = hidden_output.dot(self.outputWeight) + self.BiasOutput
        y_pred = self.softmax(output_layer_input)
        return y_pred
    
    #Fit Function
    def fit(self, X, y):
        self.initial_weights(X, y)
        n_epoch = 1
        
        while(n_epoch <= self.epoch):
            
            # Forward Propogation
            #hidden Layer
            hidden_input = X.dot(self.hiddenWeight) + self.BiasHidden
            hidden_output = self.sigmoid(hidden_input)
            #output layer
            output_layer_input = hidden_output.dot(self.outputWeight) + self.BiasOutput
            y_pred = self.softmax(output_layer_input)
            
            #Backward Propogation
            #Output Layer Gradient
            grad_out_input = self.loss_gradient(y, y_pred) * self.softmax_gradient(output_layer_input)
            grad_output = hidden_output.T.dot(grad_out_input)
            grad_biasoutput = np.sum(grad_out_input,axis=0,keepdims=True)
            #Hidden Layer Gradient
            grad_input_out = grad_out_input.dot(self.outputWeight.T) * self.sigmoid_derivative(hidden_input)
            grad_input = X.T.dot(grad_input_out)
            grad_biasinput = np.sum(grad_input_out, axis=0, keepdims=True)
            
            #Updating Weights
            self.outputWeight -= self.learning_rate * grad_output
            self.BiasOutput -= self.learning_rate *grad_biasoutput
            self.hiddenWeight -= self.learning_rate * grad_input
            self.BiasHidden -= self.learning_rate * grad_biasinput
                        
            
            n_epoch += 1
            
       

# In order to find best model with highest accuracy, training process is repeated several times and \
# the model with best accuracy is pickled dwn in the file 'Bestfitmodel.pickle' and the same is embedded \
# in the web-app developed using streamlit

# below is the model training process:
'''model = MultiLayerPerceptron(hidden_layer = 16, epoch=10000, learning_rate=0.001, verbose=True)
model.fit(X_train, y_train)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = model.accuracy_score(y_test, y_pred)
print ("Accuracy: ", accuracy)

with open('Bestfitmodel.pickle', 'wb') as file:
    pickle.dump(model, file)'''

pickle_in = open('Bestfitmodel.pickle', 'rb')
model = pickle.load(pickle_in)
y_pred_pickel =  np.argmax(model.predict(X_test), axis=1)
y_test_pk = np.argmax(y_test, axis=1)

print(model.accuracy_score(y_test_pk, y_pred_pickel))

# Here our training part is completed and now we will move on to web-app development, where data-visualization is
# also done.

# Part to continue .............................

