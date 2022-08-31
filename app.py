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

feature_std = []
feature_mean = []
for i in range(X_label.shape[1]):
    feature_mean.append(X1[:, i].mean())
    feature_std.append(X1[:, i].std())

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
# '''model = MultiLayerPerceptron(hidden_layer = 16, epoch=10000, learning_rate=0.001, verbose=True)
# model.fit(X_train, y_train)
#
# y_pred = np.argmax(model.predict(X_test), axis=1)
# y_test = np.argmax(y_test, axis=1)
# 
# accuracy = model.accuracy_score(y_test, y_pred)
# print ("Accuracy: ", accuracy)
#
# with open('Bestfitmodel.pickle', 'wb') as file:
#     pickle.dump(model, file)'''

pickle_in = open('Bestfitmodel.pickle', 'rb')
model = pickle.load(pickle_in)
y_pred_pickel =  np.argmax(model.predict(X_test), axis=1)
y_test_pk = np.argmax(y_test, axis=1)

print(model.accuracy_score(y_test_pk, y_pred_pickel))

# Here our training part is completed and now we will move on to web-app development, where data-visualization is
# also done.

# Part to continue .............................

nav_choice = st.sidebar.radio('Navigation', ('Home', 'Data Visualization', 'Classification'), index=0)

if nav_choice == 'Home':
    st.image("./Assets/Cross-section-human-heart.jpg", width=800)

    st.success('Here, for the purpose of prediction, this app uses Linear Regression algorithm − '
               'one of the classic supervised machine learning algorithms.')

    st.warning('For the purpose of prediction, only features given in the table '
                'below are used. Detailed description about the features is provided within the table.')

    #Readme
    st.markdown('<table>'
                '<tr>'
                    '<th align=\centre\><b>Features</b></th>'
                    '<th align=\centre\><b>Description</b></th>'
                '</tr>'
                '<tr>'
                    '<td>Age</td>' 
                    '<td>Age of the patient</td>'
                '</tr>'
                '<tr>'
                    '<td>Sex </td>'
                    '<td>Sex of the patient</td>'
                '</tr>'
                '<tr>'
                    '<td>exang</td>'
                    '<td>Exercise induced angina'                        
                    '<li>1 = yes</li>'
                        '<li>0 = no</li>'
                    '</td>'
                '</tr>'
                '<tr>'
                    '<td>ca</td>'
                    '<td>Number of major vessels (0-3)</td>'
                '</tr>'
                '<tr>'
                    '<td>cp </td>'
                    '<td>Chest Pain type chest pain type:'
                        '<ul>'
                            '<li>Value 1: Typical angina</li>'
                            '<li>Value 2: Atypical angina</li>'
                            '<li>Value 3: Non-anginal pain</li>'
                            '<li>Value 4: Asymptomatic</li>'
                        '</ul>'
                    '</td>'
                '</tr>'
                '<tr>'
                    '<td>trtbps </td>'
                    '<td>Resting Blood Pressure (in mm Hg)</td>'
                '</tr>'
                '<tr>'
                    '<td>chol </td>'
                    '<td>Cholestoral in mg/dl fetched via BMI Sensor</td>'
                '</tr>'
                '<tr>'
                    '<td>fbs </td>'
                    '<td>'
                        '(Fasting blood sugar > 120 mg/dl) '
                        '<li>1 = true</li>'
                        '<li>0 = false </li>'
                    '</td>'
                '</tr>'
                '<tr>'
                    '<td>Trades</td>'
                    '<td>'
                        '<ol>'
                            '<li>Value 0: Normal</li>'
                            '<li>Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)</li>'
                            '<li>Value 2: Showing probable or definite left ventricular hypertrophy by Estes criteria</li>'
                        '</ul>'
                    '</td>'
                '</tr>'
                '<tr>'
                    '<td>thalach </td>'
                    '<td>Maximum Heart Rate achieved</td>'
                '</tr>'
                '<tr>'
                    '<td>target </td>'
                    '<td>Percentage of deliverable volume'
                        '<li> 0= less chance of heart attack</li>'
                        '<li> 1= more chance of heart attack</li>'
                    '</td>'
                '</tr>'
            '</table><br>' , unsafe_allow_html=True)

    st.markdown('<b><font color=\'yellow\'>Given below is the link of the data used for the purpose of training of the model'
                '</font></b>'
                '<br><a href=\'https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset\' '
                'target=\'_blank\'>Heart Attack Analysis & Prediction Dataset</a>'
                , unsafe_allow_html=True)


elif nav_choice == 'Data Analysis':
    exit()
    
elif nav_choice == 'Classification':
    
    st.markdown('Kindly hit ENTER after each entry')
        
    age = st.number_input('Age of the patient',step=1)

    Sex  = st.selectbox('Sex of patient',
                        ('Male','Female'))

    cp = st.selectbox('Chest Pain type chest pain type',
                        ('Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic') )

    trtbps  = st.number_input('Resting blood pressure (in mm Hg)',step=1)

    chol  = st.number_input('Cholestoral in mg/dl ',step=1)

    fbs  = st.selectbox('Is Fasting blood sugar > 120 mg/dl', options=("Yes","No"))

    rest_ecg = st.number_input(label = 'Resting Electrocardiographic Results:' ,step=1,min_value=0, max_value=2)

    thalach  = st.number_input('Maximum heart rate achieved',step=1)

    exang = st.selectbox('Does exercise induced Angina',
                        ('Yes','No'))

    oldpeak = st.number_input('Old Peak')

    slp = st.selectbox('Slope of the peak exercise ST segment',
                        (
                            'Downsloping',
                            'Flat',
                            'Upsloping')
                        )

    ca = st.number_input('Number of major vessels',step=1,min_value=1,max_value=3)
    thal = st.selectbox('A blood disorder called thalassemia',
                            (
                            'Fixed defect ',
                            'Normal blood flow',
                            'Reversible defect'                           
                            )
                        )

    submit = st.button('Predict')

    def featureset_scale(arr):
        for i in range(X_test.shape[1]):
            arr[i] = (arr[i] - feature_mean[i]) / feature_std[i]
        return arr


    if submit:
        featureset = [age,Sex,cp,trtbps,chol,fbs,rest_ecg,thalach,exang,oldpeak,slp,ca,thal]
        featureset = np.array(featureset)

        for i in range(len(featureset)):
            if i==0 or i ==3 or i==4  or i==6 or i==11:
                featureset[i] = featureset[i].astype(int)

            elif i==1:
                if featureset[i] == "Male": featureset[i] = int(0)
                else: featureset[i] = int(1)

            elif i==2:
                if featureset[i] =='Typical Angina': featureset[i] = int(0)
                elif featureset[i] =='Atypical Angina': featureset[i] = int(1)
                elif featureset[i] =='Non-Anginal Pain': featureset[i] = int(2)
                else: featureset[i] = int(3)
            elif i == 5:
                if featureset[i] == "Yes": featureset[i] = int(1)
                else: featureset[i] = int(0)
            
            elif i==7:
                featureset[i] = featureset[i].astype(float)


            elif i==8:
                if featureset[i] == "Yes": featureset[i] = int(0)
                else: featureset[i] = int(1)
        
            elif i==9:  
                featureset[i] = featureset[i].astype(float)

            elif i==10:
                if featureset[i] =='Downsloping': featureset[i] = int(0)
                elif featureset[i] =='Flat': featureset[i] = int(1)
                else: featureset[i] = int(2)

            else:
                if featureset[i] =='Fixed defect ': 
                    featureset[i] = int(0)
                    featureset[i] = featureset[i].astype(int)
                elif featureset[i] =='Normal blood flow': featureset[i] = int(1)
                else: featureset[i] = int(2)
        
       
        featureset = featureset.astype(float) 
        featureset = featureset_scale(featureset)

        prediction = np.argmax(model.predict(featureset),axis=1)
        
        load = st.progress(0)
        for i in range(100):
            time.sleep(0.00001)
            load.progress(i + 1)
        st.success(f'Heart Rate classification : {prediction}')