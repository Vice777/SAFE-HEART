# Safe-Heart-Classifier-Multi-Layer-Perceptron-from-Scratch

LINK : https://vice777-safe-heart-classifier-multilayer-perceptron--app-idf5su.streamlitapp.com/
___
<a href="url"><img src="https://i.pinimg.com/originals/3d/e3/a6/3de3a6cae7d628ad1ae7b6d03a4cd649.gif" align="right" height="240" width="240" ></a>

## Description:<br>
Classify the chances of having a Heart Attack based on your Heart's Condition.<br>
In this end-to-end Machine Learning project-tutorial, I have created and trained Multi-Layer model from scratch, using NumPy.<br>
Furthermore, the model with the best accuracy is embedded in the web-app developed using streamlit module for the purpose of classification of your Heart's Condition.   <br>
___

<h2>Understanding the Problem Statement</h2>

This project uses the popular <a href="https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset" target="_blank">Heart Attack Analysis & Prediction Dataset</a>  for training the model and making predictions.<br>

For the purpose of prediction and classification, the features given in the table below are used. <br>
Detailed description about the features is provided within the table.<br><br>

 <table> 
    <tr> 
        <th align=\centre\><b>Features</b></th> 
        <th align=\centre\><b>Description</b></th> 
    </tr> 
    <tr> 
        <td>Age</td>  
        <td>Age of the patient</td> 
    </tr> 
    <tr> 
        <td>Sex </td> 
        <td>Sex of the patient</td> 
    </tr> 
    <tr> 
        <td>cp </td> 
        <td>Chest Pain type chest pain type: 
            <ul> 
                <li>Value 1: Typical angina</li> 
                <li>Value 2: Atypical angina</li> 
                <li>Value 3: Non-anginal pain</li> 
                <li>Value 4: Asymptomatic</li> 
            </ul> 
        </td> 
    </tr> 
    <tr> 
        <td>trtbps </td> 
        <td>Resting Blood Pressure (in mm Hg)</td> 
    </tr> 
    <tr> 
        <td>chol </td> 
        <td>Cholestoral in mg/dl fetched via BMI Sensor</td> 
    </tr> 
    <tr> 
        <td>fbs </td> 
        <td> 
            (Fasting blood sugar > 120 mg/dl)  
            <li>1 = true</li> 
            <li>0 = false </li> 
        </td> 
    </tr> 
    <tr> 
        <td>restecg</td> 
        <td> 
            <ol> 
                <li>Value 0: Normal</li> 
                <li>Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)</li> 
                <li>Value 2: Showing probable or definite left ventricular hypertrophy by Estes criteria</li> 
            </ul> 
        </td> 
    </tr> 
    <tr> 
        <td>thalach </td> 
        <td>Maximum Heart Rate achieved</td> 
    </tr> 
    <tr> 
        <td>exang</td> 
        <td>Exercise induced angina                         
        <li>1 = yes</li> 
            <li>0 = no</li> 
        </td> 
    </tr> 
    <tr> 
        <td>Oldpeak</td>  
        <td>ST depression induced by exercise relative to rest </td> 
    </tr> 
    <tr> 
        <td>slp </td> 
        <td>Peak exercise ST segment Slop 
            <li> 0 = Downsloping</li> 
            <li> 1 = Flat</li> 
            <li> 2 = Upsloping</li> 
        </td> 
    </tr> 
    <tr> 
        <td>caa</td>  
        <td>The number of major vessels (0–3)</td> 
    </tr> 
    <tr> 
        <td>thall </td> 
        <td>A blood disorder called Thalassemia 
            <li> Value 1: fixed defect (no blood flow in some part of the heart)</li> 
            <li> Value 2: normal blood flow</li> 
            <li> Value 3: reversible defect (a blood flow is observed but it is not normal)</li> 
        </td> 
    </tr> 
    <tr> 
        <td>target </td> 
        <td>Percentage of deliverable volume 
            <li> 0= less chance of heart attack</li> 
            <li> 1= more chance of heart attack</li> 
        </td> 
    </tr> 
</table><br> 

 

___
<h2>Key Project Takeaways</h2>
This project provided hands-on experience in real-time data handling and working behind Neural Networks :<br><br>
  <ul>Data preprocessing and cleaning for training and testing the data</ul>
  <ul>Building an efficient Neural Network <b>(Multi-Layer Perceptron)</b> from scratch using NumPy</ul>
  <ul>Mathematics behind <b> Activation Functions</b> and <b>Gradient Losses</b></ul>
  <ul>Web-app development using Streamlit</ul>
