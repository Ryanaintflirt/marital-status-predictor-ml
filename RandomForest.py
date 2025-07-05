import pandas as pd
import numpy as np
from joblib import dump
from joblib import load
data = pd.read_csv(r"")

#data prepocessing
# Data Encoding -> (0,1) or (true, false)
from sklearn.preprocessing import LabelEncoder
Martial_encoder = LabelEncoder()
X = data.drop(columns=['Marital Status'])
y = data['Marital Status']
y = Martial_encoder.fit_transform(y)
# Save the fitted encoder
dump(Martial_encoder, 'marital_encoder.joblib')
X = pd.get_dummies(X, columns=['Gender'])

#split train data, test data and training 20% to make accuracy   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# model train 
from sklearn.ensemble import RandomForestClassifier#forest classifier
from sklearn.metrics import accuracy_score #how much accuracy in model evaluation 
model = RandomForestClassifier(n_estimators=100,max_depth=1000, random_state=42) #define model
model.fit(X_train,y_train) #train model
#RandomForestClassifier(max_depth=10, random_state=42)

#model evaluation CHECK accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test) #predict
accuracy = accuracy_score(y_test, y_pred) #comparison and make score
print("Model Accuracy : ", accuracy)

y_pred =Martial_encoder.inverse_transform(y_pred.astype(int))

#Save trained model
dump(model,'model.joblib')
#load model
model = load('model.joblib')

new_data = pd.DataFrame({
    'Age' : np.random.randint(18,60, size = 10),
    'Income' : np.ceil(np.random.randint(0,100000, size = 10)/1000)*1000,
    'Gender' : np.random.choice(['Male','Female'],size = 10),
    'Marital Status' : np.random.choice(['Single','Married'],size =10)
})

new_data = pd.get_dummies(new_data, columns=['Gender'])
new_data_pred_features = new_data.drop(columns=['Marital Status'])

new_data_target=new_data['Marital Status']

new_pred = model.predict(new_data_pred_features)
new_pred = Martial_encoder.inverse_transform(new_pred.astype(int))

test_accuracy = accuracy_score(new_data_target, new_pred)
print(f"Test Accuracy: {test_accuracy}")

# Load model and encoder
model = load('model.joblib')
encoder = load('marital_encoder.joblib')  # This must be fitted

# Prepare test data
test_data = pd.DataFrame({
    'Age': [30],
    'Income': [500000],
    'Gender': ['Female']
})

# One-hot encode Gender
test_data = pd.get_dummies(test_data)

# Ensure all columns exist
for col in ['Gender_Female', 'Gender_Male']:
    if col not in test_data:
        test_data[col] = 0

test_data = test_data[['Age', 'Income', 'Gender_Female', 'Gender_Male']]

# Predict
y_pred = model.predict(test_data)

# Inverse transform to label (make sure it's fitted!)
predicted_label = encoder.inverse_transform(y_pred.astype(int))
print("Predicted Marital Status:", predicted_label[0])