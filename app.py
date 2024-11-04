from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = 'cancer.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app=Flask(__name__,template_folder='Templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    Gender = [a for a in request.form.values()]
    Age = [a for a in request.form.values()]
    Smoking = [a for a in request.form.values()]
    Yellow_Fingers = [a for a in request.form.values()]
    Anxiety = [a for a in request.form.values()]
    Peer_Pressure = [a for a in request.form.values()]
    Chronic_Disease = [a for a in request.form.values()]
    Fatigue = [a for a in request.form.values()]
    Allergy = [a for a in request.form.values()]
    Wheezing = [a for a in request.form.values()]
    Alchohol_Consuming = [a for a in request.form.values()]
    Coughing = [a for a in request.form.values()]
    Shortness_Of_Breath = [a for a in request.form.values()]
    Swallowing_Difficulty = [a for a in request.form.values()]
    Chest_Pain = [a for a in request.form.values()]
    
    final_features = pd.DataFrame([Gender,Age,Smoking,Yellow_Fingers,Anxiety,Peer_Pressure,Chronic_Disease,Fatigue,Allergy,Wheezing,Alchohol_Consuming,Coughing,Shortness_Of_Breath,Swallowing_Difficulty,Chest_Pain],columns=['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH','SWALLOWING_DIFFICULTY','CHEST_PAIN'])
    final_features.replace(to_replace={'no':1,'yes':2,'No':1,'Yes':2,'NO':1,'YES':2},inplace=True)
    final_features["GENDER"].replace(to_replace={'MALE':2,'FEMALE':1,'Male':2,'Female':1,'male':2,'female':1},inplace=True)
    
    
    
    # Make prediction
    prediction =model.predict(final_features)
    #prediction.replace(to_replace ={0:'Not Have Heart Disease',1:'Have Heart Disease'},inplace=True)
    output = prediction[0]
    return render_template('index.html', prediction_text='You not have heart disease' if output ==0 else 'You have heart disease')

    #return render_template('index.html', prediction_text='You {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)