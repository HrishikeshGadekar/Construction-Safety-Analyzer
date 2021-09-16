import numpy as np
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
# read our pickle file and label our logisticmodel as model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    prediction = model.predict(request.form.values())
    
    # Output Confidence scores
    probability_class_GO = model.predict_proba(request.form.values())[:, 0]
    probability_class_UA = model.predict_proba(request.form.values())[:, 1]
    probability_class_UC = model.predict_proba(request.form.values())[:, 2]
    

    if probability_class_GO[0]==0.08765182261021813 or probability_class_UA[0]==0.1355507998162925 or probability_class_UC[0]==0.7767973775734893:
        return render_template('index.html',
                               inp='Input: {}'.format(request.form.to_dict(flat=True)['Safety Observation']),
                               prediction_text='---> This is an Invalid Input. Please try again.',
                               )
    elif prediction==0:
        return render_template('index.html',
                               inp='Input: {}'.format(request.form.to_dict(flat=True)['Safety Observation']),
                               prediction_text='---> This is a Good Observation!\n The Prediction Confidence for each class are-\n GO: {},\n UA: {},\n UC: {}'.format(probability_class_GO[0], probability_class_UA[0],probability_class_UC[0]),
                               )
    elif prediction==1:
        return render_template('index.html',
                               inp='Input: {}'.format(request.form.to_dict(flat=True)['Safety Observation']),
                               prediction_text='---> This is an Unsafe Act !!\nThe Prediction Confidence for each class are- \n GO: {}, UA: {}, UC: {}'.format(probability_class_GO[0], probability_class_UA[0],probability_class_UC[0])
                               )
    else:
        return render_template('index.html',
                               inp='Input: {}'.format(request.form.to_dict(flat=True)['Safety Observation']),
                               prediction_text='---> This is an Unsafe Condition !!\nThe Prediction Confidence for each class are- \n GO: {}, UA: {}, UC: {}'.format(probability_class_GO[0], probability_class_UA[0],probability_class_UC[0])
                              )

if __name__ == "__main__":
    app.run(debug=True)
