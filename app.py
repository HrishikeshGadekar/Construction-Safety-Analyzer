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

    vectorizer = TfidfVectorizer(min_df= 2, sublinear_tf=True, norm='l2', ngram_range=(1, 3))
    #final_features = vectorizer.fit_transform(request.form.values()).toarray()
    
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
    prediction = model.predict(request.form.values())
    
    # Output Confidence scores
    probability_class_GO = model.predict_proba(request.form.values())[:, 0]
    probability_class_UA = model.predict_proba(request.form.values())[:, 1]
    probability_class_UC = model.predict_proba(request.form.values())[:, 2]

    
    if prediction==0:
        return render_template('index.html',
                               prediction_text='{}: This is a Good Observation!\n         The Prediction Confidence for each class are- \n GO: {}, UA: {}, UC: {}'.format(prediction, probability_class_GO, probability_class_UA,probability_class_UC),
                               
                               )
    elif prediction==1:
        return render_template('index.html',
                               prediction_text='{}: This is an Unsafe Act !!\n            The Prediction Confidence for each class are- \n GO: {}, UA: {}, UC: {}'.format(prediction, probability_class_GO, probability_class_UA,probability_class_UC),
                               )
    else:
        return render_template('index.html',
                               prediction_text='{}: This is an Unsafe Condition !!\n      The Prediction Confidence for each class are- \n GO: {}, UA: {}, UC: {}'.format(prediction, probability_class_GO, probability_class_UA,probability_class_UC),
                              )

if __name__ == "__main__":
    app.run(debug=True)
