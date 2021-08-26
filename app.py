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

    
    if prediction==0:
        return render_template('index.html',
                               prediction_text='{}: This is a Good Observation!'.format(prediction),
                               )
    elif prediction==1:
        return render_template('index.html',
                               prediction_text='{}: This is an Unsafe Act !!'.format(prediction),
                               )
    else:
        return render_template('index.html',
                               prediction_text='{}: This is an Unsafe Condition !!'.format(prediction),
                              )

if __name__ == "__main__":
    app.run(debug=True)