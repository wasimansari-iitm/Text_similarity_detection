from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from helper import query_point_creator

app = Flask(__name__)
# Set your secret key here
app.secret_key = 'my_secret_key'

# Load models and scaler
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('word2vec_model.pkl', 'rb') as f:
    w2v_model = pickle.load(f)
wv = w2v_model.wv

# Global list to store previously asked questions
asked_questions = []

@app.route('/', methods=['GET', 'POST'])
@app.route('/predictor', methods=['GET', 'POST'])
def index():
    global asked_questions
    prediction = None
    proba = None
    q1 = q2 = ''
    if request.method == 'POST':
        q1 = request.form.get('question1', '')
        q2 = request.form.get('question2', '')
        if q1 and q2:
            flash('Vectorizing questions...', 'info')
            features = query_point_creator(q1, q2, wv=wv, scaler=scaler)
            flash('Scaling features...', 'info')
            features = features.reshape(1, -1)
            flash('Predicting...', 'info')
            prediction = int(xgb_model.predict(features)[0])
            proba = float(xgb_model.predict_proba(features)[0][1])
            # Append the questions to the global list
            asked_questions.append((q1, q2))
            # Debug logs to verify data passed to the template
            print(f"Prediction: {prediction}")
            print(f"Probability: {proba}")
            print(f"Question 1: {q1}")
            print(f"Question 2: {q2}")
            print(f"Asked Questions: {asked_questions}")
    return render_template('index.html', prediction=prediction, proba=proba, q1=q1, q2=q2, asked_questions=asked_questions)

if __name__ == '__main__':
    app.run(debug=True)

# To run the app, use the following command in your terminal:
# docker run -p 8000:8000 text-similarity-detector
# open http://localhost:8000 in your browser