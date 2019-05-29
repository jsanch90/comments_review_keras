from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
from time import sleep
import keras
from keras.models import load_model
import tensorflow as tf
import re
import numpy as np

app = Flask(__name__)
CORS(app)

global graph
graph = tf.get_default_graph()

model = load_model('saved_models/model_drop_9351.h5')
print('----------Model loaded----------')

word_index = keras.datasets.imdb.get_word_index()

def get_indices_from_review(review):
    regex = re.compile(r'[!"#$%&\()*+,-./:;<=>?@\[\]\\^_`{|}~]')
    s = regex.sub('', review)
    # 2 is "unknown"
    sequence = map(lambda word: word_index.get(word, 2) + 3, s.lower().split())
    sequence = map(lambda index: 2 if index >= 80000 else index, sequence)
    # 1 is "start of sequence"
    return [1] + list(sequence)

def vectorize_sequences(sequences, dim):
    vec = np.zeros(shape=(len(sequences), dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        vec[i, seq] = 1
    return vec

def predict(review):
    review_vec = get_indices_from_review(review)
    vec = vectorize_sequences([review_vec], dim=80000)
    res = np.squeeze(model.predict(vec))
    return res * 100

@app.route('/',methods=['GET', 'POST'])
def index():
    score=0
    if request.method == 'POST':
        with graph.as_default():
            score = predict(request.form['description'])
        data = request.form['description']
        score2 = round(score,3)
        score = str(round(score,5))+'%'
        return render_template('report.html',score2=score2, score=str(score), data=data)
    else:
        return render_template('report.html',score=str(score))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000 ,debug=True)