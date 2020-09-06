
from flask import Flask, request, jsonify
import json
import flask
import pandas as pd
import requests
import numpy
import pickle
import sklearn
import numpy as np
from flask_cors import CORS  # This is the magic

with open('fin.pkl', 'rb') as f:
    data = pickle.load(f)
model = data

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=["POST"])
# @cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def hello_world():
    print(request.is_json)
    content = request.get_json()

    array = []
    array.append(int(content['gender']))
    array.append(float(content['sphere']))
    array.append(float(content['acd']))
    array.append(float(content['lt']))
    array.append(float(content['sport']))
    array.append(int(content['mommy']))
    array.append(int(content['daddy']))
    array.append(int(content['age'] == 9))
    np_array = np.array(array)[np.newaxis]

    ans_ar = model.predict_proba(np_array)
    ans = ans_ar[0][1]*100
    ans = jsonify(ans)
    return ans


if __name__ == '__main__':
    app.run(debug=True)
