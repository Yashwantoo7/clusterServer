from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
import pickle
# import numpy as np


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

kmeans = pickle.load(open('model.pkl','rb'))

# @app.route("/")
# def home():
#     return "hello world"

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json()
    clusters=[]
    for i in list(kmeans.predict(data['data'])):
        clusters.append(int(i))
    return jsonify(clusters)
