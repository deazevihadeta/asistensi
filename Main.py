from flask import Flask, jsonify, request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
import pandas as pandas


app = Flask(__name__)
Swagger(app)
# CORS(app)

@app.route('/predict/task', methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Memprediksi Jenis Hewan
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Petal
          required:
            - milk
            - toothed
            - eggs
            - feathers
            - backbone
            - legs
            - breathes
            - tail
            - fins
            - aquatic
          properties:
            milk:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            toothed:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            eggs:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            feathers:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            backbone:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            legs:
              type: int
              description: Please input with value 0, 2, 4, 6, or 8.
              default: 0
            breathes:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            tail:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            fins:
              type: int
              description: Please input with value 0 or 1.
              default: 0
            aquatic:
              type: int
              description: Please input with value 0 or 1.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    milk = new_task['milk']
    toothed = new_task['toothed']
    eggs = new_task['eggs']
    feathers = new_task['feathers']
    backbone = new_task['backbone']
    legs = new_task['legs']
    breathes = new_task['breathes']
    tail = new_task['tail']
    fins = new_task['fins']
    aquatic = new_task['aquatic']

    X_New = np.array([[legs, toothed, backbone, tail, breathes, feathers, milk,aquatic, fins, eggs]])
    # X_New = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 4, 0, 0, 0]])
    clf = joblib.load('randomForestClassifier.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message': format(clf[1][resultPredict-1])})


if __name__ == '__main__':
    app.run()