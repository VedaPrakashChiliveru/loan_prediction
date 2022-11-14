import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from pyexpat import model

app = Flask(__name__)
catb = pickle.load(open('modelcat.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    prediction = catb.predict(final_features)
    return render_template('after.html', data=prediction)


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=false,host='0.0.0.0')
