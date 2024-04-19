import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction
    if output == 0:
        output = 10-26-26
    elif output == 1:
        output = "14-35-14"
    elif output == 2:
        output= "17-17-17"
    elif output == 3:
        output= "20-20"
    elif output==4:
        output = "28-28"
    elif output == 5:
        output = "DAP"
    else:
        output= "Urea"

    return render_template('index.html', prediction_text='Fertilizer used should be {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)