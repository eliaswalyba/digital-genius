from flask import Flask, render_template, url_for,request
import pickle
from sklearn.externals import joblib
from model import Model

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        choosen_model = request.form["model"]
        Classifier = open(f'../saved_models/{choosen_model}_model.pkl', 'rb')
        Classifier = joblib.load(Classifier)
        message = request.form['message']
        pred = Classifier.test([message])
        pred = 'Order Status' if pred[0] == 'order_status' else 'Cancel Order'
    return render_template('result.html', prediction = pred, message = message)

if __name__ == '__main__':
	app.run(debug=True)