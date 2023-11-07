import joblib
import warnings
from flask import jsonify
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=DataConversionWarning)
# Load the trained model
model = joblib.load('model4.pkl')
scaler = joblib.load('standard_scaler.pkl')
from flask import Flask, render_template, request
app = Flask(__name__, template_folder='C:\\Users\\NARESHSARATHY S\\Downloads\\river_quality')
@app.route('/')
def home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        bsk5 = float(request.form['bsk5'])
        suspended = float(request.form['suspended'])
        o2 = float(request.form['o2'])
        no3 = float(request.form['no3'])
        so4 = float(request.form['so4'])
        po4 = float(request.form['po4'])
        cl = float(request.form['cl'])
        # Create a feature vector with the user input
        user_input = [bsk5, suspended, o2, no3, so4, po4, cl] 
        # Scale the input data
        scaled_input = scaler.transform([user_input])
        # Make a prediction
        predicted_value = model.predict(scaled_input)[0]
        # Display the prediction on a result page
        return jsonify({'prediction': predicted_value})
if __name__ == '__main__':
    app.run(debug=True)
