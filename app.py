from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

# load columns
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = float(request.form['insured_sex'])
        premium = float(request.form['policy_annual_premium'])
        hour = float(request.form['incident_hour_of_the_day'])
        vehicles = float(request.form['number_of_vehicles_involved'])
        damage = float(request.form['incident_severity_Major Damage'])

        # create full input with zeros
        input_data = np.zeros(len(columns))

        # manually map important features
        for i, col in enumerate(columns):
            if col == 'age':
                input_data[i] = age
            elif col == 'insured_sex':
                input_data[i] = sex
            elif col == 'policy_annual_premium':
                input_data[i] = premium
            elif col == 'incident_hour_of_the_day':
                input_data[i] = hour
            elif col == 'number_of_vehicles_involved':
                input_data[i] = vehicles
            elif col == 'incident_severity_Major Damage':
                input_data[i] = damage

        final_input = input_data.reshape(1, -1)

        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)[0][1]
        print("Final Input:", final_input)
        print("Prediction:", prediction)
        result = f"Fraud 🚨 ({probability*100:.2f}% chance)" if prediction[0] == 1 else f"Not Fraud ✅ ({(1-probability)*100:.2f}% safe)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)