from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_file = 'trained_model.sav'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from the form
        pregnancies = int(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        # Create DataFrame from user input
        data = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]], 
            columns=[
                'Pregnancies', 
                'Glucose', 
                'BloodPressure', 
                'SkinThickness', 
                'Insulin', 
                'BMI', 
                'DiabetesPedigreeFunction', 
                'Age'
            ]
        )
        
        # Make prediction
        prediction = model.predict(data)[0]
        
        # Output result
        result = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
