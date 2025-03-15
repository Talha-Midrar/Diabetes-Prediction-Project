from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and feature names
model = joblib.load('diabetes_pred_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Load mean and std of Glucose (for z-score calculation)
glucose_stats = np.load('glucose_stats.npz')
glucose_mean = glucose_stats['mean']
glucose_std = glucose_stats['std']

# Function to generate BMI pie chart
def generate_bmi_pie_chart(df):
    # Create BMI categories if they don't exist
    if 'BMI_Category' not in df.columns:
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 25, 30, 100], labels=['Normal', 'Overweight', 'Obese'])

    # Count the number of people in each BMI category
    bmi_counts = df['BMI_Category'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(bmi_counts, labels=bmi_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('BMI Distribution')

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{chart_image}" alt="BMI Pie Chart">'

# Function to generate Age bar chart
def generate_age_bar_chart(df):
    # Create age groups if they don't exist
    if 'Age_Group' not in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])

    # Count the number of people in each age group
    age_counts = df['Age_Group'].value_counts()

    # Create a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(age_counts.index, age_counts.values, color='skyblue')
    plt.title('Age Group Distribution')
    plt.xlabel('Age Group')
    plt.ylabel('Count')

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{chart_image}" alt="Age Bar Chart">'

def generate_glucose_impact_chart(df):
    # Create a scatter plot to show glucose impact on diabetes
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Glucose'], df['Outcome'], alpha=0.5, color='blue')
    plt.title('Impact of Glucose Levels on Diabetes')
    plt.xlabel('Glucose Level')
    plt.ylabel('Diabetes Outcome (0 = No, 1 = Yes)')
    plt.grid(True)

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{chart_image}" alt="Glucose Impact Chart">'
def generate_probability_chart(probability):
    # Create a bar chart for probability
    plt.figure(figsize=(6, 4))
    plt.bar(['No Diabetes', 'Diabetes'], [1 - probability, probability], color=['green', 'red'])
    plt.title('Probability of Diabetes')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{chart_image}" alt="Probability Chart">'

def generate_risk_level_chart(probability):
    # Determine risk level based on probability
    if probability < 0.3:
        risk_level = 'Low Risk'
    elif probability < 0.7:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'High Risk'

    # Create a pie chart for risk level
    plt.figure(figsize=(6, 6))
    plt.pie([probability, 1 - probability], labels=[risk_level, ''], autopct='%1.1f%%', startangle=140, colors=['orange', 'lightgray'])
    plt.title('Diabetes Risk Level')

    # Save the chart to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'<img src="data:image/png;base64,{chart_image}" alt="Risk Level Chart">'
# Home page with visualizations
    
@app.route('/')
def home():
    # Load the dataset (for visualization purposes)
    df = pd.read_csv('diabetes_dataset.csv')

    # Generate charts
    bmi_chart = generate_bmi_pie_chart(df)
    age_chart = generate_age_bar_chart(df)
     # Generate charts
    bmi_chart = generate_bmi_pie_chart(df)
    age_chart = generate_age_bar_chart(df)
    glucose_chart = generate_glucose_impact_chart(df)

    # Render the home page with charts
    return render_template('index.html', bmi_chart=bmi_chart, age_chart=age_chart, glucose_chart=glucose_chart)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form.to_dict()

    # Convert data into a DataFrame
    input_data = pd.DataFrame([data])

    # Handle categorical variables
    age = float(input_data['Age'].iloc[0])
    if age >= 31 and age <= 50:
        input_data['Age_Group_31-50'] = 1
    else:
        input_data['Age_Group_31-50'] = 0

    if age >= 51:
        input_data['Age_Group_51+'] = 1
    else:
        input_data['Age_Group_51+'] = 0

    bmi = float(input_data['BMI'].iloc[0])
    if bmi < 25:
        input_data['BMI_Category_Normal'] = 1
        input_data['BMI_Category_Overweight'] = 0
        input_data['BMI_Category_Obese'] = 0
    elif bmi >= 25 and bmi < 30:
        input_data['BMI_Category_Normal'] = 0
        input_data['BMI_Category_Overweight'] = 1
        input_data['BMI_Category_Obese'] = 0
    else:
        input_data['BMI_Category_Normal'] = 0
        input_data['BMI_Category_Overweight'] = 0
        input_data['BMI_Category_Obese'] = 1

    glucose = float(input_data['Glucose'].iloc[0])
    if glucose >= 126:
        input_data['Glucose_Category_Diabetes'] = 1
        input_data['Glucose_Category_Prediabetes'] = 0
    elif glucose >= 100 and glucose < 126:
        input_data['Glucose_Category_Diabetes'] = 0
        input_data['Glucose_Category_Prediabetes'] = 1
    else:
        input_data['Glucose_Category_Diabetes'] = 0
        input_data['Glucose_Category_Prediabetes'] = 0

    input_data['z_score'] = (glucose - glucose_mean) / glucose_std

    # Add missing features with default values
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder input data columns to match the model's feature names
    input_data = input_data[feature_names]

    # Convert data types to numeric
    input_data = input_data.astype(float)

    # Make a prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Prepare the result
    result = {
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0][1])
    }

    # Generate charts for the result page
    probability_chart = generate_probability_chart(result['probability'])
    risk_level_chart = generate_risk_level_chart(result['probability'])


    # Render the result in a new HTML page
    return render_template('result.html', result=result, probability_chart=probability_chart, risk_level_chart=risk_level_chart)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
