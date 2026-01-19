import gradio as gr
import pandas as pd
import pickle
import cloudpickle
import numpy as np

# 1. Load the Model
with open('best_model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_insurance(age, bmi, children, sex, smoker, region):
    
    # Feature engineering (calculated internally)
    bmi_per_age = bmi / age 
    bmi_log = np.log(bmi + 1)  # Log transformation
    age_group = 'Young' if age < 25 else ('Adult' if age < 65 else 'Senior')

    # Pack inputs into a DataFrame 
    input_df = pd.DataFrame([[ 
        age, bmi, children, sex, smoker, region,
        bmi_per_age, bmi_log, age_group
    ]], 
    columns=[ 
        'age', 'bmi', 'children', 'sex', 'smoker', 'region',
        'bmi_per_age', 'bmi_log', 'age_group'
    ])
    
    # Predict the insurance charge using the model
    prediction = model.predict(input_df)[0]
    
    # Return formatted result (prediction in dollar value)
    return f"Predicted Insurance Charge: ${prediction:,.2f}"

# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    gr.Number(label="Age", value=30, interactive=True),
    gr.Number(label="BMI", value=25.0, interactive=True), 
    gr.Slider(0, 10, step =1, label="Children"), 
    gr.Radio(["male", "female"], label="Gender"),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Radio(["southwest", "southeast", "northwest", "northeast"], label="Region"),
]

app = gr.Interface(
    fn=predict_insurance,
    inputs=inputs,
    outputs="text", 
    title="Medical Insurance Prediction",
    description="Enter the details to predict medical insurance charges based on the model."
)

app.launch(share=True)