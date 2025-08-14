# 🩺 Symptom to Disease Classifier

This is a Machine Learning application built with **Streamlit** that predicts possible diseases based on user-provided symptoms.

## Features
- Text-based symptom input
- Predicts disease using a trained SVM model
- Displays prediction probabilities
- Includes medical disclaimer
- Warns when confidence is below 90%

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/symptom2disease.git
cd symptom2disease
pip install -r requirements.txt
```

## Usage
Run the app locally:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

## Disclaimer
This tool is for informational purposes only and does not replace professional medical advice.  
Always consult a healthcare professional for medical concerns.  
Predictions below **90% confidence** should not be relied upon for decision-making.
