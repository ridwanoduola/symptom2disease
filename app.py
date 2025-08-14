import streamlit as st
import joblib

# ===== Load Saved Objects =====
svc_model = joblib.load("svc_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ===== Streamlit App UI =====
st.title("🩺 Symptom to Disease Classifier")
st.write("Enter patient symptoms and get a predicted disease.")

# Disclaimer
st.markdown(
    """
    > **Disclaimer:** This tool is intended for informational purposes only 
    > and does not replace professional medical advice, diagnosis, or treatment.  
    > Always consult a qualified healthcare professional for any medical concerns.  
    > Predictions with a confidence level below **90%** should not be relied upon for decision-making.
    """
)

# Input text box
user_input = st.text_area("Describe the symptoms here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms before predicting.")
    else:
        # Transform input using vectorizer
        input_tfidf = vectorizer.transform([user_input])
        
        # Predict
        prediction = svc_model.predict(input_tfidf)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Predict probabilities
        proba = svc_model.predict_proba(input_tfidf)[0]
        top_probs = sorted(
            list(zip(label_encoder.classes_, proba)),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Display result
        top_disease, top_confidence = top_probs[0]
        if top_confidence < 0.90:
            st.warning(f"⚠ Predicted Disease: **{top_disease}** (Confidence: {top_confidence:.2%}) — Below reliable threshold.")
        else:
            st.success(f"✅ Predicted Disease: **{top_disease}** (Confidence: {top_confidence:.2%})")
        
        # Show probabilities
        st.subheader("Prediction Probabilities")
        for disease, prob in top_probs:
            st.write(f"{disease}: {prob:.2%}")