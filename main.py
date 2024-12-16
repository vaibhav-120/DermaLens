import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="DermaLens", layout="wide", page_icon="🩺")

def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

CLASS_NAMES = [
    'BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus',
    'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles'
]

st.sidebar.title("Navigation")
pages = st.sidebar.radio(
    "Choose a page:",
    ["Home", "Skin Disease Classification", "About"]
)

if pages == "Home":
    st.title("🩺 DermaLens")
    st.write("""
    Welcome to **DermaLens**, your AI-powered skin health companion.  
    This app leverages advanced machine learning models to analyze images of skin conditions and provide insights.  
    **Key Features**:
    - Upload images to detect common skin diseases.
    - Learn about symptoms, causes, and remedies for each condition.
    - Estimate severity based on symptoms.
    - Track healing progress over time.

    **Disclaimer**: This app is for educational purposes only. Always consult a medical professional for diagnosis and treatment.
    """)

elif pages == "Skin Disease Classification":
    st.title("Skin Disease Classification")
    st.write("Upload an image of the affected skin area to analyze and classify the condition.")

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((224, 224))
        image_array = np.expand_dims(image, axis=0)

        predictions = model.predict(image_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.subheader("Prediction")
        st.success(f"**Disease Type:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.subheader("More About This Disease")
        if st.button("Get Information"):
            # Fetch detailed information (hardcoded for now, replace with a database or API later)
            disease_info = {
                'BA-cellulitis': "Cellulitis is a bacterial infection of the skin and tissues beneath it. Common symptoms include redness, swelling, and pain.",
                'BA-impetigo': "Impetigo is a contagious bacterial skin infection that often starts as red sores or blisters.",
                'FU-athlete-foot': "Athlete's foot is a fungal infection causing itching, cracking, and peeling of the skin between the toes.",
                'FU-nail-fungus': "Nail fungus is a common condition that begins as a white or yellow spot under the nail tip and may cause nail discoloration and thickening.",
                'FU-ringworm': "Ringworm is a fungal infection that appears as a circular, red, itchy rash on the skin.",
                'PA-cutaneous-larva-migrans': "Cutaneous larva migrans is a skin condition caused by hookworm larvae, resulting in itchy, winding rash patterns.",
                'VI-chickenpox': "Chickenpox is a viral infection characterized by an itchy, blister-like rash. It's highly contagious.",
                'VI-shingles': "Shingles is caused by the varicella-zoster virus and results in a painful rash, often with blisters."
            }
            st.write(disease_info.get(predicted_class, "Information not available."))

        if st.button("Get Remedies"):
            # Fetch remedies (hardcoded for now, replace with a database or API later)
            remedies = {
                'BA-cellulitis': "Treatment involves antibiotics. Keep the area clean and elevate it to reduce swelling.",
                'BA-impetigo': "Apply prescription antibiotic ointments and maintain hygiene to prevent spread.",
                'FU-athlete-foot': "Use antifungal creams or sprays. Keep feet dry and wear breathable footwear.",
                'FU-nail-fungus': "Apply antifungal nail polish or take oral antifungal medications as prescribed.",
                'FU-ringworm': "Topical antifungal creams or powders can be used. Avoid sharing personal items.",
                'PA-cutaneous-larva-migrans': "Treat with antiparasitic medications. Avoid walking barefoot in contaminated areas.",
                'VI-chickenpox': "Use calamine lotion for itching. Stay hydrated and rest. Avoid scratching to prevent scarring.",
                'VI-shingles': "Antiviral medications like acyclovir can help. Use pain relievers and keep the rash clean."
            }
            st.write(remedies.get(predicted_class, "Remedies not available."))

elif pages == "About":
    st.title("About DermaLens")
    st.write("""
    **DermaLens** is a state-of-the-art AI-powered tool designed to assist medical professionals, students, and individuals in understanding and identifying common skin diseases.
    - This app is developed by **Vaibhav Shrivastava**.
    - It leverages deep learning models trained on dermatological datasets to deliver high-accuracy predictions.
    - The app provides detailed information and remedies for several common skin diseases.
    - It is intended for educational purposes only and should not replace professional medical consultation.

    **Technologies Used**:
    - TensorFlow/Keras for model training.
    - Streamlit for web app development.
    - PIL and NumPy for image preprocessing.

    For feedback or suggestions, reach out to **vs2409425@gmail.com**.
    """)


st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Vaibhav Shrivastava**")
st.sidebar.markdown("This app is for educational purposes only. Consult a medical professional for accurate diagnosis and treatment.")
