import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan fitur terpilih
with open('depresi_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Fitur input untuk pengguna
st.title("Prediksi Depresi Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi kemungkinan mengalami depresi.")

gender = st.radio("Jenis Kelamin", ("Laki-laki", "Perempuan"))
age = st.slider("Umur", 16, 35, 20)
year = st.selectbox("Tahun Studi Saat Ini", (1, 2, 3, 4))
gpa = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
anxiety = st.radio("Apakah Anda Mengalami Kecemasan?", ("Ya", "Tidak"))
panic = st.radio("Apakah Anda Pernah Mengalami Serangan Panik?", ("Ya", "Tidak"))
treatment = st.radio("Pernahkah Anda Menemui Spesialis Kesehatan Mental?", ("Ya", "Tidak"))

# Proses input
input_dict = {
    'Choose your gender': 1 if gender == "Laki-laki" else 0,
    'Age': age,
    'Your current year of Study': year,
    'CGPA_numeric': gpa,
    'Do you have Anxiety?': 1 if anxiety == "Ya" else 0,
    'Do you have Panic attack?': 1 if panic == "Ya" else 0,
    'Did you seek any specialist for a treatment?': 1 if treatment == "Ya" else 0,
    'GPAxYear': gpa * year
}

# Konversi ke DataFrame
input_df = pd.DataFrame([input_dict])
input_selected = input_df[selected_features]

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_selected)[0]
    if prediction == 1:
        st.error("Hasil: Mahasiswa kemungkinan mengalami depresi.")
    else:
        st.success("Hasil: Mahasiswa kemungkinan tidak mengalami depresi.")
