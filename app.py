import streamlit as st
import pickle
import numpy as np

# === Load model dan fitur ===
with open('depresi_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('selected_features.pkl', 'rb') as features_file:
    selected_features = pickle.load(features_file)

# === Judul Aplikasi ===
st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="centered")
st.title("🎓 Prediksi Depresi Mahasiswa")
st.write("Isi data berikut untuk mengetahui kemungkinan depresi berdasarkan model pembelajaran mesin.")

# === Input dari pengguna ===
gender = st.radio("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
age = st.number_input("Usia", min_value=15, max_value=40, value=21)
year = st.selectbox("Tahun Studi Saat Ini", ['year 1', 'year 2', 'year 3', 'year 4'])
cgpa = st.slider("IPK Saat Ini", min_value=1.0, max_value=4.0, step=0.01, value=3.0)
anxiety = st.radio("Apakah Anda mengalami kecemasan (Anxiety)?", ['Ya', 'Tidak'])
panic = st.radio("Apakah Anda pernah mengalami serangan panik (Panic Attack)?", ['Ya', 'Tidak'])
treatment = st.radio("Apakah Anda pernah berkonsultasi dengan spesialis?", ['Ya', 'Tidak'])

# === Preprocessing Input ===
gender = 1 if gender == 'Laki-laki' else 0
year_mapping = {'year 1': 1, 'year 2': 2, 'year 3': 3, 'year 4': 4}
year = year_mapping[year]
anxiety = 1 if anxiety == 'Ya' else 0
panic = 1 if panic == 'Ya' else 0
treatment = 1 if treatment == 'Ya' else 0

gpaxyear = cgpa * year

# === Gabungkan input ke array ===
input_dict = {
    'Choose your gender': gender,
    'Age': age,
    'Your current year of Study': year,
    'CGPA_numeric': cgpa,
    'Do you have Anxiety?': anxiety,
    'Do you have Panic attack?': panic,
    'Did you seek any specialist for a treatment?': treatment,
    'GPAxYear': gpaxyear
}

# Hanya ambil fitur terpilih (6 fitur)
input_data = np.array([[input_dict[feat] for feat in selected_features]])

# === Prediksi ===
if st.button("Prediksi Depresi"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Model memprediksi Anda mengalami gejala depresi.")
    else:
        st.success("✅ Model memprediksi Anda tidak mengalami gejala depresi.")

    st.markdown("---")
    st.subheader("📌 Rekomendasi Umum:")
    if prediction == 1:
        st.write("- Cobalah untuk mencari bantuan dari psikolog kampus atau tenaga profesional.")
        st.write("- Pastikan Anda cukup istirahat, olahraga, dan berbagi cerita dengan orang terpercaya.")
    else:
        st.write("- Tetap jaga kesehatan mental Anda, dan bantu teman yang mungkin butuh dukungan.")
