import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# Nama-nama kelas (kelas-kelas yang sudah dilatih pada model)
class_names = [
    'African_Wildcat', 'Blackfoot_Cat', 'Cheetah', 'Domestic_Cat',
    'European_Wildcat', 'Jaguar', 'Leopard', 'Lion',
    'Macan_Tutul_Salju', 'Puma_Concolor', 'Sand_Cat', 'Tiger'
]

# Memuat model yang telah disimpan
@st.cache_resource
def load_model_cached():
    return load_model('initial_model.h5')

model = load_model_cached()

# Fungsi prediksi
@tf.function
def predict_image(img_array):
    return model(img_array)

# Data taksonomi
taxonomy_data = {
    "African_Wildcat": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Felis, Species: Felis lybica",
        "deskripsi": "African Wildcat adalah leluhur langsung dari kucing domestik, ditemukan di Afrika dan sebagian Asia."
    },
    "Blackfoot_Cat": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Felis, Species: Felis nigripes",
        "deskripsi": "Blackfoot Cat adalah salah satu spesies kucing terkecil di dunia, hidup di padang rumput Afrika Selatan."
    },
    "Cheetah": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Acinonyx, Species: Acinonyx jubatus",
        "deskripsi": "Cheetah adalah kucing besar yang terkenal karena kecepatannya yang luar biasa, mencapai hingga 112 km/jam."
    },
    "Domestic_Cat": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Felis, Species: Felis catus",
        "deskripsi": "Domestic Cat adalah kucing peliharaan yang telah beradaptasi hidup bersama manusia selama ribuan tahun."
    },
    "European_Wildcat": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Felis, Species: Felis silvestris",
        "deskripsi": "European Wildcat adalah kucing liar yang hidup di hutan-hutan Eropa dan Kaukasus."
    },
    "Jaguar": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera onca",
        "deskripsi": "Jaguar adalah kucing besar yang hidup di Amerika Selatan dan dikenal dengan gigitan terkuat di antara kucing besar."
    },
    "Leopard": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera pardus",
        "deskripsi": "Leopard adalah kucing besar yang memiliki pola totol khas, hidup di Afrika dan Asia."
    },
    "Lion": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera leo",
        "deskripsi": "Lion adalah kucing besar yang dikenal sebagai 'raja hutan', hidup di padang rumput Afrika."
    },
    "Macan_Tutul_Salju": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera uncia",
        "deskripsi": "Macan Tutul Salju adalah kucing besar yang hidup di pegunungan tinggi Asia Tengah, terkenal dengan bulunya yang tebal."
    },
    "Puma_Concolor": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Puma, Species: Puma concolor",
        "deskripsi": "Puma, atau dikenal juga sebagai cougar, adalah kucing besar yang hidup di Amerika Utara hingga Selatan."
    },
    "Sand_Cat": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Felis, Species: Felis margarita",
        "deskripsi": "Sand Cat adalah kucing kecil yang hidup di padang pasir Afrika Utara, Timur Tengah, dan Asia Tengah."
    },
    "Tiger": {
        "taksonomi": "Kingdom: Animalia, Phylum: Chordata, Class: Mammalia, Order: Carnivora, Family: Felidae, Genus: Panthera, Species: Panthera tigris",
        "deskripsi": "Tiger adalah kucing besar yang dikenal dengan corak belangnya, hidup di Asia dan menjadi simbol kekuatan."
    }
}

# Aplikasi Streamlit
st.set_page_config(page_title="Klasifikasi Kucing - Keluarga Felidae", page_icon="🐱", layout="wide")

# Sidebar
st.sidebar.image("https://static.wikia.nocookie.net/prehistrico/images/4/40/Felidae.jpg/revision/latest?cb=20140521182910&path-prefix=es", use_column_width=True)
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan model deep learning untuk mengklasifikasikan "
    "gambar kucing dari keluarga taksonomi Felidae. Upload gambar kucing, "
    "dan aplikasi akan menampilkan prediksi spesies, taksonomi, serta deskripsi singkat."
)

# Main content
st.title("🐱 Klasifikasi Kucing - Keluarga Felidae")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### Cara Penggunaan:
        1. Upload gambar kucing dari keluarga Felidae
        2. Tunggu hasil prediksi
        3. Lihat informasi taksonomi dan deskripsi
        """
    )
    
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file is not None:
        # Proses gambar
        img = Image.open(uploaded_file)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = predict_image(img_array)
        predictions_np = predictions.numpy()[0]
        predicted_class_index = np.argmax(predictions_np)
        predicted_class = class_names[predicted_class_index]
        predicted_prob = np.max(predictions_np)

        # Tampilkan hasil
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)
        st.subheader("Hasil Prediksi")
        st.write(f"**Spesies yang terdeteksi:** {predicted_class}")
        st.write(f"**Probabilitas:** {predicted_prob*100:.2f}%")

        # Tampilkan taksonomi
        species_info = taxonomy_data.get(predicted_class, {})
        if species_info:
            st.markdown("### Informasi Taksonomi dan Deskripsi")
            st.write(f"**Taksonomi:** {species_info['taksonomi']}")
            st.write(f"**Deskripsi:** {species_info['deskripsi']}")

        # Probabilitas untuk setiap kelas
        st.markdown("### Probabilitas Semua Kelas")
        prob_df = pd.DataFrame({
            'Spesies': class_names,
            'Probabilitas': predictions_np * 100
        })
        prob_df = prob_df.sort_values('Probabilitas', ascending=False).reset_index(drop=True)
        st.dataframe(prob_df, use_container_width=True)

        # Visualisasi probabilitas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(prob_df['Spesies'], prob_df['Probabilitas'])
        ax.set_xlabel('Probabilitas (%)')
        ax.set_ylabel('Spesies')
        ax.set_title('Probabilitas Prediksi untuk Setiap Spesies')
        st.pyplot(fig)

st.markdown("---")
st.info("Pastikan gambar yang diunggah merupakan anggota keluarga Felidae untuk hasil yang lebih akurat.")

