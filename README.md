# Face Recognition 
Sistem face recognition berbasis web interaktif menggunakan Streamlit 
yang mengimplementasikan Eigenface dengan fitur auto-crop menggunakan Haar Cascade. 

# Fitur
- Eigen Calculation: Menghitung Nilai Eigen dan Vektor Eigen.
- Euclidean Distance: Menghitung jarak tingkat kemiripan wajah.
- Auto-Crop Face: Mendeteksi dan memotong wajah secara otomatis dari citra asli menggunakan OpenCV Haar Cascade sebelum diproses.

# Persyaratan 
Menginstal Python (versi 3.8 atau lebih baru). 
Library yang dibutuhkan:
- streamlit
- opencv-python
- numpy

# Installasi
Ketik:
pip install streamlit opencv-python numpy
Cara membuka project:
streamlit run src/facerecog.py

# Langkah Pengujian di Web:
   - Di sidebar, masukkan nama folder dataset (contoh:./dataset).
   - Klik browse files untuk mengunggah gambar baru (*Test Image*) yang ingin dikenali.
   - Sistem akan memproses dataset (menghitung Eigenface) dan mencari kecocokan wajah terdekat.
   - Hasil kemiripan dan nilai jarak Euclidean akan ditampilkan di layar.

# Pengaturan Threshold
Jika mengenali wajah yang salah atau sering memunculkan pesan 
"Tidak terdapat citra wajah yang mirip", ubah nilai threshold di dalam kode.
