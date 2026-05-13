import os
import cv2
import numpy as np
import streamlit as st
import time

#Haar Cascade untuk otomatis crop deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = gray[y:y+h, x:x+w]
        return cropped_face
    else:
        return gray

#Perhitungan euclidean distance 
def euclidean_distance(v1, v2):
    dist = 0.0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i])**2
    return dist**0.5

def eigen(A, num_components=30, iterations=100):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    A_copy = A.copy()
    
    for _ in range(min(num_components, n)):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(iterations):
            v_new = np.dot(A_copy, v)
            v_new_norm = np.linalg.norm(v_new)
            if v_new_norm == 0: break
            v_new = v_new / v_new_norm
            v = v_new
            
        eigenvalue = np.dot(v.T, np.dot(A_copy, v))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        
        A_copy = A_copy - eigenvalue * np.outer(v, v)
        
    return np.array(eigenvalues), np.array(eigenvectors).T

# Perhitungan eigenfaces 
@st.cache_data
def process_dataset(dataset_path, img_size=(64, 64)):
    images = []
    labels = []
    image_paths = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                
                cropped_gray = detect_and_crop_face(img)
                equalized_gray = cv2.equalizeHist(cropped_gray)
                resized = cv2.resize(equalized_gray, img_size)
                
                images.append(resized.flatten())
                labels.append(os.path.basename(root))
                image_paths.append(path)
                
    if len(images) == 0:
        return None, None, None, None, None
        
    matrix_images = np.array(images, dtype=np.float32)
    mean_face = np.mean(matrix_images, axis=0)
    phi = matrix_images - mean_face
    
    C = np.dot(phi.T, phi) / len(images)
    
    eigenvalues, eigenfaces = eigen(C, num_components=30)
    weights = np.dot(phi, eigenfaces)
    
    return mean_face, eigenfaces, weights, labels, image_paths

def recognize_face(test_img, mean_face, eigenfaces, weights, labels, image_paths, img_size=(64, 64)):
    cropped_gray_test = detect_and_crop_face(test_img)
    
    equalized_gray_test = cv2.equalizeHist(cropped_gray_test)
    
    resized_test = cv2.resize(equalized_gray_test, img_size)
    flattened_test = resized_test.flatten()
    
    phi_test = flattened_test - mean_face
    weight_test = np.dot(phi_test, eigenfaces)
    
    min_dist = float('inf')
    best_match_idx = -1
    
    for i in range(len(weights)):
        dist = euclidean_distance(weight_test, weights[i])
        if dist < min_dist:
            min_dist = dist
            best_match_idx = i
            
    return min_dist, labels[best_match_idx], image_paths[best_match_idx]

# GUI Streamlit
st.set_page_config(layout="wide")
st.title("Face Recognition")

st.sidebar.header("Inputs")
dataset_folder = st.sidebar.text_input("Insert Your Dataset Path", value="./dataset")
uploaded_file = st.sidebar.file_uploader("Insert Your Image", type=['jpg', 'png', 'jpeg'])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    start_time = time.time()
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test_img = cv2.imdecode(file_bytes, 1)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.subheader("Test Image")
        st.image(test_img_rgb, use_container_width=True)
        
    if os.path.exists(dataset_folder):
        st.sidebar.text("Memproses dataset...")
        mean_face, eigenfaces, weights, labels, image_paths = process_dataset(dataset_folder)
        
        if mean_face is not None:
            dist, label, match_img_path = recognize_face(test_img, mean_face, eigenfaces, weights, labels, image_paths)
            
            end_time = time.time()
            exec_time = round(end_time - start_time, 2)
            
            with col2:
                st.subheader("Closest Result")
                st.write(f"**Nilai Jarak Euclidean Asli:** {dist:.2f}")
                threshold = 15000.0 
                
                if dist < threshold:
                    match_img = cv2.imread(match_img_path)
                    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                    st.image(match_img_rgb, use_container_width=True)
                    st.success(f"Result: {label}")
                else:
                    st.error("Tidak terdapat citra wajah yang mirip.")
                    
            st.sidebar.success(f"Execution time: {exec_time}s")
        else:
            st.sidebar.error("Dataset folder kosong atau tidak valid.")
    else:
        st.sidebar.error("Folder dataset tidak ditemukan.")