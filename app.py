import streamlit as st
import cv2
import os
import pickle
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

# --- CONFIGURATION GLOBALE ---
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "registre_presence.csv"
THRESHOLD = 0.45  # Seuil strict pour éviter les faux positifs

st.set_page_config(
    page_title="Système de Reconnaissance Faciale & Pointage",
    page_icon="📸",
    layout="wide"
)

# --- FONCTIONS DU MOTEUR IA & GESTION DES DONNÉES ---
def load_encodings():
    """Charge les embeddings depuis le fichier cache ou le dataset."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    
    known_encodings, known_names = [], []
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        return known_encodings, known_names

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name.capitalize())

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    return known_encodings, known_names

def save_encodings(known_encodings, known_names):
    """Sauvegarde les encodages mis à jour."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

def distance_to_confidence(distance):
    """Convertit la distance euclidienne en pourcentage de confiance."""
    if distance > THRESHOLD:
        return max(0.0, (1.0 - distance) * 50)
    else:
        return min(100.0, (1.0 - (distance / (THRESHOLD * 2.0))) * 100)

def log_attendance(name, confidence):
    """Enregistre le pointage dans le fichier CSV."""
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Nom", "Date", "Heure", "Confiance"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    # Évite les doublons à moins de 30 minutes
    already_logged = False
    if not df.empty:
        user_logs = df[(df["Nom"] == name) & (df["Date"] == today)]
        if not user_logs.empty:
            last_time = user_logs.iloc[-1]["Heure"]
            last_dt = datetime.strptime(f"{today} {last_time}", "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_dt).total_seconds() < 1800:
                already_logged = True

    if not already_logged:
        new_row = pd.DataFrame([[name, today, now_time, f"{confidence:.1f}%"]], columns=["Nom", "Date", "Heure", "Confiance"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# --- INTERFACE UTILISATEUR STREAMLIT ---
st.title("🧠 Système Intelligent de Reconnaissance Faciale & Pointage")
st.markdown("Projet Deep Learning | Espace vectoriel 128D & Affichage des Landmarks en temps réel.")

tabs = st.tabs(["🎥 Reconnaissance en Direct", "➕ Enregistrement de Personne", "📊 Registre & Vérification"])

# --- ONGLET 1 : RECONNAISSANCE EN DIRECT AVEC LANDMARKS ---
with tabs[0]:
    st.subheader("Flux Webcam & Extraction des Points de Repère (Landmarks)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        run_camera = st.checkbox("Activer la caméra", value=False)
        camera_placeholder = st.empty()
    
    with col2:
        st.info("💡 **Détails techniques :**\n- Les points bleus affichent les **68 landmarks** (géométrie faciale).\n- Le cadre vert/rouge s'appuie sur l'embedding 128D et le modèle ResNet.")
        log_placeholder = st.empty()

    known_encodings, known_names = load_encodings()

    if run_camera:
        cap = cv2.VideoCapture(0)
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Impossible d'accéder à la webcam.")
                break

            # Optimisation par redimensionnement
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Extraction des 68 points repères (Landmarks)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

            scale_up = 4  # Inverse de 0.25

            # Dessin des 68 points sur l'image
            for landmarks in face_landmarks_list:
                for feature, points in landmarks.items():
                    for pt in points:
                        pt_scaled = (int(pt[0] * scale_up), int(pt[1] * scale_up))
                        cv2.circle(frame, pt_scaled, 2, (255, 0, 0), -1)  # Points bleus

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = top * scale_up, right * scale_up, bottom * scale_up, left * scale_up
                
                name = "Inconnu"
                confidence = 0.0

                if len(known_encodings) > 0:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(distances)
                    best_distance = distances[best_match_index]

                    if best_distance < THRESHOLD:
                        name = known_names[best_match_index]
                        log_attendance(name, confidence)
                    
                    confidence = distance_to_confidence(best_distance)

                color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                label = f"{name} ({confidence:.1f}%)"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Conversion pour affichage Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
        cap.release()

# --- ONGLET 2 : ENREGISTREMENT D'UNE NOUVELLE PERSONNE ---
with tabs[1]:
    st.subheader("Ajouter une nouvelle personne au Dataset")
    
    new_name = st.text_input("Prénom / Nom de la personne :")
    uploaded_files = st.file_uploader("Importer des photos (3 à 5 recommandées)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if st.button("Enregistrer et Mettre à jour l'IA"):
        if new_name and uploaded_files:
            person_dir = os.path.join(DATASET_DIR, new_name.lower())
            os.makedirs(person_dir, exist_ok=True)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                bytes_data = uploaded_file.read()
                img_path = os.path.join(person_dir, f"photo_{idx+1}.jpg")
                with open(img_path, "wb") as f:
                    f.write(bytes_data)

            # Recalcul des encodages
            if os.path.exists(ENCODINGS_FILE):
                os.remove(ENCODINGS_FILE)
            load_encodings()
            
            st.success(f"'{new_name.capitalize()}' a été ajouté avec succès et le modèle 128D a été mis à jour !")
        else:
            st.warning("Veuillez entrer un nom et fournir au moins une photo.")

# --- ONGLET 3 : REGISTRE DE PRÉSENCE & VÉRIFICATION ---
with tabs[2]:
    st.subheader("Consultation du Registre & Vérification d'Existence")
    
    if os.path.exists(ATTENDANCE_FILE):
        df_att = pd.read_csv(ATTENDANCE_FILE)
        
        # Filtre de recherche par nom
        search_query = st.text_input("🔍 Vérifier si une personne est enregistrée / présente :")
        if search_query:
            filtered_df = df_att[df_att["Nom"].str.contains(search_query, case=False, na=False)]
            if not filtered_df.empty:
                st.success(f"Trouvé ! {search_query.capitalize()} a pointé aux dates suivantes :")
                st.dataframe(filtered_df)
            else:
                st.warning(f"Aucun historique de présence trouvé pour '{search_query}'.")
        
        st.markdown("### 📋 Historique global des présences")
        st.dataframe(df_att, use_container_width=True)
        
        # Bouton de téléchargement CSV
        csv = df_att.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger le registre en CSV",
            data=csv,
            file_name='registre_presence.csv',
            mime='text/csv',
        )
    else:
        st.info("Aucun registre de présence pour le moment.")