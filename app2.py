import streamlit as st
import cv2
import os
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime
from insightface.app import FaceAnalysis

# --- CONFIGURATION GLOBALE ---
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings_insightface.pkl"
ATTENDANCE_FILE = "registre_presence.csv"
SIMILARITY_THRESHOLD = 0.40

# Mot de passe administrateur (À modifier selon vos besoins)
ADMIN_PASSWORD = "admin123"

st.set_page_config(
    page_title="Système IA - InsightFace (512D)",
    page_icon="⚡",
    layout="wide"
)

# Initialisation de l'état de session pour l'authentification admin
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

@st.cache_resource
def init_insightface():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

face_app = init_insightface()

def load_encodings():
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
                img = cv2.imread(filepath)
                if img is not None:
                    faces = face_app.get(img)
                    if len(faces) > 0:
                        known_encodings.append(faces[0].normed_embedding)
                        known_names.append(person_name.capitalize())

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    return known_encodings, known_names

def similarity_to_confidence(similarity):
    if similarity < SIMILARITY_THRESHOLD:
        return max(0.0, (similarity / SIMILARITY_THRESHOLD) * 50.0)
    else:
        norm = (similarity - SIMILARITY_THRESHOLD) / (1.0 - SIMILARITY_THRESHOLD)
        return min(100.0, 50.0 + (norm * 50.0))

def log_attendance(name, confidence):
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Nom", "Date", "Heure", "Confiance"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    already_logged = False
    if not df.empty:
        # On vérifie si la personne a déjà un pointage aujourd'hui (peu importe l'heure)
        user_logs = df[(df["Nom"] == name) & (df["Date"] == today)]
        if not user_logs.empty:
            already_logged = True

    # Si elle n'a pas encore pointé aujourd'hui, on enregistre
    if not already_logged:
        new_row = pd.DataFrame([[name, today, now_time, f"{confidence:.1f}%"]], columns=["Nom", "Date", "Heure", "Confiance"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True
    
    return False # Déjà pointé aujourd'hui, on ignore
# --- INTERFACE UTILISATEUR ---
st.title("⚡ Système de Reconnaissance Faciale & Pointage Intelligent")
st.markdown("Architecture : InsightFace (ArcFace 512D)  | Similarité Cosinus & Visualisation des Embeddings.")

# Barre latérale pour la gestion de l'authentification admin
with st.sidebar:
    st.header("🔐 Espace Administrateur")
    if not st.session_state.is_admin:
        pwd_input = st.text_input("Mot de passe admin", type="password")
        if st.button("Se connecter"):
            if pwd_input == ADMIN_PASSWORD:
                st.session_state.is_admin = True
                st.success("Connexion réussie !")
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
    else:
        st.success("Mode Admin Actif 🟢")
        if st.button("Se déconnecter"):
            st.session_state.is_admin = False
            st.rerun()

tabs = st.tabs(["🎥 Caméra en Direct", "➕ Enregistrement", "📊 Registre & Vérification", "🔧 Visualisation du Modèle"])

# --- ONGLET 0 : CAMERA ---
with tabs[0]:
    st.subheader("Flux Vidéo & Pointage en Direct")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        fps_metric = st.empty()
    with metric_col2:
        count_metric = st.empty()

    col1, col2 = st.columns([2, 1])
    with col1:
        run_camera = st.checkbox("Activer la webcam", value=False)
        camera_placeholder = st.empty()
    
    with col2:
        st.info("💡 **Règles :**\n- Restez **5 secondes** face caméra pour valider.\n- Utilisation de la **Similarité Cosinus** (comparaison hypersphérique).\n- Affichage en temps réel du vecteur 512D. ")
        status_box = st.empty()
        embedding_box = st.empty()

    known_encodings, known_names = load_encodings()
    presence_timers = {}

    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Impossible d'accéder à la webcam.")
        else:
            prev_frame_time = 0
            while run_camera:
                new_frame_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
                prev_frame_time = new_frame_time

                faces = face_app.get(frame)
                num_persons = len(faces)
                
                fps_metric.metric("⚡ Fréquence (FPS)", f"{fps:.1f} FPS")
                count_metric.metric("👥 Personnes détectées", f"{num_persons}")

                current_time = time.time()
                detected_names_in_frame = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    kps = face.kps.astype(int)
                    test_embedding = face.normed_embedding

                    for pt in kps:
                        cv2.circle(frame, tuple(pt), 3, (255, 0, 0), -1)

                    name = "Inconnu"
                    similarity = 0.0

                    if len(known_encodings) > 0:
                        similarities = np.dot(known_encodings, test_embedding)
                        best_idx = np.argmax(similarities)
                        similarity = similarities[best_idx]

                        if similarity > SIMILARITY_THRESHOLD:
                            name = known_names[best_idx]

                    confidence = similarity_to_confidence(similarity)
                    detected_names_in_frame.append(name)

                    with embedding_box.container():
                        st.text(f"Dernier Embedding ({name}) :")
                        st.json(test_embedding[:5].tolist())
                        st.caption(f"Score Cosinus : {similarity:.4f}")

                    if name != "Inconnu":
                        if name not in presence_timers:
                            presence_timers[name] = current_time
                        
                        elapsed_time = current_time - presence_timers[name]
                        time_left = max(0, 5 - int(elapsed_time))

                        if elapsed_time >= 5:
                            success = log_attendance(name, confidence)
                            if success:
                                status_box.success(f"✅ Pointage validé pour {name} !")
                            label = f"{name} - POINTÉ ({confidence:.1f}%)"
                            color = (0, 255, 0)
                        else:
                            label = f"{name} - Maintien ({time_left}s)"
                            color = (0, 165, 255)
                    else:
                        label = "Inconnu"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                for p_name in list(presence_timers.keys()):
                    if p_name not in detected_names_in_frame:
                        del presence_timers[p_name]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
            cap.release()

# --- ONGLET 1 : ENREGISTREMENT (Sécurisé) ---
with tabs[1]:
    if not st.session_state.is_admin:
        st.warning("🔒 Zone sécurisée. Veuillez vous connecter via le panneau latéral (Admin) pour enregistrer de nouvelles personnes.")
    else:
        st.subheader("➕ Enregistrer une nouvelle personne")
        reg_mode = st.radio("Méthode d'enregistrement :", ["Capturer avec la webcam en direct", "Importer des photos"])
        new_name = st.text_input("Nom complet de la personne :")

        if reg_mode == "Importer des photos":
            uploaded_files = st.file_uploader("Importer des photos de référence", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            if st.button("Enregistrer & Mettre à jour le modèle 512D"):
                if new_name and uploaded_files:
                    person_dir = os.path.join(DATASET_DIR, new_name.lower())
                    os.makedirs(person_dir, exist_ok=True)
                    
                    for idx, file in enumerate(uploaded_files):
                        with open(os.path.join(person_dir, f"photo_{idx+1}.jpg"), "wb") as f:
                            f.write(file.read())

                    if os.path.exists(ENCODINGS_FILE):
                        os.remove(ENCODINGS_FILE)
                    load_encodings()
                    st.success(f"'{new_name.capitalize()}' enregistré avec succès !")
                else:
                    st.warning("Veuillez renseigner un nom et charger au moins une image.")
        else:
            if st.button("Lancer la capture en direct"):
                if not new_name:
                    st.warning("Veuillez d'abord entrer un nom valide.")
                else:
                    person_dir = os.path.join(DATASET_DIR, new_name.lower())
                    os.makedirs(person_dir, exist_ok=True)
                    
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Impossible d'accéder à la webcam.")
                    else:
                        capture_placeholder = st.empty()
                        for countdown in range(3, 0, -1):
                            ret, temp_frame = cap.read()
                            if ret:
                                cv2.putText(temp_frame, f"Preparation... {countdown}", (50, 100), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                                capture_placeholder.image(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                                time.sleep(1)

                        for i in range(1, 4):
                            ret, temp_frame = cap.read()
                            if ret:
                                cv2.imwrite(os.path.join(person_dir, f"capture_{i}.jpg"), temp_frame)
                                capture_placeholder.image(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                                time.sleep(0.8)
                        cap.release()
                        
                        if os.path.exists(ENCODINGS_FILE):
                            os.remove(ENCODINGS_FILE)
                        load_encodings()
                        st.success(f"Captures terminées pour '{new_name.capitalize()}'.")

# --- ONGLET 2 : REGISTRE & VERIFICATION (Sécurisé) ---
with tabs[2]:
    if not st.session_state.is_admin:
        st.warning("🔒 Zone sécurisée. Veuillez vous connecter via le panneau latéral (Admin) pour consulter le registre des présences.")
    else:
        st.subheader("📊 Tableau de bord de Contrôle")
        
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Heure'] = pd.to_datetime(df['Heure'], format='%H:%M:%S').dt.time
            
            # --- FILTRES DE CONTRÔLE ---
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_date = st.date_input("Filtrer par date", datetime.now().date())
            with col_f2:
                heure_limite = st.time_input("Heure limite d'arrivée (pour contrôle)", datetime.strptime("09:00:00", "%H:%M:%S").time())

            # Filtrage
            df_filtered = df[df['Date'].dt.date == selected_date].copy()

            # --- KPI DASHBOARD ---
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total présences", len(df_filtered))
            
            # Calcul des retards
            retardataires = df_filtered[df_filtered['Heure'] > heure_limite]
            kpi2.metric("Retardataires", len(retardataires), delta=f"-{len(retardataires)}", delta_color="inverse")
            kpi3.metric("Taux de ponctualité", f"{((len(df_filtered)-len(retardataires))/len(df_filtered)*100):.1f}%" if len(df_filtered)>0 else "0%")

            # --- VISUALISATION ---
            st.divider()
            
            # Graphique des arrivées par heure
            if not df_filtered.empty:
                df_filtered['Heure_H'] = pd.to_datetime(df_filtered['Heure'], format='%H:%M:%S').dt.hour
                chart_data = df_filtered.groupby('Heure_H').size()
                st.bar_chart(chart_data, use_container_width=True)
            
            # --- TABLEAU DÉTAILLÉ AVEC CONTRÔLE ---
            st.subheader("Détail des entrées")
            
            # Coloration conditionnelle pour les retards
            def color_row(row):
                if row['Heure'] > heure_limite:
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_filtered.style.apply(color_row, axis=1),
                use_container_width=True
            )
            
            # Actions de contrôle
            if st.button("Exporter le rapport du jour"):
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger CSV", csv, f"rapport_{selected_date}.csv", "text/csv")
        else:
            st.info("Aucune donnée de présence disponible.")
   

# --- ONGLET 3 : ADMINISTRATION DU MODELE (Sécurisé) ---
with tabs[3]:
    if not st.session_state.is_admin:
        st.warning("🔒 Zone réservée à l'administrateur.")
    else:
        st.subheader("🔧 Inspection des Embeddings & Système")
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            
            st.success(f"Modèle actif : {len(data['names'])} visages en mémoire.")
            
            df_debug = pd.DataFrame({
                "Nom": data["names"],
                "Dimensions": [str(e.shape) for e in data["encodings"]],
                "Aperçu (5 premières valeurs)": [str(e[:5].tolist()) for e in data["encodings"]]
            })
            st.dataframe(df_debug, use_container_width=True)
            
            if st.button("🗑️ Réinitialiser le modèle (Supprimer le fichier Pickle)"):
                os.remove(ENCODINGS_FILE)
                st.warning("Fichier supprimé. Rechargez l'application.")
                st.rerun()
        else:
            st.error("Fichier d'encodage introuvable.")