import os
import cv2
import face_recognition
import numpy as np
from config import DATASET_DIR, RECOGNITION_THRESHOLD, FRAME_RESIZE_FACTOR

class FaceRecognitionEngine:
    def __init__(self, dataset_dir=DATASET_DIR, threshold=RECOGNITION_THRESHOLD):
        self.dataset_dir = dataset_dir
        self.threshold = threshold
        self.known_encodings = []
        self.known_names = []
        self.load_dataset()

    def load_dataset(self):
        """Parcourt le dossier dataset et calcule les embeddings des visages connus."""
        print("[INFO] Chargement du dataset...")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"[AVERTISSEMENT] Le dossier {self.dataset_dir} a été créé. Ajoutez-y des dossiers d'images.")
            return

        for person_name in os.listdir(self.dataset_dir):
            person_dir = os.path.join(self.dataset_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(person_name.capitalize())

        print(f"[INFO] {len(self.known_encodings)} visage(s) chargé(s) pour {len(set(self.known_names))} personne(s).")

    def distance_to_confidence(self, distance):
        """
        Convertit la distance euclidienne d'embedding en pourcentage de similarité (%).
        """
        if distance > self.threshold:
            # Si la distance est supérieure au seuil, la confiance chute fortement
            range_diff = (1.0 - self.threshold)
            linear_val = (1.0 - distance) / (range_diff * 2.0)
            return max(0.0, linear_val * 100)
        else:
            # Conversion linéaire où 0 distance = 100% et le seuil = ~70%
            linear_val = 1.0 - (distance / (self.threshold * 2.0))
            return min(100.0, linear_val * 100)

    def process_frame(self, frame):
        """
        Détecte et reconnaît les visages dans l'image.
        Retourne : localisations, noms, et scores de confiance (%).
        """
        # Redimensionner l'image pour optimiser la vitesse de calcul
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Détection des localisations et calcul des embeddings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        confidences = []

        for face_encoding in face_encodings:
            name = "Inconnu"
            confidence = 0.0

            if len(self.known_encodings) > 0:
                # Calcul des distances avec tous les visages connus
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]

                if best_distance < self.threshold:
                    name = self.known_names[best_match_index]
                
                # Calcul du pourcentage de confiance basé sur la meilleure distance
                confidence = self.distance_to_confidence(best_distance)

            face_names.append(name)
            confidences.append(confidence)

        # Repositionnement des coordonnées à l'échelle originale de l'image (x4 si FRAME_RESIZE_FACTOR=0.25)
        scale_up = int(1 / FRAME_RESIZE_FACTOR)
        scaled_locations = [
            (top * scale_up, right * scale_up, bottom * scale_up, left * scale_up)
            for top, right, bottom, left in face_locations
        ]

        # Retourne exactement les 3 éléments attendus par main.py
        return scaled_locations, face_names, confidences