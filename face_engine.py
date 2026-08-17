import os
import cv2
import pickle
import face_recognition
import numpy as np
from config import DATASET_DIR, ENCODINGS_FILE, RECOGNITION_THRESHOLD, FRAME_RESIZE_FACTOR

class FaceRecognitionEngine:
    def __init__(self, dataset_dir=DATASET_DIR, encodings_file=ENCODINGS_FILE, threshold=RECOGNITION_THRESHOLD):
        self.dataset_dir = dataset_dir
        self.encodings_file = encodings_file
        self.threshold = threshold
        self.known_encodings = []
        self.known_names = []
        self.load_dataset()

    def load_dataset(self, force_reencode=False):
        """
        Charge les encodages depuis le fichier de cache s'il existe.
        Sinon, calcule les encodages du dataset et les sauvegarde.
        """
        # Si le fichier de cache existe et qu'on ne force pas le rechargement
        if os.path.exists(self.encodings_file) and not force_reencode:
            print("[INFO] Chargement instantané des encodages depuis le cache...")
            with open(self.encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            print(f"[INFO] {len(self.known_encodings)} visage(s) chargé(s) depuis le cache pour {len(set(self.known_names))} personne(s).")
            return

        # Sinon, recalcul complet à partir des images
        print("[INFO] Calcul des encodages du dataset (cela peut prendre quelques secondes)...")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"[AVERTISSEMENT] Le dossier {self.dataset_dir} a été créé.")
            return

        self.known_encodings = []
        self.known_names = []

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

        # Sauvegarde dans le fichier cache (.pkl)
        data = {"encodings": self.known_encodings, "names": self.known_names}
        with open(self.encodings_file, "wb") as f:
            pickle.dump(data, f)

        print(f"[INFO] Encodages sauvegardés dans {self.encodings_file}.")
        print(f"[INFO] {len(self.known_encodings)} visage(s) chargé(s) pour {len(set(self.known_names))} personne(s).")

    def distance_to_confidence(self, distance):
        if distance > self.threshold:
            range_diff = (1.0 - self.threshold)
            linear_val = (1.0 - distance) / (range_diff * 2.0)
            return max(0.0, linear_val * 100)
        else:
            linear_val = 1.0 - (distance / (self.threshold * 2.0))
            return min(100.0, linear_val * 100)

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        confidences = []

        for face_encoding in face_encodings:
            name = "Inconnu"
            confidence = 0.0

            if len(self.known_encodings) > 0:
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]

                if best_distance < self.threshold:
                    name = self.known_names[best_match_index]
                
                confidence = self.distance_to_confidence(best_distance)

            face_names.append(name)
            confidences.append(confidence)
            
        scale_up = int(1 / FRAME_RESIZE_FACTOR)
        scaled_locations = [
            (top * scale_up, right * scale_up, bottom * scale_up, left * scale_up)
            for top, right, bottom, left in face_locations
        ]

        return scaled_locations, face_names, confidences