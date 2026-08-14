import csv
import os
import time
from datetime import datetime
from config import ATTENDANCE_FILE, PRESENCE_HOLD_TIME, COOLDOWN_TIME

class AttendanceLogger:
    def __init__(self, filepath=ATTENDANCE_FILE):
        self.filepath = filepath
        self.detection_start_time = {}  # {person_name: timestamp_apparition}
        self.last_registered_time = {}  # {person_name: timestamp_dernier_enregistrement}
        self._init_file()

    def _init_file(self):
        """Initialise le fichier CSV s'il n'existe pas."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Nom", "Date", "Heure", "Confiance"])

    def process_person(self, person_name: str, confidence_score: float):
        """
        Gère la double temporisation :
        1. 30 min entre deux enregistrements.
        2. 10s de présence continue dans le cadre.
        Retourne l'état / message à afficher.
        """
        if person_name == "Inconnu":
            return "Inconnu"

        now = time.time()

        # 1. Vérification du Cooldown de 30 minutes (1800 secondes)
        if person_name in self.last_registered_time:
            elapsed_since_reg = now - self.last_registered_time[person_name]
            if elapsed_since_reg < COOLDOWN_TIME:
                rem_min = int((COOLDOWN_TIME - elapsed_since_reg) // 60)
                return f"{person_name} (Enregistre - Re-check dans {rem_min}m)"

        # 2. Gestion de la temporisation de 10 secondes dans le cadre
        if person_name not in self.detection_start_time:
            self.detection_start_time[person_name] = now

        time_in_frame = now - self.detection_start_time[person_name]

        if time_in_frame >= PRESENCE_HOLD_TIME:
            # Enregistrement effectif
            self._write_to_csv(person_name, confidence_score)
            self.last_registered_time[person_name] = now
            del self.detection_start_time[person_name]
            return f"{person_name} - VALIDE !"
        else:
            sec_left = int(PRESENCE_HOLD_TIME - time_in_frame)
            return f"{person_name} (Validation dans {sec_left}s)"

    def clean_absent_persons(self, currently_visible_names):
        """
        Réinitialise le compteur de 10s pour les personnes qui quittent le cadre.
        """
        tracked_names = list(self.detection_start_time.keys())
        for name in tracked_names:
            if name not in currently_visible_names:
                del self.detection_start_time[name]

    def _write_to_csv(self, person_name: str, confidence_score: float):
        """Écriture dans le fichier CSV."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([person_name, today_str, time_str, f"{confidence_score:.1f}%"])
        print(f"[POINTAGE VALIDE] {person_name} ({confidence_score:.1f}%) a {time_str}")