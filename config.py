import os

# Chemins des dossiers et fichiers
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "registre_presence.csv"

# Paramètres du modèle et de l'IA
RECOGNITION_THRESHOLD = 0.6   # Seuil de tolérance (distance euclidienne)
FRAME_RESIZE_FACTOR = 0.25    # Réduction à 25% pour accélérer les calculs

# Paramètres de temporisation (en secondes)
PRESENCE_HOLD_TIME = 5       # Rester 5s continue devant la caméra pour valider
COOLDOWN_TIME = 1800          # 30 minutes (30 * 60s) avant ré-enregistrement

# Paramètres de la webcam
WEBCAM_INDEX = 0