import cv2
import time
from config import WEBCAM_INDEX
from face_engine import FaceRecognitionEngine
from attendance import AttendanceLogger

def main():
    engine = FaceRecognitionEngine()
    logger = AttendanceLogger()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'accéder à la webcam.")
        return

    prev_time = time.time()
    print("[INFO] Lancement du flux vidéo OpenCV avec temporisation...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Traitement IA
        locations, names, confidences = engine.process_frame(frame)

        # Réinitialise le compte à rebours de 10s pour ceux qui ne sont plus dans l'image
        logger.clean_absent_persons(names)

        total_faces = len(locations)

        for (top, right, bottom, left), name, conf in zip(locations, names, confidences):
            # Traitement de la temporisation et obtention du statut
            status_text = logger.process_person(name, conf)

            # Définition du libellé d'affichage
            if name != "Inconnu":
                label = f"{status_text} ({conf:.1f}%)"
                color = (0, 255, 0)
            else:
                label = "Inconnu"
                color = (0, 0, 255)

            # Dessin du cadre principal
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Bandeau d'en-tête en haut du cadre
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            banner_width = max(right - left, text_size[0] + 12)
            cv2.rectangle(frame, (left, top - 35), (left + banner_width, top), color, cv2.FILLED)

            # Texte au-dessus du cadre
            cv2.putText(frame, label, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Calcul des FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Affichage des métriques générales
        cv2.putText(frame, f"Personnes detectees: {total_faces}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Système de Reconnaissance Faciale & Pointage", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()