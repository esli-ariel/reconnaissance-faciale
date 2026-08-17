import cv2
import time
from config import WEBCAM_INDEX
from face_engine import FaceRecognitionEngine
from attendance import AttendanceLogger
from PyQt5.QtWidgets import QApplication, QInputDialog

def ask_name_popup():
    """
    Boîte de dialogue native ultrastable sur macOS avec PyQt5.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    text, ok = QInputDialog.getText(None, "Ajout d'une personne", "Entrez le prénom / nom :")
    if ok and text:
        return text.strip()
    return None
    """
    Affiche une fenêtre graphique pop-up pour saisir le nom.
    """
    # Création d'une fenêtre cachée Tkinter (nécessaire pour afficher le dialogue)
    root = tk.Tk()
    root.withdraw()

    # Empêche le plantage NSException sur macOS
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.deiconify()
    root.withdraw()

    # Maintient la fenêtre pop-up au premier plan
    root.attributes('-topmost', True)
    
    # Boîte de dialogue de saisie
    person_name = simpledialog.askstring(
        title="Ajout d'une personne",
        prompt="Entrez le prénom / nom de la personne à ajouter :"
    )
    
    root.destroy()
    return person_name.strip() if person_name else None

def main():
    engine = FaceRecognitionEngine()
    logger = AttendanceLogger()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'accéder à la webcam.")
        return

    prev_time = time.time()
    print("[INFO] Lancement du flux vidéo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Traitement IA
        locations, names, confidences = engine.process_frame(frame)
        logger.clean_absent_persons(names)

        total_faces = len(locations)

        for (top, right, bottom, left), name, conf in zip(locations, names, confidences):
            status_text = logger.process_person(name, conf)

            if name != "Inconnu":
                label = f"{status_text} ({conf:.1f}%)"
                color = (0, 255, 0)
            else:
                label = "Inconnu"
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            banner_width = max(right - left, text_size[0] + 12)
            cv2.rectangle(frame, (left, top - 35), (left + banner_width, top), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Affichage FPS & Instructions
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"Personnes: {total_faces} | FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "[A] Ajouter personne | [Q] Quitter", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Système de Reconnaissance Faciale & Pointage", frame)

        key = cv2.waitKey(1) & 0xFF

        # --- TOUCHE 'A' : SAISIE DANS UNE FENÊTRE POP-UP ---
        if key == ord('a'):
            # Ouvre la boîte de dialogue graphique
            new_name = ask_name_popup()
            
            if new_name:
                captured_frames = []
                
                # Compte à rebours visuel de 3 secondes sur le flux webcam
                for countdown in range(3, 0, -1):
                    temp_ret, temp_frame = cap.read()
                    if temp_ret:
                        cv2.putText(temp_frame, f"Attention ! Capture dans {countdown}...", (50, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                        cv2.imshow("Système de Reconnaissance Faciale & Pointage", temp_frame)
                        cv2.waitKey(1000)

                # Prise automatique de 3 photos
                for photo_num in range(1, 4):
                    temp_ret, temp_frame = cap.read()
                    if temp_ret:
                        captured_frames.append(temp_frame.copy())
                        cv2.putText(temp_frame, f"PHOTO {photo_num}/3 CAPTUREE !", (50, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        cv2.imshow("Système de Reconnaissance Faciale & Pointage", temp_frame)
                        cv2.waitKey(500)

                # Mise à jour de l'IA en arrière-plan
                engine.add_new_person_and_reload(new_name, captured_frames)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()