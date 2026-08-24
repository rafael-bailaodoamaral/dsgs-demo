import csv
import json
import math
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import cv2
import mediapipe as mp
import numpy as np

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


CAMERA_INDEX = 0
DATA_FILE = "dsgs_landmarks.csv"
MODEL_FILE = "dsgs_model.json"
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.55
STABLE_FRAMES = 8
PREDICTION_THRESHOLD = 0.72


class DSGSTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DSGS Gebärdensprach-Übersetzer")
        self.root.geometry("1250x850")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.running = False
        self.capture = None
        self.last_prediction = ""
        self.current_prediction = ""
        self.stable_prediction = ""
        self.stable_count = 0
        self.last_saved_prediction = ""
        self.recording_label = None
        self.recording_samples = []
        self.dataset_x = []
        self.dataset_y = []
        self.classifier = None
        self.scaler = None
        self.last_frame = None

        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        self.build_ui()
        self.load_dataset()
        self.load_model()

    def build_ui(self):
        self.root.configure(bg="#151515")

        top = tk.Frame(self.root, bg="#151515")
        top.pack(fill="x", padx=15, pady=12)

        title = tk.Label(
            top,
            text="DSGS Gebärdensprach-Übersetzer",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#151515"
        )
        title.pack(side="left")

        self.status = tk.Label(
            top,
            text="Kamera gestoppt",
            font=("Segoe UI", 11),
            fg="#bbbbbb",
            bg="#151515"
        )
        self.status.pack(side="right")

        center = tk.Frame(self.root, bg="#151515")
        center.pack(fill="both", expand=True, padx=15, pady=5)

        video_frame = tk.Frame(center, bg="#222222")
        video_frame.pack(side="left", fill="both", expand=True)

        self.video = tk.Label(
            video_frame,
            text="Kamera starten",
            font=("Segoe UI", 20),
            fg="white",
            bg="#222222"
        )
        self.video.pack(fill="both", expand=True)

        right = tk.Frame(center, width=330, bg="#202020")
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        tk.Label(
            right,
            text="Erkennung",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg="#202020"
        ).pack(pady=(20, 5))

        self.result = tk.Label(
            right,
            text="—",
            font=("Segoe UI", 30, "bold"),
            fg="white",
            bg="#202020",
            wraplength=290
        )
        self.result.pack(pady=15)

        self.confidence = tk.Label(
            right,
            text="Konfidenz: —",
            font=("Segoe UI", 12),
            fg="#bbbbbb",
            bg="#202020"
        )
        self.confidence.pack()

        self.hand_info = tk.Label(
            right,
            text="Hände: 0",
            font=("Segoe UI", 11),
            fg="#bbbbbb",
            bg="#202020"
        )
        self.hand_info.pack(pady=5)

        ttk.Separator(right).pack(fill="x", padx=20, pady=20)

        self.start_button = tk.Button(
            right,
            text="Kamera starten",
            command=self.toggle_camera,
            font=("Segoe UI", 12, "bold"),
            bg="#eeeeee",
            fg="#111111",
            relief="flat",
            padx=10,
            pady=10
        )
        self.start_button.pack(fill="x", padx=25, pady=5)

        tk.Button(
            right,
            text="Gebärde trainieren",
            command=self.start_recording,
            font=("Segoe UI", 12),
            bg="#eeeeee",
            fg="#111111",
            relief="flat",
            padx=10,
            pady=10
        ).pack(fill="x", padx=25, pady=5)

        tk.Button(
            right,
            text="Modell trainieren",
            command=self.train_model,
            font=("Segoe UI", 12),
            bg="#eeeeee",
            fg="#111111",
            relief="flat",
            padx=10,
            pady=10
        ).pack(fill="x", padx=25, pady=5)

        tk.Button(
            right,
            text="Datensatz anzeigen",
            command=self.show_dataset_info,
            font=("Segoe UI", 12),
            bg="#eeeeee",
            fg="#111111",
            relief="flat",
            padx=10,
            pady=10
        ).pack(fill="x", padx=25, pady=5)

        ttk.Separator(right).pack(fill="x", padx=20, pady=20)

        self.recording_status = tk.Label(
            right,
            text="Kein Training aktiv",
            font=("Segoe UI", 10),
            fg="#bbbbbb",
            bg="#202020",
            wraplength=280
        )
        self.recording_status.pack(padx=20, pady=5)

        info = tk.Label(
            right,
            text="Für echte DSGS-Übersetzung müssen die Gebärden mit deinen Kamerabildern trainiert werden.",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#202020",
            wraplength=280,
            justify="left"
        )
        info.pack(padx=20, pady=20)

        bottom = tk.Frame(self.root, bg="#151515")
        bottom.pack(fill="x", padx=15, pady=10)

        self.translation = tk.Label(
            bottom,
            text="Übersetzung: —",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#151515",
            anchor="w"
        )
        self.translation.pack(fill="x")

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.capture.isOpened():
            messagebox.showerror(
                "Kamera",
                "Die Kamera konnte nicht geöffnet werden."
            )
            return

        self.running = True
        self.start_button.config(text="Kamera stoppen")
        self.status.config(text="Kamera läuft")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None
        self.start_button.config(text="Kamera starten")
        self.status.config(text="Kamera gestoppt")

    def normalize_points(self, points):
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0:
            return []

        origin = points[0].copy()
        points = points - origin

        scale = np.max(np.linalg.norm(points[:, :2], axis=1))
        if scale < 1e-6:
            scale = 1.0

        points = points / scale
        return points.flatten().tolist()

    def hand_features(self, hand_landmarks):
        points = []
        for p in hand_landmarks.landmark:
            points.append([p.x, p.y, p.z])
        return self.normalize_points(points)

    def pose_features(self, pose_landmarks):
        selected = [11, 12, 13, 14, 15, 16, 23, 24]
        points = []
        for index in selected:
            p = pose_landmarks.landmark[index]
            points.append([p.x, p.y, p.z])
        return self.normalize_points(points)

    def face_features(self, face_landmarks):
        selected = [0, 13, 14, 61, 291, 78, 308, 159, 145, 386, 374]
        points = []
        for index in selected:
            p = face_landmarks.landmark[index]
            points.append([p.x, p.y, p.z])
        return self.normalize_points(points)

    def build_feature_vector(self, result_hands, result_pose, result_face):
        left = [0.0] * 63
        right = [0.0] * 63

        if result_hands.multi_hand_landmarks and result_hands.multi_handedness:
            for hand, handedness in zip(
                result_hands.multi_hand_landmarks,
                result_hands.multi_handedness
            ):
                label = handedness.classification[0].label
                values = self.hand_features(hand)
                if label == "Left":
                    left = values
                else:
                    right = values

        pose = [0.0] * 24
        if result_pose.pose_landmarks:
            pose = self.pose_features(result_pose.pose_landmarks)

        face = [0.0] * 33
        if result_face.multi_face_landmarks:
            face = self.face_features(result_face.multi_face_landmarks[0])

        return np.asarray(left + right + pose + face, dtype=np.float32)

    def simple_gesture(self, result_hands):
        if not result_hands.multi_hand_landmarks:
            return "—", 0.0

        hand = result_hands.multi_hand_landmarks[0]
        p = hand.landmark

        fingers = []

        fingers.append(p[8].y < p[6].y)
        fingers.append(p[12].y < p[10].y)
        fingers.append(p[16].y < p[14].y)
        fingers.append(p[20].y < p[18].y)

        thumb = abs(p[4].x - p[2].x) > abs(p[3].x - p[2].x) * 1.35

        count = sum(fingers)

        if count == 0 and not thumb:
            return "FAUST", 0.82

        if count == 0 and thumb:
            return "DAUMEN_HOCH", 0.84

        if count == 4 and thumb:
            return "OFFENE_HAND", 0.90

        if count == 2 and fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
            return "FRIEDEN", 0.86

        if count == 1 and fingers[0]:
            return "ZEIGEFINGER", 0.84

        if count == 3 and fingers[0] and fingers[1] and fingers[2]:
            return "DREI", 0.83

        if count == 4:
            return "VIER", 0.82

        return "HANDFORM", 0.55

    def update_frame(self):
        if not self.running or not self.capture:
            return

        ok, frame = self.capture.read()

        if not ok:
            self.root.after(20, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_result = self.hands.process(rgb)
        pose_result = self.pose.process(rgb)
        face_result = self.face.process(rgb)

        if hand_result.multi_hand_landmarks:
            for hand in hand_result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

        if pose_result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        if face_result.multi_face_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                face_result.multi_face_landmarks[0],
                self.mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None
            )

        features = self.build_feature_vector(
            hand_result,
            pose_result,
            face_result
        )

        prediction, confidence = self.predict(features)

        if prediction == "UNBEKANNT":
            fallback_prediction, fallback_confidence = self.simple_gesture(hand_result)
            if fallback_confidence > confidence:
                prediction = fallback_prediction
                confidence = fallback_confidence

        self.update_prediction(prediction, confidence)

        cv2.putText(
            frame,
            f"{prediction}  {confidence:.0%}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame_rgb, (900, 680))
        image = np.ascontiguousarray(image)

        try:
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(image)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.video.configure(image=tk_image, text="")
            self.video.image = tk_image
        except Exception:
            self.video.configure(text="Pillow fehlt")

        if self.recording_label:
            self.recording_samples.append(features.tolist())
            self.recording_status.config(
                text=f"Training: {self.recording_label}\n"
                     f"{len(self.recording_samples)} Samples"
            )

        self.root.after(15, self.update_frame)

    def update_prediction(self, prediction, confidence):
        if prediction == self.current_prediction:
            self.stable_count += 1
        else:
            self.current_prediction = prediction
            self.stable_count = 1

        if self.stable_count >= STABLE_FRAMES:
            self.stable_prediction = prediction
            self.last_prediction = prediction

        shown = self.stable_prediction or prediction
        self.result.config(text=shown)
        self.confidence.config(
            text=f"Konfidenz: {confidence:.0%}"
        )

        hands = 0
        self.hand_info.config(text=f"Hände: {hands}")

        if shown not in ("—", "UNBEKANNT", "HANDFORM"):
            self.translation.config(
                text=f"Übersetzung: {self.translate_label(shown)}"
            )
        else:
            self.translation.config(
                text="Übersetzung: —"
            )

    def translate_label(self, label):
        translations = {
            "FAUST": "Faust / Hand geschlossen",
            "DAUMEN_HOCH": "Daumen hoch",
            "OFFENE_HAND": "offene Hand",
            "FRIEDEN": "Zwei Finger",
            "ZEIGEFINGER": "Zeigefinger",
            "DREI": "Drei Finger",
            "VIER": "Vier Finger",
            "HANDFORM": "Handform erkannt"
        }
        return translations.get(label, label)

    def predict(self, features):
        if not self.classifier or self.scaler is None:
            return "UNBEKANNT", 0.0

        try:
            scaled = self.scaler.transform([features])
            probabilities = self.classifier.predict_proba(scaled)[0]
            index = int(np.argmax(probabilities))
            confidence = float(probabilities[index])
            label = str(self.classifier.classes_[index])

            if confidence < PREDICTION_THRESHOLD:
                return "UNBEKANNT", confidence

            return label, confidence
        except Exception:
            return "UNBEKANNT", 0.0

    def start_recording(self):
        if not self.running:
            messagebox.showinfo(
                "Kamera",
                "Starte zuerst die Kamera."
            )
            return

        label = simpledialog.askstring(
            "Gebärde trainieren",
            "Name der Gebärde:\n\n"
            "Für DSGS kannst du z. B. einen Gloss verwenden, "
            "wie er in deiner Annotation steht."
        )

        if not label:
            return

        self.recording_label = label.strip()
        self.recording_samples = []

        self.recording_status.config(
            text=f"Training: {self.recording_label}\n"
                 "Bewege die Hände langsam und deutlich.\n"
                 "Zum Beenden erneut klicken."
        )

        self.root.after(5000, self.finish_recording)

    def finish_recording(self):
        if not self.recording_label:
            return

        if len(self.recording_samples) < 15:
            messagebox.showwarning(
                "Training",
                "Zu wenige Samples. Bitte mindestens 15 Samples aufnehmen."
            )
            self.recording_label = None
            self.recording_samples = []
            self.recording_status.config(text="Kein Training aktiv")
            return

        for sample in self.recording_samples:
            self.dataset_x.append(sample)
            self.dataset_y.append(self.recording_label)

        self.save_dataset()

        label = self.recording_label
        count = len(self.recording_samples)

        self.recording_label = None
        self.recording_samples = []

        self.recording_status.config(
            text=f"{count} Samples für {label} gespeichert."
        )

        messagebox.showinfo(
            "Training",
            f"{count} Samples für '{label}' wurden gespeichert.\n\n"
            "Trainiere jetzt das Modell."
        )

    def save_dataset(self):
        if not self.dataset_x:
            return

        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + [f"f{i}" for i in range(len(self.dataset_x[0]))])
            for label, values in zip(self.dataset_y, self.dataset_x):
                writer.writerow([label] + values)

    def load_dataset(self):
        if not os.path.exists(DATA_FILE):
            return

        try:
            with open(DATA_FILE, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)

                for row in reader:
                    if len(row) < 2:
                        continue
                    self.dataset_y.append(row[0])
                    self.dataset_x.append([float(x) for x in row[1:]])
        except Exception:
            self.dataset_x = []
            self.dataset_y = []

    def train_model(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showerror(
                "scikit-learn fehlt",
                "Installiere scikit-learn mit:\n\n"
                "pip install scikit-learn"
            )
            return

        if len(self.dataset_x) < 30:
            messagebox.showwarning(
                "Modell",
                "Der Datensatz ist zu klein.\n"
                "Trainiere mindestens 2 Gebärden mit jeweils 15 bis 30 Samples."
            )
            return

        labels = sorted(set(self.dataset_y))

        if len(labels) < 2:
            messagebox.showwarning(
                "Modell",
                "Es werden mindestens zwei verschiedene Gebärden benötigt."
            )
            return

        x = np.asarray(self.dataset_x, dtype=np.float32)
        y = np.asarray(self.dataset_y)

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x)

        neighbors = min(7, len(x_scaled))
        self.classifier = KNeighborsClassifier(
            n_neighbors=neighbors,
            weights="distance"
        )
        self.classifier.fit(x_scaled, y)

        self.save_model()

        messagebox.showinfo(
            "Modell trainiert",
            f"Modell erfolgreich trainiert.\n\n"
            f"Samples: {len(x)}\n"
            f"Gebärden: {len(labels)}\n\n"
            f"{', '.join(labels[:20])}"
        )

    def save_model(self):
        if not self.classifier or self.scaler is None:
            return

        model_data = {
            "classes": self.classifier.classes_.tolist(),
            "x_train": self.classifier._fit_X.tolist(),
            "y_train": self.classifier._y.tolist(),
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist()
        }

        with open(MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump(model_data, f)

    def load_model(self):
        if not SKLEARN_AVAILABLE:
            return

        if not os.path.exists(MODEL_FILE):
            return

        try:
            with open(MODEL_FILE, encoding="utf-8") as f:
                data = json.load(f)

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.asarray(data["mean"], dtype=np.float32)
            self.scaler.scale_ = np.asarray(data["scale"], dtype=np.float32)
            self.scaler.var_ = self.scaler.scale_ ** 2
            self.scaler.n_features_in_ = len(self.scaler.mean_)

            x_train = np.asarray(data["x_train"], dtype=np.float32)
            y_train = np.asarray(data["y_train"])

            self.classifier = KNeighborsClassifier(
                n_neighbors=min(7, len(x_train)),
                weights="distance"
            )
            self.classifier.fit(x_train, y_train)
        except Exception:
            self.classifier = None
            self.scaler = None

    def show_dataset_info(self):
        if not self.dataset_x:
            messagebox.showinfo(
                "Datensatz",
                "Noch keine Trainingsdaten vorhanden."
            )
            return

        counts = {}
        for label in self.dataset_y:
            counts[label] = counts.get(label, 0) + 1

        text = [
            f"Samples: {len(self.dataset_x)}",
            f"Gebärden: {len(counts)}",
            "",
            "Gebärden:"
        ]

        for label, count in sorted(counts.items()):
            text.append(f"{label}: {count}")

        messagebox.showinfo(
            "Datensatz",
            "\n".join(text)
        )

    def close(self):
        self.stop_camera()

        try:
            self.hands.close()
            self.pose.close()
            self.face.close()
        except Exception:
            pass

        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DSGSTranslator()
    app.run()