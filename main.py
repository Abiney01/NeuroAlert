import cv2
import time
import numpy as np
from imutils.video import VideoStream
import imutils
import mediapipe as mp
from scipy.spatial import distance as dist
import playsound
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- Configuration ---
RESIZE_FRAME_WIDTH = 640
DISPLAY_FPS = True

# --- Calibration Settings ---
CALIBRATION_FRAMES = 50
EAR_THRESHOLD_FACTOR = 0.75
MAR_THRESHOLD_FACTOR = 1.3

# --- Drowsiness Thresholds (Default values before calibration) ---
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 25
MAR_THRESHOLD = 0.7
MAR_CONSEC_FRAMES = 20
HEAD_TILT_THRESHOLD_VERTICAL = 0.35
HEAD_TILT_CONSEC_FRAMES = 15
# A new threshold for combining CNN and EAR logic
CNN_EAR_CONSEC_FRAMES = 10 

# --- Alarm System ---
ALARM_FILE_PATH = "alarm.wav"
TOTAL_VIOLATIONS_THRESHOLD = 50

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- PyTorch Model Configuration ---
MODEL_PATH = 'drowsiness_model.pth'
IMAGE_SIZE = (227, 227)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play_alarm():
    if os.path.exists(ALARM_FILE_PATH):
        try:
            playsound.playsound(ALARM_FILE_PATH, block=False)
        except Exception as e:
            print(f"[ALARM] Drowsiness detected! Error playing sound: {e}")
            print("[ALARM] Drowsiness detected! Wake up!")
    else:
        print("[ALARM] Drowsiness detected! Alarm file not found.")
        print("[ALARM] Drowsiness detected! Wake up!")

# --- Model Architecture  ---
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- Utility Functions for EAR, MAR, and Head Pose with MediaPipe ---
def get_ear(landmarks, eye_points):
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]
    A = dist.euclidean([p2.x, p2.y], [p6.x, p6.y])
    B = dist.euclidean([p3.x, p3.y], [p5.x, p5.y])
    C = dist.euclidean([p1.x, p1.y], [p4.x, p4.y])
    ear = (A + B) / (2.0 * C)
    return ear

def get_mar(landmarks, mouth_points):
    p1 = landmarks[mouth_points[0]]
    p2 = landmarks[mouth_points[1]]
    p3 = landmarks[mouth_points[2]]
    p4 = landmarks[mouth_points[3]]
    p5 = landmarks[mouth_points[4]]
    p6 = landmarks[mouth_points[5]]
    p7 = landmarks[mouth_points[6]]
    p8 = landmarks[mouth_points[7]]
    A = dist.euclidean([p2.x, p2.y], [p8.x, p8.y])
    B = dist.euclidean([p3.x, p3.y], [p7.x, p7.y])
    C = dist.euclidean([p4.x, p4.y], [p6.x, p6.y])
    D = dist.euclidean([p1.x, p1.y], [p5.x, p5.y])
    mar = (A + B + C) / (3.0 * D)
    return mar

def get_head_pose_ratio(landmarks):
    nose_tip = landmarks[1]
    chin = landmarks[152]
    left_side = landmarks[226]
    right_side = landmarks[446]
    vertical_dist = dist.euclidean([nose_tip.x, nose_tip.y], [chin.x, chin.y])
    horizontal_dist = dist.euclidean([left_side.x, left_side.y], [right_side.x, right_side.y])
    if horizontal_dist > 0 and vertical_dist > 0:
        vertical_ratio = horizontal_dist / vertical_dist
        return vertical_ratio
    return 1000

def calibrate():
    print("[INFO] Starting calibration phase. Please look at the camera, eyes open, for a few seconds...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    ear_values = []
    mar_values = []
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        
        for i in range(CALIBRATION_FRAMES):
            frame = vs.read()
            if frame is None: break
            frame = imutils.resize(frame, width=RESIZE_FRAME_WIDTH)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
                MOUTH_POINTS = [61, 39, 40, 0, 269, 291, 321, 314]
                left_ear = get_ear(landmarks, LEFT_EYE_POINTS)
                right_ear = get_ear(landmarks, RIGHT_EYE_POINTS)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = get_mar(landmarks, MOUTH_POINTS)
                ear_values.append(avg_ear)
                mar_values.append(mar)
            
            cv2.putText(frame, f"Calibrating: {i+1}/{CALIBRATION_FRAMES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    vs.stop()
    cv2.destroyAllWindows()

    if len(ear_values) > 0 and len(mar_values) > 0:
        avg_ear_calibrated = np.mean(ear_values)
        avg_mar_calibrated = np.mean(mar_values)
        ear_threshold_new = avg_ear_calibrated * EAR_THRESHOLD_FACTOR
        mar_threshold_new = avg_mar_calibrated * MAR_THRESHOLD_FACTOR
        print(f"[INFO] Calibration complete.")
        print(f"[INFO] Average EAR (Open): {avg_ear_calibrated:.3f}")
        print(f"[INFO] Dynamic EAR Threshold: {ear_threshold_new:.3f}")
        print(f"[INFO] Average MAR (Closed): {avg_mar_calibrated:.3f}")
        print(f"[INFO] Dynamic MAR Threshold: {mar_threshold_new:.3f}")
        return ear_threshold_new, mar_threshold_new
    
    print("[ERROR] Calibration failed. Using default thresholds.")
    return EAR_THRESHOLD, MAR_THRESHOLD

def drowsiness_detection_system_robust(ear_threshold, mar_threshold):
    
    # --- Load PyTorch Model ---
    try:
        cnn_model = DrowsinessCNN().to(device)
        cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        cnn_model.eval()
        print(f"[INFO] PyTorch model loaded successfully on device: {device}")
    except Exception as e:
        print(f"[ERROR] Could not load PyTorch model: {e}")
        return

    # --- Preprocessing pipeline for inference ---
    inference_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
    MOUTH_POINTS = [61, 39, 40, 0, 269, 291, 321, 314]

    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    start_time, frame_count, fps = time.time(), 0, 0
    ear_consec_counter = 0
    mar_consec_counter = 0
    head_tilt_consec_counter = 0
    total_violation_counter = 0
    drowsiness_alarm_active = False

    print("Starting Drowsiness Detection System (Live). Press 'q' to quit.")

    while True:
        frame = vs.read()
        if frame is None: break
        
        frame = imutils.resize(frame, width=RESIZE_FRAME_WIDTH)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        status_text = "Status: OK"
        color = (0, 255, 0)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ear = get_ear(landmarks, LEFT_EYE_POINTS)
            right_ear = get_ear(landmarks, RIGHT_EYE_POINTS)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = get_mar(landmarks, MOUTH_POINTS)
            vertical_ratio = get_head_pose_ratio(landmarks)
            
            is_cnn_drowsy = False
            # If eyes are likely closed (low EAR), run CNN for confirmation
            if avg_ear < ear_threshold:
                # Find bounding box for eyes from landmarks
                all_eyes_x = [landmarks[i].x for i in LEFT_EYE_POINTS + RIGHT_EYE_POINTS]
                all_eyes_y = [landmarks[i].y for i in LEFT_EYE_POINTS + RIGHT_EYE_POINTS]
                min_x = int(min(all_eyes_x) * frame.shape[1])
                max_x = int(max(all_eyes_x) * frame.shape[1])
                min_y = int(min(all_eyes_y) * frame.shape[0])
                max_y = int(max(all_eyes_y) * frame.shape[0])
                
                eye_roi_color = frame[min_y:max_y, min_x:max_x]
                
                if eye_roi_color.size > 0:
                    try:
                        eye_roi_pil = Image.fromarray(cv2.cvtColor(eye_roi_color, cv2.COLOR_BGR2RGB))
                        input_tensor = inference_transform(eye_roi_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = cnn_model(input_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            # Class 0: Drowsy, Class 1: Non-Drowsy
                            if predicted.item() == 0:
                                is_cnn_drowsy = True
                    except Exception as e:
                        print(f"[WARNING] CNN prediction failed: {e}")
                        
            violation_this_frame = False
            
            # Use CNN and EAR for a combined decision
            if avg_ear < ear_threshold and is_cnn_drowsy:
                ear_consec_counter += 1
                if ear_consec_counter >= CNN_EAR_CONSEC_FRAMES:
                    status_text = "Status: Eyes Closed!"
                    color = (0, 0, 255)
                    violation_this_frame = True
            else:
                ear_consec_counter = 0

            if mar > mar_threshold:
                mar_consec_counter += 1
                if mar_consec_counter >= MAR_CONSEC_FRAMES:
                    status_text = "Status: Yawning!"
                    color = (0, 165, 255)
                    violation_this_frame = True
            else:
                mar_consec_counter = 0
            
            if vertical_ratio < HEAD_TILT_THRESHOLD_VERTICAL:
                head_tilt_consec_counter += 1
                if head_tilt_consec_counter >= HEAD_TILT_CONSEC_FRAMES:
                    status_text = "Status: Head Nodding!"
                    color = (0, 255, 255)
                    violation_this_frame = True
            else:
                head_tilt_consec_counter = 0

            if violation_this_frame:
                total_violation_counter += 1
            else:
                total_violation_counter = 0

            if total_violation_counter >= TOTAL_VIOLATIONS_THRESHOLD and not drowsiness_alarm_active:
                drowsiness_alarm_active = True
                play_alarm()
            elif total_violation_counter == 0 and drowsiness_alarm_active:
                drowsiness_alarm_active = False

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Head: {vertical_ratio:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            ear_consec_counter = 0; mar_consec_counter = 0; head_tilt_consec_counter = 0
            total_violation_counter = 0
            status_text = "Status: No Face Detected"
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if DISPLAY_FPS:
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if drowsiness_alarm_active:
            cv2.putText(frame, "ALARM ACTIVE!", (RESIZE_FRAME_WIDTH - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Drowsiness Detection System', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    vs.stop()
    cv2.destroyAllWindows()
    print("[INFO] Application stopped gracefully.")

if __name__ == "__main__":
    calibrated_ear_threshold, calibrated_mar_threshold = calibrate()
    drowsiness_detection_system_robust(calibrated_ear_threshold, calibrated_mar_threshold)