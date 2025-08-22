# Neuro Alert: Smart Drowsiness & Fatigue Detection

Neuro Alert is a real-time drowsiness detection system that uses a hybrid approach of computer vision and deep learning. It's designed to non-invasively monitor a user's facial cues and trigger a timely alert when signs of fatigue are detected. A core principle of the system is its ability to adapt to individual user characteristics, making it both highly accurate and resilient to false alarms.

---

### 🌟 Features

* **Dynamic, Personalized Thresholds:** Unlike systems with fixed, hardcoded thresholds (e.g., Eye Aspect Ratio (EAR) < 0.25), this project includes an interactive **calibration phase**. This process measures a user's unique "awake" metrics for EAR and Mouth Aspect Ratio (MAR), and then dynamically sets detection thresholds as a percentage of these values.
* **Multi-Feature Analysis:** The system doesn't rely on a single metric that can be fooled by a simple blink. Instead, it simultaneously monitors three distinct indicators for a more confident prediction: **Eye Aspect Ratio (EAR)**, **Mouth Aspect Ratio (MAR)**, and **Head Pose Ratio**. Drowsiness is determined by the cumulative behavior of these features over time.
* **Robust Landmark Detection:** The final implementation uses **MediaPipe**, a modern and highly optimized framework from Google. Its Face Mesh solution provides 468 facial landmarks with exceptional speed and accuracy, forming a stable foundation for all geometric calculations.

---

### ⚙️ System Architecture & Implementation

The project is structured as a hybrid machine learning pipeline:

1.  **Input:** The system captures real-time video frames from a webcam.
2.  **Calibration Phase:** A brief, one-time process establishes personalized baseline metrics for the user.
3.  **Landmark Detection:** MediaPipe processes each frame to extract 468 facial landmarks.
4.  **Feature Extraction:** From these landmarks, the system calculates EAR, MAR, and a Head Pose Ratio.
5.  **Multi-Modal Classification:** The system performs two checks for a combined decision: a **Geometric Check** against the calibrated thresholds and a **CNN Check** that passes the eye region to a pre-trained PyTorch CNN model. This multi-modal check drastically reduces false positives from quick blinks.
6.  **Scoring & Decision:** A counter accumulates "drowsy" flags, and an audible alarm is triggered if a sustained period of drowsiness is detected.

---

### 🛠️ Technical Breakdown of Core Functions

* `calibrate()`: This function is the cornerstone of the system's accuracy. It captures a predefined number of frames (`CALIBRATION_FRAMES`) to compute a user's average "awake" EAR and MAR values. The final thresholds are then derived by applying `_FACTOR` hyperparameters to these averages, creating a personalized model.
* `get_ear()`, `get_mar()`, `get_head_pose_ratio()`: These utility functions extract meaningful data from MediaPipe's raw landmark output. They use the `scipy.spatial.distance.euclidean` function to calculate distances between specific landmark points to produce a single, normalized score. This geometric approach is both computationally efficient and reliable.
* `drowsiness_detection_system_robust()`: This is the main function that runs the continuous loop. It reads frames, computes ratios, and evaluates them against the dynamic thresholds. A key feature is its combined decision logic, which requires both a low EAR score and the PyTorch CNN model to detect eye closure before triggering a "drowsy" flag.

---

### 🚀 Installation & Usage

### **Prerequisites**

* Python 3.x
* A webcam

### **Installation**

To get started, clone the repository and install the required libraries listed in `requirements.txt`.

```bash
git clone <repository_url>
cd neuro-alert
pip install -r requirements.txt
