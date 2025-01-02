import cv2
import time
import threading
import logging
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askdirectory

# General Configuration
OUTPUT_DIR = "output"
VIDEO_RECORD_SECONDS = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Logging Setup
logging.basicConfig(
    filename="security_camera.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_event(camera_index, event_type, file_path):
    logging.info(f"Camera {camera_index}: {event_type} - {file_path}")    

def detect_motion(camera_index=0, sensitivity=50):
    cap = cv2.VideoCapture(camera_index)
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while True:
        ret, frame = cap.read()
        
        if not ret:
            log_event(camera_index, "Error", "Failed to capture frame")
            print(f"Failed to capture frame for camera {camera_index}")
            return None
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        delta_frame = cv2.absdiff(prev_gray, gray_frame)
        _, thresh_frame = cv2.threshold(delta_frame, sensitivity, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_file = f"{OUTPUT_DIR}/motion_{camera_index}_{timestamp}.jpg"

            cv2.imwrite(image_file, frame)
            log_event(camera_index, "Motion Detected", image_file)
            print(f"Motion detected for camera {camera_index} - {image_file}")
            time.sleep(VIDEO_RECORD_SECONDS)

        prev_gray = gray_frame
    cap.release()
    cv2.destroyAllWindows()


def start_camera_thread(camera_index, sensitivity):
    # Starts a motion detection thread for a specific camera
    threading.Thread(
        target=detect_motion, 
        args=(camera_index, sensitivity),
        daemon=True
    ).start()

def detect_motion_for_multiple_cameras(camera_indices, sensitivity):
    # Starts monitoring for multiple cameras
    for camera_index in camera_indices:
        start_camera_thread(camera_index, sensitivity)


# GUI Application for the Security Camera System
class SecurityCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Security Camera System")
        self.root.geometry("300x300")

        self.camera_index = tk.IntVar(value=0)
        self.sensitivity = tk.IntVar(value=50)
        self.output_dir = tk.StringVar(value="./output")

        tk.Label(root, text="Camera Index:").pack(pady=5)
        tk.Entry(root, textvariable=self.camera_index).pack(pady=5)

        tk.Label(root, text="Sensitivity:").pack(pady=5)
        tk.Scale(root, variable=self.sensitivity, from_=10, to=100, orient=tk.HORIZONTAL).pack(pady=5)

        tk.Label(root, text="Output Directory:").pack(pady=5)
        tk.Entry(root, textvariable=self.output_dir).pack(pady=5)
        tk.Button(root, text="Browse", command=self.select_output_dir).pack(pady=5)

        tk.Button(root, text="Start Monitoring", command=self.start_monitoring).pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Idle", fg="blue")
        self.status_label.pack(pady=20)

        self.monitoring_threads = []

    def select_output_dir(self):
        # Open a directory dialog to select the output directory
        directory = askdirectory()
        if directory:
            self.output_dir.set(directory)

    def start_monitoring(self):
        # Start motion detection
        camera_idx = self.camera_index.get()
        sensitivity = self.sensitivity.get()

        thread = threading.Thread(
            target=detect_motion,
            args=(camera_idx, sensitivity),
            daemon=True
        )
        thread.start()
        self.monitoring_threads.append(thread)

        self.status_label.config(text="Status: Monitoring...", fg="green")
        messagebox.showinfo("Info", "Monitoring started!")


if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityCameraApp(root)
    root.mainloop()