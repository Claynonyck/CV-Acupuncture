import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import os
import time
import math
import traceback
from ai_processor import AIProcessor

class InkDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x700")
        self.window.eval('tk::PlaceWindow . center')
        
        self.is_running = False
        self.vid = None
        
        # Shared variables for threading
        self.latest_frame = None
        
        self.latest_acu_points = {}
        self.latest_hand_orientation = "Unknown"
        self.latest_detected_marks = []
        
        self.lock = threading.Lock()
        
        self.lbl_title = tk.Label(window, text="Acupoints", font=("Arial", 14, "bold"), fg="#333")
        self.lbl_title.pack(pady=10)
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'best2.pt')
            self.ai = AIProcessor(model_path)
            print(f"Loaded YOLO and MediaPipe.")
        except Exception as e:
            print(f"CRITICAL ERROR LOADING AI MODELS:\n{e}")
            traceback.print_exc()
            messagebox.showerror("Model Error", f"Error loading AI models: {e}")
            self.ai = None

        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack(pady=5)
        
        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(pady=10)
        
        self.btn_start = tk.Button(self.btn_frame, text="Turn on", font=("Arial", 12, "bold"), bg="#000000", fg="white", command=self.start_camera, width=15)
        self.btn_start.grid(row=0, column=0, padx=10)
        
        self.btn_stop = tk.Button(self.btn_frame, text="Turn off", font=("Arial", 12, "bold"), bg="#000000", fg="white", command=self.stop_camera, width=15, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=1, padx=10)
        
        # UI cho kết quả đánh giá
        self.result_label = tk.Label(window, text="Waiting for evaluation...", font=("Consolas", 12, "bold"), fg="#D32F2F", justify=tk.CENTER)
        self.result_label.pack(pady=5)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def start_camera(self):
        if self.ai is None:
            print("WARNING: Cannot start camera because AI Engine is None.")
            messagebox.showwarning("No AI Engine found", "AI Engine failed to load.")
            return
            
        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            print("ERROR: cv2.VideoCapture(0, cv2.CAP_DSHOW) failed. Cannot open camera.")
            messagebox.showerror("Camera Error", "Cannot open camera")
            return
            
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        # Start background threads
        threading.Thread(target=self.camera_thread_func, daemon=True).start()
        threading.Thread(target=self.mediapipe_thread_func, daemon=True).start()
        threading.Thread(target=self.yolo_thread_func, daemon=True).start()
        
        self.update_ui()
        
    def stop_camera(self):
        self.is_running = False
        if self.vid and self.vid.isOpened():
            self.vid.release()
            self.canvas.delete("all")
            
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
    def camera_thread_func(self):
        while self.is_running:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.latest_frame = frame
                    
    def mediapipe_thread_func(self):
        while self.is_running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                # Chạy MediaPipe 
                acu_points, hand_orientation = self.ai.process_mediapipe(frame_to_process)
                with self.lock:
                    self.latest_acu_points = acu_points
                    self.latest_hand_orientation = hand_orientation
                    
            time.sleep(0.01)

    def yolo_thread_func(self):
        while self.is_running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is not None:
                # Chạy YOLO 
                detected_marks = self.ai.process_yolo(frame_to_process)
                with self.lock:
                    self.latest_detected_marks = detected_marks
                    
            time.sleep(0.01)
            
    def update_ui(self):
        if not self.is_running:
            return
            
        base_frame = None
        acu_points = {}
        hand_orientation = "Unknown"
        detected_marks = []
        
        with self.lock:
            if self.latest_frame is not None:
                base_frame = self.latest_frame.copy()
            acu_points = self.latest_acu_points
            hand_orientation = self.latest_hand_orientation
            detected_marks = self.latest_detected_marks
                
        if base_frame is not None:
            annotated_frame = base_frame
            
            # Hiển thị Sấp / Ngửa
            cv2.putText(annotated_frame, f"State: {hand_orientation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Vẽ các huyệt vị 
            for name, (ax, ay) in acu_points.items():
                cv2.circle(annotated_frame, (ax, ay), 4, (255, 0, 0), -1)
                
            # Tính toán và vẽ đánh giá sai số
            evaluation = []
            threshold = 10 # pixels
            
            for mark in detected_marks:
                cx, cy = mark['center']
                min_dist = float('inf')
                closest_acu = None
                
                for name, (ax, ay) in acu_points.items():
                    dist = math.sqrt((cx - ax)**2 + (cy - ay)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_acu = name
                        
                if closest_acu:
                    is_correct = min_dist <= threshold
                    evaluation.append({
                        'mark_id': mark['id'],
                        'closest_acu': closest_acu,
                        'dist': min_dist,
                        'is_correct': is_correct,
                        'mark_center': (cx, cy),
                        'acu_center': acu_points[closest_acu],
                        'bbox': mark['bbox']
                    })
                    
            result_texts = []
            for ev in evaluation:
                mx, my = ev['mark_center']
                ax, ay = ev['acu_center']
                x1, y1, x2, y2 = ev['bbox']
                color = (0, 255, 0) if ev['is_correct'] else (0, 0, 255)
                
                # Vẽ khung YOLO
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated_frame, (mx, my), 3, color, -1)
                
                # Vẽ đường nối giữa vết mực và huyệt
                cv2.line(annotated_frame, (mx, my), (ax, ay), (0, 255, 255), 1)
                
                status = "PASS" if ev['is_correct'] else f"FAIL ({int(ev['dist'])}px)"
                label_text = f"{ev['closest_acu']}: {status}"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                result_texts.append(label_text)
                
            if result_texts:
                self.result_label.config(text=" | ".join(result_texts))
            else:
                self.result_label.config(text="No marks detected or hands visible.")
                
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(annotated_frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
        self.window.after(15, self.update_ui)
        
    def on_closing(self):
        self.stop_camera()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = InkDetectorApp(root, "Acupuncture Evaluation System")
    root.mainloop()
