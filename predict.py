import cv2
import math
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# =====================================================================
# 0. CENTROID TRACKER IMPLEMENTATION WITH EMBEDDED CLASS LABELS
# =====================================================================
class Tracker:
    def __init__(self):
        self.center_points = {}
        self.classes = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h, cls_name = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id_val, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 60:
                    self.center_points[id_val] = (cx, cy)
                    self.classes[id_val] = cls_name
                    objects_bbs_ids.append([x, y, w, h, id_val, cls_name])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.classes[self.id_count] = cls_name
                objects_bbs_ids.append([x, y, w, h, self.id_count, cls_name])
                self.id_count += 1

        new_center_points = {}
        new_classes = {}
        for obj in objects_bbs_ids:
            obj_id = obj[4]
            new_center_points[obj_id] = self.center_points[obj_id]
            new_classes[obj_id] = self.classes[obj_id]

        self.center_points = new_center_points.copy()
        self.classes = new_classes.copy()
        return objects_bbs_ids


# =====================================================================
# 1. FILE DIALOG FOR VIDEO SELECTION
# =====================================================================
def select_video_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("Opening system file dialog... Please select a traffic video.")
    file_path = filedialog.askopenfilename(
        title="Select Traffic Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
    )
    root.destroy()
    return file_path

video_path = select_video_file()

if not video_path:
    print("Error: No video file selected. Terminating execution.")
    exit()

# =====================================================================
# 2. INFERENCE CONFIGURATION & PIPELINE SETUP
# =====================================================================
# Official Pretrained YOLOv8 Nano weights
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None or math.isnan(fps):
    fps = 30.0

class_names = model.names
tracker = Tracker()

count = 0
prev_positions = {}
down_timestamps = {}
up_timestamps = {}
counter_down = []
counter_up = []

speed_records = {}
overspeed_ids = set()

# Virtual measurement boundaries
RED_LINE_Y = 210
BLUE_LINE_Y = 368
DISTANCE_METERS = 10.0

WINDOW_NAME = "Automated Vehicle Speed Estimation System (Pretrained YOLOv8)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1020, 500)

# =====================================================================
# 3. REAL-TIME PROCESSING LOOP
# =====================================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video stream finished or end of file reached.")
        break

    count += 1
    current_time_sec = count / fps

    frame = cv2.resize(frame, (1020, 500))

    # Pretrained COCO vehicle filter: 2:car, 3:motorcycle, 5:bus, 7:truck
    results = model.predict(frame, imgsz=416, classes=[2, 3, 5, 7], verbose=False)
    
    detections_list = []
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for box, cls_id in zip(xyxy, clss):
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            c_name = class_names[int(cls_id)].upper()

            detections_list.append([x1, y1, w, h, c_name])

    bbox_id = tracker.update(detections_list)

    for bbox in bbox_id:
        x3, y3, w, h, id_val, cls_name = bbox
        cx = x3 + w // 2
        cy = y3 + h // 2

        # ─── DIRECTIONAL CROSSING & VELOCITY CALCULATION ───
        if id_val in prev_positions:
            prev_cy = prev_positions[id_val]

            # Downward Vector (Red Line -> Blue Line)
            if prev_cy <= RED_LINE_Y and cy >= RED_LINE_Y:
                down_timestamps[id_val] = current_time_sec

            if id_val in down_timestamps and prev_cy <= BLUE_LINE_Y and cy >= BLUE_LINE_Y:
                if id_val not in counter_down:
                    counter_down.append(id_val)
                    elapsed_time = current_time_sec - down_timestamps[id_val]
                    if elapsed_time > 0:
                        speed_kmh = (DISTANCE_METERS / elapsed_time) * 3.6
                        speed_records[id_val] = int(speed_kmh)
                        if speed_kmh > 40:
                            overspeed_ids.add(id_val)

            # Upward Vector (Blue Line -> Red Line)
            if prev_cy >= BLUE_LINE_Y and cy <= BLUE_LINE_Y:
                up_timestamps[id_val] = current_time_sec

            if id_val in up_timestamps and prev_cy >= RED_LINE_Y and cy <= RED_LINE_Y:
                if id_val not in counter_up:
                    counter_up.append(id_val)
                    elapsed_time = current_time_sec - up_timestamps[id_val]
                    if elapsed_time > 0:
                        speed_kmh = (DISTANCE_METERS / elapsed_time) * 3.6
                        speed_records[id_val] = int(speed_kmh)
                        if speed_kmh > 40:
                            overspeed_ids.add(id_val)

        prev_positions[id_val] = cy

        # ─── HUD LABEL RENDERING (NO BOUNDING BOXES) ───
        display_text = f"ID:{id_val} {cls_name}"

        if id_val in speed_records:
            display_text += f" | {speed_records[id_val]} Km/h"
        else:
            display_text += " | Tracking..."

        text_color = (0, 0, 255) if id_val in overspeed_ids else (0, 255, 0)

        cv2.putText(frame, display_text, (x3, max(20, y3 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

    # ─── STATISTICAL DASHBOARD OVERLAY ───
    cv2.rectangle(frame, (0, 0), (280, 80), (0, 255, 255), -1)
    cv2.line(frame, (0, RED_LINE_Y), (1020, RED_LINE_Y), (0, 0, 255), 2)
    cv2.putText(frame, 'Red Line', (20, RED_LINE_Y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.line(frame, (0, BLUE_LINE_Y), (1020, BLUE_LINE_Y), (255, 0, 0), 2)
    cv2.putText(frame, 'Blue Line', (20, BLUE_LINE_Y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, f'Downward Count: {len(counter_down)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f'Upward Count: {len(counter_up)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()