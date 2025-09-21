import cv2
from ultralytics import YOLO
import csv
import datetime

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_NAME = "yolov8n.pt"
TARGET_CLASS = "person"
SAVE_LOG = True
CSV_FILE = "object_counts.csv"

# -----------------------------
# INITIALIZE
# -----------------------------
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(0)

if SAVE_LOG:
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Class", "Count"])

print("🚀 Object Counter Started! Click buttons on screen.")

# -----------------------------
# BUTTON DEFINITIONS
# -----------------------------
buttons = [
    {"label": "person", "pos": (20, 20, 140, 60)},
    {"label": "car", "pos": (160, 20, 280, 60)},
    {"label": "dog", "pos": (300, 20, 420, 60)},
    {"label": "bicycle", "pos": (440, 20, 560, 60)},
    {"label": "quit", "pos": (580, 20, 700, 60)}
]

running = True

def mouse_click(event, x, y, flags, param):
    global TARGET_CLASS, running
    if event == cv2.EVENT_LBUTTONDOWN:
        for btn in buttons:
            x1, y1, x2, y2 = btn["pos"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                if btn["label"] == "quit":
                    running = False
                else:
                    TARGET_CLASS = btn["label"]
                    print(f"🔄 Switched target to: {TARGET_CLASS}")

cv2.namedWindow("Object Counter")
cv2.setMouseCallback("Object Counter", mouse_click)

# -----------------------------
# MAIN LOOP
# -----------------------------
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)
    detections = results[0].boxes

    count = 0
    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls_id]

        if label == TARGET_CLASS:
            count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show info
    cv2.putText(frame, f"Class: {TARGET_CLASS} | Count: {count}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    # Draw buttons
    for btn in buttons:
        x1, y1, x2, y2 = btn["pos"]
        if btn["label"] == TARGET_CLASS:
            color = (0, 200, 0)  # Active = green
        elif btn["label"] == "quit":
            color = (0, 0, 200)  # Quit = red
        else:
            color = (200, 200, 200)  # Normal = gray

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)  # border
        cv2.putText(frame, btn["label"], (x1 + 10, y2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save log
    if SAVE_LOG:
        with open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, TARGET_CLASS, count])

    # Show video
    cv2.imshow("Object Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Project Ended. Logs saved in:", CSV_FILE)
