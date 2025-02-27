import torch
import cv2
from ultralytics import YOLO
import json
import numpy as np

# Define class names and colors for visualization
CLASS_NAMES = {0: "person", 2: "car", 7: "truck"}
CLASS_COLORS = {0: (255, 0, 0), 2: (0, 255, 0), 7: (0, 0, 255)}  # Blue: Person, Green: Car, Red: Truck

# Track detected objects across frames (simple tracking using dictionary)
tracked_objects = {}

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# Open video file
capture = cv2.VideoCapture("video.mp4")

# Get video properties
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = int(capture.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' instead of 'XVID'
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Initialize JSON storage
detections = []

frame_count = 0  # Keep track of frames

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break  # Stop if video ends
    
    frame_count += 1
    
    # Run YOLO inference directly on the BGR frame
    results = model(frame)

    frame_detections = []  # Store detections for this frame
    person_count, car_count, truck_count = 0, 0, 0  # Object counts

    # Get the first detection result
    if len(results) > 0:
        result = results[0]
        
        # Check if boxes exist in the result
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu().numpy()  # Convert tensor to numpy
                x1, y1, x2, y2 = map(int, box)  # Convert to integers
                conf = float(result.boxes.conf[i].cpu().numpy())  # Confidence score
                cls = int(result.boxes.cls[i].cpu().numpy())  # Class ID

                # Check if class is in our defined categories
                if cls in CLASS_NAMES:
                    obj_name = CLASS_NAMES[cls]  # Get object name
                    color = CLASS_COLORS[cls]  # Assign color

                    # Count objects separately
                    if cls == 0:
                        person_count += 1
                    elif cls == 2:
                        car_count += 1
                    elif cls == 7:
                        truck_count += 1

                    # Object Tracking (store last known position)
                    tracked_objects[i] = (x1, y1, x2, y2)

                    # Draw bounding box with class label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Save detection data
                    frame_detections.append({
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "class": obj_name, "confidence": float(conf)
                    })

    # Save detections per frame
    detections.append({
        "frame": frame_count,
        "person_count": person_count,
        "car_count": car_count,
        "truck_count": truck_count,
        "objects": frame_detections
    })

    # Display object counts
    cv2.putText(frame, f"People: {person_count}  Cars: {car_count}  Trucks: {truck_count}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame in real-time
    cv2.imshow("Vehicle Detection", frame)

    # Save frame to output video
    out.write(frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
out.release()
cv2.destroyAllWindows()

# Save detections to JSON file
with open("detections.json", "w") as file:
    json.dump(detections, file, indent=4)

print(f"Detections saved to detections.json")