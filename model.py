import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')  # Adjust the model variant as needed

# Capture video from file
video_path = "input2.mp4"  # Path to your input video file
cap = cv2.VideoCapture(video_path)

# Define the output video writer
output_path = "output2=2.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Calculate frame interval for 2-second processing
frame_interval = fps
frame_count = 0
markers = []  # To store persistent markers (dots)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 2 seconds
    if frame_count % frame_interval == 0:
        # Perform YOLO inference
        results = model(frame)

        # Filter detections for 'person' class and set dot color based on density
        detections = results[0].boxes
        people_count = sum(int(detection.cls[0]) == 0 for detection in detections)  # Assuming 0 is the 'person' class

        # Set density-based color
        if people_count < 10:
            dot_color = (0, 255, 0)  # Green for low density
        elif 10 <= people_count < 20:
            dot_color = (0, 255, 255)  # Yellow for medium density
        else:
            dot_color = (0, 0, 255)  # Red for high density

        # Update markers with detections
        markers = []
        for detection in detections:
            class_id = int(detection.cls[0])
            if class_id == 0:  # Assuming 0 is the class ID for 'person'
                # Get x_center and y_center directly from the detection's xywh format
                x_center, y_center = int(detection.xywh[0][0]), int(detection.xywh[0][1])
                markers.append((x_center, y_center, dot_color))

    # Draw persistent markers on all frames
    for x_center, y_center, dot_color in markers:
        cv2.circle(frame, (x_center, y_center), 5, dot_color, -1)

    # Display count on frame (from the most recent processing)
    cv2.putText(frame, f"People Count: {len(markers)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video file
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

# Display the output video in Colab
print("Processing complete. Displaying the output video...")
cap_out = cv2.VideoCapture(output_path)
while cap_out.isOpened():
    ret, frame = cap_out.read()
    if not ret:
        break
    cv2_imshow(frame)

cap_out.release()
cv2.destroyAllWindows()
