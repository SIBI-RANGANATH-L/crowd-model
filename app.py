import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO

# Title of the app
st.title("YOLOv8 Object Detection on Video")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_video_path)

    # Load YOLO model
    model = YOLO("yolov8s.pt")

    # Read the video
    cap = cv2.VideoCapture(temp_video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output file
    output_video_path = os.path.join(tempfile.gettempdir(), "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Frame processing
    frame_interval = fps
    frame_count = 0
    markers = []  # Persistent markers for people detected

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 2 seconds
        if frame_count % frame_interval == 0:
            results = model(frame)
            detections = results[0].boxes

            # Count people
            people_count = sum(int(detection.cls[0]) == 0 for detection in detections)

            # Set color based on density
            if people_count < 10:
                dot_color = (0, 255, 0)  # Green for low density
            elif 10 <= people_count < 20:
                dot_color = (0, 255, 255)  # Yellow for medium density
            else:
                dot_color = (0, 0, 255)  # Red for high density

            # Store detected people markers
            markers = []
            for detection in detections:
                class_id = int(detection.cls[0])
                if class_id == 0:
                    x_center, y_center = int(detection.xywh[0][0]), int(detection.xywh[0][1])
                    markers.append((x_center, y_center, dot_color))

        # Draw markers
        for x_center, y_center, dot_color in markers:
            cv2.circle(frame, (x_center, y_center), 5, dot_color, -1)

        # Display count on frame
        cv2.putText(frame, f"People Count: {len(markers)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1

        # Update progress bar
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()

    st.success("Processing complete! Download the processed video below.")

    # Show download link
    with open(output_video_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="output.mp4", mime="video/mp4")
