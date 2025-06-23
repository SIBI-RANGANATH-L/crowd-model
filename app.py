# import streamlit as st
# import cv2
# import tempfile
# import os
# import numpy as np
# from ultralytics import YOLO

# # Set Streamlit page config
# st.set_page_config(page_title="YOLOv8 Video Detection", layout="centered")

# # Title
# st.title("🎯 YOLOv8 Object Detection on Video")

# # File uploader
# uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

# if uploaded_file is not None:
#     # Save uploaded video temporarily
#     temp_dir = tempfile.mkdtemp()
#     temp_video_path = os.path.join(temp_dir, uploaded_file.name)
#     with open(temp_video_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Display uploaded video
#     st.video(temp_video_path)

#     # Load YOLO model (make sure yolov8s.pt is accessible)
#     st.info("Loading YOLOv8 model...")
#     model = YOLO("yolov8s.pt")

#     # Open the video file
#     cap = cv2.VideoCapture(temp_video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Prepare output video path
#     output_path = os.path.join(temp_dir, "output.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Frame processing settings
#     frame_interval = fps * 1  # every 2 seconds
#     frame_count = 0
#     markers = []

#     st.info("Processing video...")
#     progress_bar = st.progress(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Process every `frame_interval` frames
#         if frame_count % frame_interval == 0:
#             results = model(frame)
#             detections = results[0].boxes

#             # Count people
#             people_count = sum(int(det.cls[0]) == 0 for det in detections)

#             # Set color based on count
#             if people_count < 10:
#                 dot_color = (0, 255, 0)  # Green
#             elif 10 <= people_count < 20:
#                 dot_color = (0, 255, 255)  # Yellow
#             else:
#                 dot_color = (0, 0, 255)  # Red

#             # Store markers
#             markers = []
#             for det in detections:
#                 class_id = int(det.cls[0])
#                 if class_id == 0:
#                     x_center = int(det.xywh[0][0])
#                     y_center = int(det.xywh[0][1])
#                     markers.append((x_center, y_center, dot_color))

#         # Draw markers
#         for x, y, color in markers:
#             cv2.circle(frame, (x, y), 5, color, -1)

#         # Overlay count
#         cv2.putText(frame, f"People Count: {len(markers)}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         out.write(frame)
#         frame_count += 1
#         progress_bar.progress(min(frame_count / total_frames, 1.0))

#     cap.release()
#     out.release()

#     st.success("✅ Processing complete!")
#     with open(output_path, "rb") as f:
#         st.download_button("⬇️ Download Processed Video", f, file_name="output.mp4", mime="video/mp4")


# import streamlit as st
# import cv2
# import tempfile
# import os
# import numpy as np
# from ultralytics import YOLO
# from firebase_init import db
# import pandas as pd


# st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
# st.title("🎯 YOLOv8 Object Detection")

# # Select mode
# mode = st.radio("Select input mode:", ("📁 Upload Video", "🎥 Use Webcam"))

# # Load model once
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8s.pt")

# model = load_model()


# # ----------------------------------
# # 📁 Upload Video Mode
# # ----------------------------------
# if mode == "📁 Upload Video":
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

#     if uploaded_file:
#         # Save uploaded video
#         temp_dir = tempfile.mkdtemp()
#         input_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.read())

#         st.video(input_path)

#         if st.button("🚀 Run Detection"):
#             cap = cv2.VideoCapture(input_path)
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#             output_path = os.path.join(temp_dir, "output.mp4")
#             out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#             frame_interval = 1
#             frame_count = 0
#             markers = []
#             progress_bar = st.progress(0)

#             heatmap = np.zeros((height, width), dtype=np.float32)

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 if frame_count % frame_interval == 0:
#                     results = model(frame)
#                     detections = results[0].boxes
#                     people_count = sum(int(d.cls[0]) == 0 for d in detections)

#                     if people_count < 10:
#                         dot_color = (0, 255, 0)
#                     elif 10 <= people_count < 20:
#                         dot_color = (0, 255, 255)
#                     else:
#                         dot_color = (0, 0, 255)

#                     markers = []
#                     for det in detections:
#                         if int(det.cls[0]) == 0:
#                             x_center = int(det.xywh[0][0])
#                             y_center = int(det.xywh[0][1])
#                             markers.append((x_center, y_center, dot_color))

#                 for x, y, color in markers:
#                     cv2.circle(frame, (x, y), 5, color, -1)

#                 cv2.putText(frame, f"People Count: {len(markers)}", (20, 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#                 out.write(frame)
#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))

#             cap.release()
#             out.release()
#             st.success("✅ Video processing complete!")

#             with open(output_path, "rb") as f:
#                 st.download_button("⬇️ Download Processed Video", f, file_name="output.mp4", mime="video/mp4")


# # ----------------------------------
# # 🎥 Webcam Mode
# # ----------------------------------
# elif mode == "🎥 Use Webcam":
#     st.warning("Press 'Start Webcam' to launch your camera. It will run YOLOv8 live.")
#     run = st.checkbox("▶️ Start Webcam")

#     FRAME_WINDOW = st.image([])

#     if run:
#         cap = cv2.VideoCapture(0)  # 0 = default webcam

#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("❌ Cannot access webcam.")
#                 break

#             results = model(frame)
#             detections = results[0].boxes
#             people_count = sum(int(d.cls[0]) == 0 for d in detections)

#             if people_count < 10:
#                 dot_color = (0, 255, 0)
#             elif 10 <= people_count < 20:
#                 dot_color = (0, 255, 255)
#             else:
#                 dot_color = (0, 0, 255)

#             for det in detections:
#                 if int(det.cls[0]) == 0:
#                     x_center = int(det.xywh[0][0])
#                     y_center = int(det.xywh[0][1])
#                     cv2.circle(frame, (x_center, y_center), 5, dot_color, -1)

#             cv2.putText(frame, f"People Count: {people_count}", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             FRAME_WINDOW.image(frame)

#         cap.release()


import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from firebase_init import db
import time

# Log to Firestore
def log_to_firestore(people_count):
    data = {
        'timestamp': datetime.utcnow(),
        'people_count': people_count
    }
    db.collection('crowd_logs').add(data)

# Realtime Dashboard Renderer
def get_crowd_data():
    docs = db.collection('crowd_logs').order_by('timestamp').stream()
    data = []
    for doc in docs:
        row = doc.to_dict()
        data.append({
            'timestamp': row['timestamp'],
            'people_count': row['people_count']
        })
    return pd.DataFrame(data)

def render_dashboard(df, chart_container, metric_container, table_container):
    if df.empty:
        chart_container.info("No data available.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    chart_container.line_chart(df['people_count'])

    with metric_container:
        st.metric(" Max Count", df['people_count'].max())
        st.metric(" Avg Count", round(df['people_count'].mean(), 2))

    with table_container:
        st.dataframe(df)

# Streamlit UI setup
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title(" YOLOv8 Object Detection")

st.subheader(" Crowd Analytics Dashboard")
chart_container = st.empty()
metric_container = st.container()
table_container = st.expander(" Raw Data", expanded=False)

# Auto refresh checkbox
auto_refresh = st.checkbox(" Auto-refresh dashboard", value=True)
refresh_interval = 5  # seconds

if auto_refresh:
    df = get_crowd_data()
    render_dashboard(df, chart_container, metric_container, table_container)
    time.sleep(refresh_interval)
    st.rerun()
else:
    if st.button(" Manual Refresh"):
        df = get_crowd_data()
        render_dashboard(df, chart_container, metric_container, table_container)

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# Select mode
mode = st.radio("Select input mode:", (" Upload Video", " Use Webcam"))

# \ud83d\udcc1 Upload Video Mode
if mode == " Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(input_path)

        if st.button(" Run Detection"):
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = os.path.join(temp_dir, "output.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            frame_interval = 1
            frame_count = 0
            markers = []
            progress_bar = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    results = model(frame)
                    detections = results[0].boxes
                    people_count = sum(int(d.cls[0]) == 0 for d in detections)
                    log_to_firestore(people_count)

                    if people_count < 10:
                        dot_color = (0, 255, 0)
                    elif 10 <= people_count < 20:
                        dot_color = (0, 255, 255)
                    else:
                        dot_color = (0, 0, 255)

                    markers = []
                    for det in detections:
                        if int(det.cls[0]) == 0:
                            x_center = int(det.xywh[0][0])
                            y_center = int(det.xywh[0][1])
                            markers.append((x_center, y_center, dot_color))

                for x, y, color in markers:
                    cv2.circle(frame, (x, y), 5, color, -1)

                cv2.putText(frame, f"People Count: {len(markers)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                out.write(frame)
                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            st.success(" Video processing complete!")

            with open(output_path, "rb") as f:
                st.download_button(" Download Processed Video", f, file_name="output.mp4", mime="video/mp4")

# \ud83c\udfa5 Webcam Mode
elif mode == " Use Webcam":
    st.warning("Press 'Start Webcam' to launch your camera. It will run YOLOv8 live.")
    run = st.checkbox(" Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error(" Cannot access webcam.")
                break

            results = model(frame)
            detections = results[0].boxes
            people_count = sum(int(d.cls[0]) == 0 for d in detections)
            log_to_firestore(people_count)

            if people_count < 10:
                dot_color = (0, 255, 0)
            elif 10 <= people_count < 20:
                dot_color = (0, 255, 255)
            else:
                dot_color = (0, 0, 255)

            for det in detections:
                if int(det.cls[0]) == 0:
                    x_center = int(det.xywh[0][0])
                    y_center = int(det.xywh[0][1])
                    cv2.circle(frame, (x_center, y_center), 5, dot_color, -1)

            cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

# import streamlit as st
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from ultralytics import YOLO
# from firebase_init import db
# from streamlit_autorefresh import st_autorefresh  # install via pip if needed

# # Load model once and cache it
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8s.pt")

# model = load_model()

# # Log people count to Firestore
# def log_to_firestore(people_count):
#     data = {
#         'timestamp': datetime.utcnow(),
#         'people_count': people_count
#     }
#     db.collection('crowd_logs').add(data)

# # Fetch Firestore crowd data
# def fetch_crowd_data():
#     docs = db.collection('crowd_logs').order_by('timestamp').stream()
#     data = []
#     for doc in docs:
#         d = doc.to_dict()
#         data.append({'timestamp': d['timestamp'], 'people_count': d['people_count']})
#     if not data:
#         return pd.DataFrame()
#     df = pd.DataFrame(data)
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df.set_index('timestamp', inplace=True)
#     return df

# # Initialize webcam once in session_state
# if 'cap' not in st.session_state:
#     st.session_state.cap = cv2.VideoCapture(0)

# # Auto-refresh the app every 1 second (1000 ms)
# count = st_autorefresh(interval=1000, limit=None, key="webcam_refresh")

# st.title("Real-time Webcam + Crowd Analytics Dashboard")

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("🎥 Webcam Live Feed")
#     frame_placeholder = st.empty()

#     ret, frame = st.session_state.cap.read()
#     if not ret:
#         st.error("Cannot read from webcam.")
#     else:
#         results = model(frame)
#         detections = results[0].boxes
#         people_count = sum(int(d.cls[0]) == 0 for d in detections)
#         log_to_firestore(people_count)

#         # Draw detections
#         for det in detections:
#             if int(det.cls[0]) == 0:
#                 x_center = int(det.xywh[0][0])
#                 y_center = int(det.xywh[0][1])
#                 cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)
#         cv2.putText(frame, f"People Count: {people_count}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame, channels="RGB")

# with col2:
#     st.subheader("📊 Crowd Analytics Dashboard (Last 100 entries)")
#     df = fetch_crowd_data()
#     if df.empty:
#         st.info("No data available yet.")
#     else:
#         df_recent = df.tail(100)
#         st.line_chart(df_recent['people_count'])
#         st.metric("📈 Max Count", int(df_recent['people_count'].max()))
#         st.metric("📊 Avg Count", round(df_recent['people_count'].mean(), 2))
#         with st.expander("📄 Raw Data"):
#             st.dataframe(df_recent)

# # When user stops app, release webcam
# def cleanup():
#     if 'cap' in st.session_state:
#         st.session_state.cap.release()

# # Optional: register a cleanup on script stop (Streamlit currently has no official event for this)
# # You can manually stop webcam by closing the app.


