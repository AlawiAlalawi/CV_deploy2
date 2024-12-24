import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image

# Load your YOLOv8 model
model = YOLO("best4.pt")  # Replace with your YOLO model path if needed

st.title("Parking Lot Detection with Occupancy Grid")
st.write("Upload a video to detect parking spaces and display their status (occupied or empty) in a grid.")

# Sidebar settings
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
grid_rows = st.sidebar.number_input("Grid rows", min_value=1, max_value=20, value=4)
grid_cols = st.sidebar.number_input("Grid columns", min_value=1, max_value=20, value=20)

# Create a grid for spaces
def create_grid(grid_rows, grid_cols):
    spaces = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            spaces.append(f"{chr(65 + i)}{j + 1}")
    return spaces

# Map detections to parking spaces
def map_detections_to_spaces(boxes, spaces, frame_shape, grid_rows, grid_cols):
    occupancy = {space: "empty" for space in spaces}

    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        row = int(y_center * grid_rows / frame_shape[0])
        col = int(x_center * grid_cols / frame_shape[1])
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            space = f"{chr(65 + row)}{col + 1}"
            occupancy[space] = "occupied"

    return occupancy

# Create the occupancy grid visualization
def create_occupancy_map(occupancy, grid_rows, grid_cols):
    map_shape = (480, 640, 3)
    occ_map = np.full(map_shape, (25, 25, 75), dtype=np.uint8)

    for i, row in enumerate(range(65, 65 + grid_rows)):
        for j in range(grid_cols):
            space = f"{chr(row)}{j + 1}"
            x1 = 10 + j * (map_shape[1] - 20) // grid_cols
            y1 = 10 + i * (map_shape[0] - 20) // grid_rows
            x2 = 10 + (j + 1) * (map_shape[1] - 20) // grid_cols
            y2 = 10 + (i + 1) * (map_shape[0] - 20) // grid_rows

            color = (0, 255, 0) if occupancy[space] == "empty" else (0, 0, 255)
            cv2.rectangle(occ_map, (x1, y1), (x2, y2), color, -1)
            cv2.putText(occ_map, space, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return occ_map

# Process the video frame by frame
def process_video(video_path, frame_skip=15):
    video_cap = cv2.VideoCapture(video_path)
    spaces = create_grid(grid_rows, grid_cols)

    # Output placeholders
    det_out = st.empty()
    occ_out = st.empty()
    summary_out = st.empty()

    frame_counter = 0

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Skip frames based on the frame_skip value
        if frame_counter % (frame_skip + 1) != 0:
            frame_counter += 1
            continue

        # Perform YOLO prediction
        results = model.predict(source=frame, save=False, conf=conf_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
        occupancy = map_detections_to_spaces(boxes, spaces, frame.shape, grid_rows, grid_cols)

        # Visualize detections
        detection_vis = results[0].plot() if results else frame

        # Create the occupancy map
        occ_map = create_occupancy_map(occupancy, grid_rows, grid_cols)

        # Display the frames
        det_out.image(detection_vis, channels="BGR", use_container_width=True)
        occ_out.image(occ_map, use_container_width=True)

        # Display summary
        summary = {
            "Total Spaces": len(spaces),
            "Occupied": sum(1 for s in occupancy.values() if s == "occupied"),
            "Empty": sum(1 for s in occupancy.values() if s == "empty"),
        }
        summary_out.write(summary)

        frame_counter += 1

    video_cap.release()


# Upload and process video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        st.video(temp_video.name)
        if st.button("Process Video"):
            process_video(temp_video.name)
