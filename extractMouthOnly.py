import cv2
import dlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()
# Load the facial landmark predictor from dlib
predictor = dlib.shape_predictor("./face_landmarks_model/shape_predictor_68_face_landmarks.dat")

# Function to extract the mouth region from a frame
def extract_mouth(frame, landmarks):
    # Define the mouth region based on the facial landmarks
    mouth_pts = landmarks[48:68]
    # Create a mask for the mouth region
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [mouth_pts], (255, 255, 255))
    # Extract the mouth region from the frame using the mask
    mouth = cv2.bitwise_and(frame, mask)
    # Draw landmarks on the frame
    for (x, y) in mouth_pts:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    # print("Mouth region shape:", mouth.shape)
    return mouth, frame

def load_video_mouth(path:str, output_path:str) -> None: 
    import os

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open video: {path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Failed to create video writer: {output_path}")
        cap.release()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)
        if faces:
            for face in faces:
                landmarks = predictor(gray_frame, face)
                landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                mouth_region, _ = extract_mouth(frame, landmarks)
                out.write(mouth_region)
                # print("Frame written.")
        else:
            print("No faces detected in frame.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_mouth.py <input_video_path> <output_video_path>")
        sys.exit(1)
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    load_video_mouth(input_video_path, output_video_path)
    # load_video_and_extract_mouth(input_video_path, output_video_path)
