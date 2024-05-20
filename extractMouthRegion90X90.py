import cv2
import dlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./face_landmarks_model/shape_predictor_68_face_landmarks.dat")

def load_video_and_extract_mouth(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    
    # Assume the output video size is 90x90 for the mouth region
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (90, 90), isColor=False)
    
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)
        
        for face in faces:
            landmarks = predictor(gray_frame, face)
            landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            mouth_pts = landmarks[48:68]
            min_x = np.min(mouth_pts[:, 0])
            max_x = np.max(mouth_pts[:, 0])
            min_y = np.min(mouth_pts[:, 1])
            max_y = np.max(mouth_pts[:, 1])
            
            # Ensure the bounding box stays within the boundaries of the frame
            frame_height, frame_width = frame.shape[:2]
            min_x = max(min_x, 0)
            max_x = min(max_x, frame_width)
            min_y = max(min_y, 0)
            max_y = min(max_y, frame_height)
            
            # Crop the mouth region
            mouth_region = frame[min_y:max_y, min_x:max_x]
            
            # Resize the cropped mouth region to 90x90
            mouth_region_resized = cv2.resize(mouth_region, (90, 90))
            mouth_region_resized_gray = cv2.cvtColor(mouth_region_resized, cv2.COLOR_BGR2GRAY)
            
            # Write the processed frame to the output video
            out.write(mouth_region_resized_gray)
            
            # Optional: Append for additional processing (not needed for video output)
            frames.append(mouth_region_resized_gray)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Optional: Standardize frames if needed elsewhere (not needed for video output)
    if frames:
        frames_array = np.array(frames, dtype=np.float32)
        mean = np.mean(frames_array)
        std = np.std(frames_array)
        standardized_frames = (frames_array - mean) / std
        return standardized_frames



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_mouth.py <input_video_path> <output_video_path>")
        sys.exit(1)
    input_video_path = sys.argv[1]
    output_video_path = sys.argv[2]
    load_video_and_extract_mouth(input_video_path, output_video_path)