import os
import cv2
import tensorflow as tf
import numpy as np
import argparse
from matplotlib import pyplot as plt
import dlib
from typing import List

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a video with the lip-reading model.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='./output_predictions.txt', help='Output file to save the predictions.')
    return parser.parse_args()

# Function to load and process video
def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[310:370,300:435,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def ensure_frame_count(frames, target_frame_count=75):
    current_frame_count = frames.shape[0]
    if current_frame_count < target_frame_count:
        # Pad with the last frame
        padding = [frames[-1]] * (target_frame_count - current_frame_count)
        frames = tf.concat([frames, tf.stack(padding)], axis=0)
    elif current_frame_count > target_frame_count:
        # Crop to the target frame count
        frames = frames[:target_frame_count]
    return frames


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Main function to load data, model and process the video
def main():
    args = parse_arguments()

    tf.keras.utils.get_custom_objects()['CTCLoss'] = CTCLoss

    # Load video frames
    frames = load_video(args.video_path)
    frames = ensure_frame_count(frames)

    # Load a trained model
    model = tf.keras.models.load_model('./saved_models/tested_model_4/checkpoint')
    
    # Predict using the model
    predictions = model.predict(tf.expand_dims(frames, axis=0))
    input_length = [frames.shape[0]]
    decoded = tf.keras.backend.ctc_decode(predictions, input_length=input_length, greedy=True)[0][0].numpy()
    total_samples = len(decoded)

    # Create a character lookup
    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)
    
    # Convert numbers to characters and join them into a string, ensuring to handle sequences correctly
    for x in range(total_samples):
        prediction_str = tf.strings.reduce_join([num_to_char(word) for word in decoded[x]]).numpy().decode('utf-8')
        prediction_words = prediction_str.split()
        print('Prediction:', prediction_str)
        print('~' * 100)
        # Save the predictions to a file
        with open(args.output, 'w') as f:
            f.write(prediction_str)
        print('Predictions saved to:', args.output)

if __name__ == '__main__':
    main()
