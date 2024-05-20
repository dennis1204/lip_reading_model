#!/usr/bin/env python
# coding: utf-8

# 0. Install and Import Dependencies

import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
import dlib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

tf.config.list_physical_devices('GPU')

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# 1. Build Data Loading Functions

def load_video(path: str) -> List[float]: 
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

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# print(
#     f"The vocabulary is: {char_to_num.get_vocabulary()} "
#     f"(size ={char_to_num.vocabulary_size()})"
# )

char_to_num.get_vocabulary()

def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])
    output = char_to_num(tf.reshape(tf.strings.unicode_split(tf.constant(tokens), input_encoding='UTF-8'), (-1)))[1:]
    return output

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data_cantonese','s1',f'{file_name}.mp4')
    alignment_path = os.path.join('data_cantonese','alignment','s1',f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# 2. Create Data Pipeline

data = tf.data.Dataset.list_files('./data_cantonese/s1/*.mp4')
data = data.shuffle(709, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75,60,135,1],[None]))
data = data.prefetch(tf.data.AUTOTUNE)

# Determine the sizes for training and testing sets
total_samples = 354
train_size = int(0.8 * total_samples)

# Split the data
train_data = data.take(train_size)
test_data = data.skip(train_size)

# 3. Design the Deep Neural Network

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

def create_lip_reading_model(input_shape, num_classes):
    model = Sequential()
    # Modified 3D CNN (ResNet-18 style)
    model.add(Conv3D(128, 3, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))
    # Bi-LSTM (RNN)
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))
    
    # Densely Connected TCN
    model.add(Dense(num_classes, kernel_initializer='he_normal', activation='softmax'))

    return model

input_shape = (75,60,135,1)  # Modify based on actual input shape
num_classes = char_to_num.vocabulary_size() + 1 # Replace with the actual number of classes

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.dataset_iterator = None
        self.checkpoint_directory = "./saved_models/tested_model_7s_0"
    def on_epoch_begin(self, epoch, logs=None) -> None:
        # Create a new iterator at the beginning of each epoch
        self.dataset_iterator = self.dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        # Get the next batch of data from the iterator
        data = self.dataset_iterator.next()

        # Make predictions using the trained model
        yhat = self.model.predict(data[0])

        # Decode the predicted sequences using CTC decoding
        decoded = tf.keras.backend.ctc_decode(yhat, [75, 75], greedy=False)[0][0].numpy()

        total_samples = len(decoded)
        correct_predictions = 0

        for x in range(total_samples):
            original_str = tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8')
            prediction_str = tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8')

            # Tokenize strings into words
            original_words = original_str.split()
            prediction_words = prediction_str.split()
    
            # Calculate accuracy based on correct words
            correct_words = sum(1 for word in original_words if word in prediction_words)
            accuracy = correct_words / len(original_words) if len(original_words) > 0 else 0.0
    
            print('Original:', original_str)
            print('Prediction:', prediction_str)
            print('Accuracy:', accuracy * 100)
            print('~' * 100)

        # Save the last epoch to a text file
        last_epoch_path = os.path.join(self.checkpoint_directory, 'last_epoch.txt')
        current_epoch = 1
        if os.path.exists(last_epoch_path):
            with open(last_epoch_path, 'r') as file:
                current_epoch = int(file.read())
        
        # Increase the epoch by one
        new_epoch = current_epoch + 1
        
        # Write the new epoch value back to the file
        with open(last_epoch_path, 'w') as file:
            file.write(str(new_epoch))

checkpoint_callback = ModelCheckpoint(
    os.path.join('saved_models','tested_model_7s_0','checkpoint'),
    monitor='loss', 
    save_weights_only=False, 
    mode='min',
    save_best_only=True,
) 

schedule_callback = LearningRateScheduler(scheduler)

example_callback = ProduceExample(test_data)

callbacks = [checkpoint_callback, schedule_callback, example_callback]

tf.keras.utils.get_custom_objects()['CTCLoss'] = CTCLoss

# Ensure the checkpoint directory exists; if not, create it
checkpoint_directory = './saved_models/tested_model_7s_0'
os.makedirs(checkpoint_directory, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_directory, 'checkpoint')
last_epoch_path = os.path.join(checkpoint_directory, 'last_epoch.txt')

# Check if there is a checkpoint file
if os.path.exists(checkpoint_path):
    print("Checkpoint exists in models.")
    # Read the last epoch from the text file
    if os.path.exists(last_epoch_path):
        with open(last_epoch_path, 'r') as file:
            last_epoch = int(file.read())
        print("Last epoch:", last_epoch)
    else:
        last_epoch = 1
        print("No last_epoch.txt found. Starting from epoch 1.")
    model = load_model(checkpoint_path, custom_objects={'CTCLoss': CTCLoss})
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                    loss=CTCLoss)
    model.fit(train_data, epochs=50-last_epoch, validation_data=test_data, callbacks=callbacks)
else:
    model = create_lip_reading_model(input_shape, num_classes)
    print("No checkpoint found. Creating a new model.")
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                    loss=CTCLoss)
    model.fit(train_data, validation_data=test_data, epochs=50, callbacks=callbacks)
