import numpy as np
import tensorflow as tf
import os
import cv2
import librosa
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from google.colab.patches import cv2_imshow

# Constants
NUM_FRAMES = 30
FRAME_SIZE = (224, 224)
NUM_MFCC = 128

# Paths
MODEL_PATH = "/content/drive/MyDrive/NEW/deepfake_detector.h5"  # Update this
VIDEO_PATH = "/content/drive/MyDrive/Train/data/fake/00005_id01589_ZCwF6XVRiAU.mp4"  # Update this

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load feature extractor (MobileNetV2)
base_cnn = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_cnn.output)  # Reduce feature size
feature_extractor = Model(inputs=base_cnn.input, outputs=x)
feature_extractor.trainable = False

def extract_video_frames(video_path):
    """Extract frames from video and return processed frames + original frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    processed_frames = []
    original_frames = []  # Store original frames for visualization

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        original_frames.append(frame)  # Store unprocessed frame
        frame = cv2.resize(frame, FRAME_SIZE) / 255.0  # Normalize
        processed_frames.append(frame)

    cap.release()

    if len(processed_frames) != NUM_FRAMES:
        print("Warning: Not enough frames extracted")
        return None, None

    return np.array(processed_frames), original_frames

def extract_audio_features(video_path):
    """Extract MFCC features from video audio."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        video = VideoFileClip(video_path)
        if video.audio is None:
            print(f"Warning: No audio in {video_path}")
            return np.zeros((NUM_FRAMES, NUM_MFCC))

        video.audio.write_audiofile(temp_audio_path, codec="pcm_s16le", verbose=False, logger=None)
        video.close()

        y, sr = librosa.load(temp_audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NUM_MFCC)
        mfccs = np.transpose(mfccs)  # Shape: (time_steps, 128)

        original_steps = mfccs.shape[0]
        if original_steps < 1:
            return np.zeros((NUM_FRAMES, NUM_MFCC))

        # Resample to NUM_FRAMES
        resampled_mfccs = np.zeros((NUM_FRAMES, NUM_MFCC))
        for i in range(NUM_MFCC):
            resampled_mfccs[:, i] = np.interp(
                np.linspace(0, original_steps - 1, NUM_FRAMES),
                np.arange(original_steps),
                mfccs[:, i] if original_steps > 0 else np.zeros(1)
            )

        return resampled_mfccs  # Shape: (30, 128)

    except Exception as e:
        print(f"Extracting audio from {video_path}: {str(e)}")
        return np.zeros((NUM_FRAMES, NUM_MFCC))

# Process the video
processed_frames, original_frames = extract_video_frames(VIDEO_PATH)  # Extract frames
audio_features = extract_audio_features(VIDEO_PATH)  # Extract audio

if processed_frames is None or audio_features is None:
    print("Error: Could not process video")
    exit()

# Extract CNN features from frames
video_features = feature_extractor.predict(processed_frames, verbose=0)

# Store confidence scores for each frame
confidences = []

SEQ_LENGTH = 30

for i in range(len(video_features) - SEQ_LENGTH + 1):  # Ensure we have enough frames
    # Select a sequence of 30 frames
    video_input = np.expand_dims(video_features[i:i + SEQ_LENGTH], axis=0)  # (1, 30, 1280)
    audio_input = np.expand_dims(audio_features[i:i + SEQ_LENGTH], axis=0)  # (1, 30, 128)

    # Run model inference
    predictions = model.predict([video_input, audio_input])  # Shape: (1, 2)

    # Get confidence scores
    confidence_scores = predictions[0]  # Extract first batch item

    # Determine the most confident prediction (real or fake)
    predicted_label = np.argmax(confidence_scores)  # 0 = real, 1 = fake
    predicted_confidence = np.max(confidence_scores)

    # Store confidence score for this frame
    confidences.append(predicted_confidence)

    print(f"Frames {i}-{i+SEQ_LENGTH-1}: {'Fake' if predicted_label == 1 else 'Real'} with confidence {predicted_confidence:.4f}")

# Ensure confidences list is not empty
if not confidences:
    raise ValueError("No confidence values were collected. Check feature extraction.")

# Find the sequence with the highest confidence
best_frame_idx = np.argmax(confidences)
best_frame = original_frames[best_frame_idx + SEQ_LENGTH // 2]  # Pick middle frame for display
best_confidence = confidences[best_frame_idx]

# Display the best frame
print(f"Most Confident Frame - Confidence: {best_confidence:.4f}")
cv2_imshow(best_frame)  # Use cv2_imshow() instead of cv2.imshow()