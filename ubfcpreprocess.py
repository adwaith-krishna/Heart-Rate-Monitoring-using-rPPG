import os
import cv2
import numpy as np
from scipy.interpolate import interp1d

# Step 1: Extract frames from video
def extract_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them as images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


# Step 2: Load ground truth heart rate data
def load_ground_truth(ground_truth_path):
    """
    Loads ground truth data from the text file and extracts heart rate.
    Assumes the first column is the PPG signal.
    """
    heart_rates = []
    with open(ground_truth_path, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()
            if len(columns) > 0:
                # Use the first column as the PPG signal (for example)
                ppg_value = float(columns[0])
                heart_rates.append(ppg_value)
    return heart_rates


# Step 3: Resample heart rate data to match the number of frames
def resample_heart_rates(heart_rates, target_length):
    """
    Resamples heart rate data to match the target length using interpolation.
    """
    original_indices = np.arange(len(heart_rates))
    target_indices = np.linspace(0, len(heart_rates) - 1, target_length)
    interpolator = interp1d(original_indices, heart_rates, kind='linear')
    resampled_heart_rates = interpolator(target_indices)
    return resampled_heart_rates


# Step 4: Detect and crop face from a frame
def detect_and_crop_face(frame):
    """
    Detects and crops the face region from a frame using Haar Cascade.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return frame[y:y+h, x:x+w]  # Return the face region
    return None


# Step 5: Preprocess a frame (resize and normalize)
def preprocess_frame(frame, size=(64, 64)):  # Reduced resolution to 64x64
    """
    Resizes and normalizes a frame.
    """
    frame = cv2.resize(frame, size)  # Resize
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    return frame.astype(np.float32)  # Use float32 to save memory


# Step 6: Create sequences of frames and corresponding heart rates
def create_sequences(frames, heart_rates, sequence_length=30):
    """
    Organizes frames into sequences and pairs them with heart rates.
    """
    X, y = [], []

    # Ensure there are enough frames and heart rates for at least one sequence
    if len(frames) < sequence_length or len(heart_rates) < sequence_length:
        print(f"Skipping subject: Not enough frames or heart rates for sequence length {sequence_length}.")
        return np.array([]), np.array([])

    for i in range(len(frames) - sequence_length):
        X.append(frames[i:i+sequence_length])  # Sequence of frames
        y.append(heart_rates[i+sequence_length])  # Corresponding heart rate

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # Use float32


# Step 7: Preprocess the entire dataset
def preprocess_ubfc_dataset(dataset_path, output_folder, sequence_length=30, batch_size=20):
    """
    Preprocesses the entire UBFC dataset.
    """
    subjects = os.listdir(dataset_path)
    batch_count = 0

    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        if not os.path.isdir(subject_path):
            continue

        print(f"Processing {subject}...")

        # Paths to video and ground truth files
        video_path = os.path.join(subject_path, "vid.avi")
        ground_truth_path = os.path.join(subject_path, "ground_truth.txt")

        # Step 1: Extract frames from video
        frames_folder = os.path.join(output_folder, subject, "frames")
        extract_frames(video_path, frames_folder)

        # Step 2: Load ground truth heart rate data
        heart_rates = load_ground_truth(ground_truth_path)

        # Step 3: Preprocess frames (detect face, resize, normalize)
        frames = []
        frame_files = sorted(os.listdir(frames_folder))
        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)

            face_roi = detect_and_crop_face(frame)
            if face_roi is not None:
                preprocessed_frame = preprocess_frame(face_roi)
                frames.append(preprocessed_frame)

        # Step 4: Resample heart rate data to match the number of frames
        if len(heart_rates) < len(frames):
            heart_rates = resample_heart_rates(heart_rates, len(frames))

        # Step 5: Synchronize frames and heart rates
        min_length = min(len(frames), len(heart_rates))
        frames = frames[:min_length]
        heart_rates = heart_rates[:min_length]

        # Debug: Print the number of frames and heart rates
        print(f"Frames: {len(frames)}, Heart Rates: {len(heart_rates)}")

        # Step 6: Create sequences in batches and save to disk
        num_sequences = len(frames) - sequence_length
        for i in range(0, num_sequences, batch_size):
            batch_frames = frames[i:i+batch_size+sequence_length]
            batch_heart_rates = heart_rates[i:i+batch_size+sequence_length]
            X_batch, y_batch = create_sequences(batch_frames, batch_heart_rates, sequence_length)

            if len(X_batch) > 0 and len(y_batch) > 0:
                # Save the batch to disk
                batch_folder = os.path.join(output_folder, "batches")
                if not os.path.exists(batch_folder):
                    os.makedirs(batch_folder)

                np.save(os.path.join(batch_folder, f"X_batch_{batch_count}.npy"), X_batch)
                np.save(os.path.join(batch_folder, f"y_batch_{batch_count}.npy"), y_batch)
                batch_count += 1

    print(f"Preprocessing complete. Saved {batch_count} batches to disk.")


# Main function to preprocess the dataset
if __name__ == "__main__":
    # Paths
    dataset_path = "datasets/UBFC2"  # Path to the UBFC dataset
    output_folder = "preprocessed_data"  # Folder to save preprocessed data

    # Preprocess the dataset
    preprocess_ubfc_dataset(dataset_path, output_folder, sequence_length=30, batch_size=20)