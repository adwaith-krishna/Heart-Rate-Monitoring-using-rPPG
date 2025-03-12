import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save_data(data_folder, test_size=0.2, random_state=42):
    """
    Split the combined data into training and testing sets incrementally and save them.
    """
    try:
        # Iterate through each chunk file
        for file in os.listdir(data_folder):
            if file.startswith("X_combined_chunk_"):
                print(f"Processing {file}...")

                # Load X chunk
                X_chunk = np.load(os.path.join(data_folder, file))

                # Load corresponding y chunk
                y_file = file.replace("X_combined_chunk_", "y_combined_chunk_")
                y_chunk = np.load(os.path.join(data_folder, y_file))

                # Split the chunk into training and testing sets
                X_train_chunk, X_test_chunk, y_train_chunk, y_test_chunk = train_test_split(
                    X_chunk, y_chunk, test_size=test_size, random_state=random_state
                )

                # Save the training and testing sets for this chunk
                chunk_id = file.split("_")[-1].split(".")[0]  # Extract chunk ID (e.g., 0, 1, 2, ...)
                np.save(os.path.join(data_folder, f"X_train_chunk_{chunk_id}.npy"), X_train_chunk)
                np.save(os.path.join(data_folder, f"X_test_chunk_{chunk_id}.npy"), X_test_chunk)
                np.save(os.path.join(data_folder, f"y_train_chunk_{chunk_id}.npy"), y_train_chunk)
                np.save(os.path.join(data_folder, f"y_test_chunk_{chunk_id}.npy"), y_test_chunk)

        print("Training and testing data saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function
if __name__ == "__main__":
    # Path to the folder containing preprocessed data
    data_folder = "preprocessed_data"

    # Split and save the data
    split_and_save_data(data_folder, test_size=0.2, random_state=42)