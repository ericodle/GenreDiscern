import sys
import json
import os
import math
import librosa

# Constants for audio processing.
SAMPLE_RATE = 22050  # Standard sample rate for GTZAN audio data.
SONG_LENGTH = 30  # Duration of each song clip in seconds.
SAMPLE_COUNT = SAMPLE_RATE * SONG_LENGTH  # Total number of samples per clip.

def mfcc_to_json(music_path, output_path, output_filename, mfcc_count=13, n_fft=2048, hop_length=512, seg_length=30):

    # Initialize the data dictionary to store extracted features and labels.
    extracted_data = {
        "mapping": [],  # List to map numeric labels to genre names.
        "labels": [],   # List to store numeric labels for each audio clip.
        "mfcc": []      # List to store extracted MFCCs.
    }
    
    # Calculate the number of samples per segment.
    seg_samples = seg_length * SAMPLE_RATE

    # Loop through each genre folder in the GTZAN dataset.
    for i, (folder_path, folder_name, file_name) in enumerate(os.walk(music_path)):
        if folder_path != music_path:
            # Extract genre label from folder path.
            genre_label = folder_path.split("/")[-1]
            extracted_data["mapping"].append(genre_label)
            print("\nProcessing: {}".format(genre_label))

            # Iterate over each audio file in the genre folder.
            for song_clip in file_name:
                file_path = os.path.join(folder_path, song_clip)
                try:
                    # Load the audio file.
                    audio_sig, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    # Handle loading errors.
                    print(f"Error loading file {file_path}: {e}")
                    continue
                
                # Check if the song is longer than 30 seconds.
                if len(audio_sig) >= SAMPLE_RATE * seg_length:
                    # Calculate the index of the middle of the song.
                    middle_index = len(audio_sig) // 2

                    # Define start and end indices for the segment.
                    segment_start = max(0, middle_index - (seg_samples // 2))
                    segment_end = min(len(audio_sig), middle_index + (seg_samples // 2))

                    # Extract MFCCs for the segment.
                    try:
                        mfcc = librosa.feature.mfcc(y=audio_sig[segment_start:segment_end], sr=sr, n_mfcc=mfcc_count, n_fft=n_fft, hop_length=hop_length)
                        # Transpose the MFCC matrix.
                        mfcc = mfcc.T
                    except Exception as e:
                        # Handle MFCC extraction errors.
                        print(f"Error computing MFCCs for {file_path}: {e}")
                        continue

                    # Append MFCCs and label to the data dictionary.
                    extracted_data["mfcc"].append(mfcc.tolist())
                    extracted_data["labels"].append(i - 1)
                    print("{}, segment:{}".format(file_path, segment_start, segment_end))
                else:
                    print(f"{file_path} is shorter than 30 seconds. Skipping...")

    # Write the extracted data to a JSON file.
    output_filename = output_filename + ".json"
    output_file_path = os.path.join(output_path, output_filename)
    try:
        with open(output_file_path, "w") as fp:
            json.dump(extracted_data, fp, indent=4)
            print(f"Successfully wrote data to {output_file_path}")
    except Exception as e:
        print(f"Error writing data to {output_file_path}: {e}")

if __name__ == "__main__":
    # Retrieve command-line arguments
    args = sys.argv[1:]

    # Check if there are command-line arguments
    if len(args) >= 1:
        music_path = args[0]  
    if len(args) >= 2:
        output_path = args[1] 
    if len(args) >= 3:
        output_filename = args[2]

    # Call the function with the specified arguments
    mfcc_to_json(music_path, output_path, output_filename)

