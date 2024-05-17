import json
import csv
import sys

def json_to_csv(json_file, csv_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return

    mfcc_data = data['mfcc']
    labels = data['labels']
    mappings = data['mapping']

    # Calculate the maximum number of MFCC coefficients for padding
    max_length = max(len(seq) for seq in mfcc_data)

    # Add padding to sequences with fewer MFCC coefficients
    padded_mfcc_data = [seq + [[0]*13] * (max_length - len(seq)) for seq in mfcc_data]

    # Flatten the padded MFCC data
    flattened_mfcc_data = []
    for seq in padded_mfcc_data:
        for sample in seq:
            flattened_mfcc_data.append(sample)

    # Convert data to the structure needed for csv
    formatted_data = [(mappings[lbl], lbl) + tuple(sample) for lbl, sample in zip(labels, flattened_mfcc_data)]

    # Write to CSV file
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['genre', 'label'] + [f'mfcc{i}' for i in range(13)])
        csv_writer.writerows(formatted_data)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python json_to_csv.py <input_json_file> <output_csv_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    csv_file = sys.argv[2]

    json_to_csv(json_file, csv_file)

