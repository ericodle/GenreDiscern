import json
import sys

def print_dimensionality(data):
    if isinstance(data, list):
        print(f"Dimensionality: {len(data)} elements in list")
    elif isinstance(data, dict):
        print(f"Dimensionality: {len(data)} key-value pairs in dictionary")
    else:
        print("Data is neither a list nor a dictionary")

def print_labels(data):
    if isinstance(data, dict):
        print(f"Labels: {list(data.keys())}")
    else:
        print("No labels found. Data is not a dictionary")

def print_detailed_info(data):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"\nKey: '{key}'")
            if isinstance(value, list):
                print(f"Type: List with {len(value)} elements")
                print("Sample elements:")
                for element in value[:3]:
                    print(f"  - {element}")
            elif isinstance(value, dict):
                print(f"Type: Dictionary with {len(value)} key-value pairs")
                print("Sample key-value pairs:")
                for k, v in list(value.items())[:3]:
                    print(f"  - {k}: {v}")
            else:
                print(f"Type: {type(value).__name__}")
                print(f"Value: {value}")
    elif isinstance(data, list):
        print(f"Type: List with {len(data)} elements")
        print("Sample elements:")
        for element in data[:3]:
            print(f"  - {element}")
    else:
        print(f"Data is of type: {type(data).__name__}")

def print_hierarchical_sizes(data, level=0):
    indent = "  " * level
    if isinstance(data, dict):
        print(f"{indent}Dictionary with {len(data)} key-value pairs")
        for key, value in data.items():
            print(f"{indent}Key: '{key}'")
            print_hierarchical_sizes(value, level + 1)
    elif isinstance(data, list):
        print(f"{indent}List with {len(data)} elements")
        # Don't print actual elements
        for item in data[:1]:
            print_hierarchical_sizes(item, level + 1)
    else:
        print(f"{indent}Example Value: {data}")

def main(json_path):
    # Load JSON file
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        print("JSON is valid")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    
    # Print dimensionality of the data
    print("\nDimensionality of the data:")
    print_dimensionality(data)
    
    # Print labels if data is a dictionary
    print("\nLabels in the data:")
    print_labels(data)
    
    # Extract specific information (example: all 'items')
    items = data.get('items', [])
    print("\nNumber of items:")
    print(len(items))
    
    # Print detailed information about the data
    print("\nDetailed information about the data:")
    print_detailed_info(data)
    
    # Print sizes of each hierarchical level
    print("\nHierarchical sizes of the data:")
    print_hierarchical_sizes(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python json_probe.py <path_to_json_file>")
    else:
        main(sys.argv[1])

