import os
import csv
import requests
import sys

# Configuration
DATASET_ROOT = r"d:\Disertatie\DataSet\demo\DataSet\archive\Testing"
API_URL = "http://127.0.0.1:8000/api/detect-tumor/neuronal-network/voting-label"
OUTPUT_CSV = "classification_report.csv"

def get_prediction(image_path):
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            data = response.json()
            # Extract winning class from the voting result
            return data.get('voting_result', {}).get('winning_class', 'Unknown')
        else:
            print(f"Error processing {image_path}: Status {response.status_code}")
            return "Error"
    except Exception as e:
        print(f"Exception processing {image_path}: {e}")
        return "Error"

def main():
    results = []
    
    # Check if dataset root exists
    if not os.path.exists(DATASET_ROOT):
        print(f"Dataset root not found: {DATASET_ROOT}")
        return

    print(f"Scanning directory: {DATASET_ROOT}")
    
    # Iterate through each class folder
    for class_name in os.listdir(DATASET_ROOT):
        class_dir = os.path.join(DATASET_ROOT, class_name)
        
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing class: {class_name}")
        
        # Iterate through images in the class folder
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            
            # Simple check for image extensions
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue
                
            print(f"  Processing {image_name}...", end='', flush=True)
            
            detected_class = get_prediction(image_path)
            
            print(f" Detected: {detected_class}")
            
            results.append({
                'clasa_reala': class_name,
                'clasa_detectata': detected_class,
                'file_name': image_name
            })

    # Write to CSV
    if results:
        print(f"Writing results to {OUTPUT_CSV}...")
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_name', 'clasa_reala', 'clasa_detectata']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print("Done.")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
