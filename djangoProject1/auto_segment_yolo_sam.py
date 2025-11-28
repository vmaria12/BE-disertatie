import os
from ultralytics.data.annotator import auto_annotate

def run_auto_segmentation():
    # Base directory of the project
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to your custom YOLO model
    # Adjust this path if your model is located elsewhere
    det_model_path = os.path.join(base_dir, 'Models', 'Yolo', 'yolo_v12.pt')

    # SAM model name (will be downloaded automatically by ultralytics if not present)
    # You can also specify a local path if you have downloaded it manually
    sam_model_name = "sam2.1_b.pt"

    # Directory containing the images you want to annotate
    # Defaulting to 'assets' in the current directory. Change this to your image folder path.
    data_dir = os.path.join(base_dir, 'assets')

    # Directory where the annotations will be saved
    output_dir = os.path.join(base_dir, 'assets_auto_annotate_labels')

    # Check if the YOLO model exists
    if not os.path.exists(det_model_path):
        print(f"Error: YOLO model not found at {det_model_path}")
        print("Please check the path to your YOLO model.")
        return

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist.")
        print(f"Creating '{data_dir}'... Please place your images inside this folder.")
        os.makedirs(data_dir, exist_ok=True)
        # We can stop here or let auto_annotate handle an empty dir (it might complain)
        return

    print("Starting auto-annotation process...")
    print(f"Detection Model: {det_model_path}")
    print(f"Segmentation Model: {sam_model_name}")
    print(f"Input Images: {data_dir}")
    print(f"Output Labels: {output_dir}")

    try:
        # Run the auto_annotate function
        # This uses the YOLO model to detect objects and SAM to generate segmentation masks
        auto_annotate(
            data=data_dir,
            det_model=det_model_path,
            sam_model=sam_model_name,
            output_dir=output_dir
        )
        print("Successfully completed auto-annotation.")
        print(f"Labels saved in: {output_dir}")

    except Exception as e:
        print(f"An error occurred during annotation: {e}")

if __name__ == "__main__":
    run_auto_segmentation()
