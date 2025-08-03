# YOLOv8 Object Detection Pipeline

A Python-based pipeline for performing object detection on images using a pre-trained YOLOv8 model. This script takes an image path as input and produces a JSON file containing the detected objects, their classes, confidence scores, and bounding box coordinates.

---

## Setup Instructions

Follow these steps to set up the local environment required to run the script.

1.  **Clone the Repository**
    ```bash
    git clone <your-github-repository-url>
    cd <repository-folder-name>
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

To execute the object detection pipeline, run the `detect.py` script from the terminal. You must provide the path to an input image using the `--image` command-line argument.

```bash
python detect.py --image path/to/your/image.jpg
````

The script will process the image and save the detection results to the location specified in `config.yaml` (by default, `sample_output/detections.json`).

-----

## Project Structure

The repository is organized as follows:

```
.
├── detect.py               # Main detection script
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── sample_output/          # Directory for output files
│   └── detections.json     # Sample output file
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
```

-----

## Configuration

Key parameters can be modified in the `config.yaml` file:

  - `model_path`: The path to the YOLOv8 model file (e.g., `yolov8n.pt`).
  - `confidence_threshold`: The minimum score for a detection to be considered valid (0.0 to 1.0).
  - `output_path`: The path where the resulting JSON file will be saved.

-----

## Output Format

The script generates a JSON file containing a list of all detected objects. Each object in the list is a dictionary with the following structure:

```json
[
    {
        "class_name": "person",
        "confidence": 0.9234,
        "bounding_box": {
            "x1": 479,
            "y1": 224,
            "x2": 640,
            "y2": 480
        }
    }
]
```

```
```
