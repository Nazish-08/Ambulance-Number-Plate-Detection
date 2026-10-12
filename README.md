# Ambulance Number Plate Detection using YOLO11n

A computer vision project for detecting ambulance number plates using a custom-trained YOLO11n object detection model. The project includes manual annotation using CVAT, COCO-to-YOLO conversion, dataset splitting, model training, and image inference.

## Project Overview

The goal of this project is to detect number plates in ambulance images using a YOLO object detection model.

The model identifies the location of number plates and draws bounding boxes around detected regions. Reading the actual characters on a number plate requires an additional OCR system, which is not included in this detection model.

## Features

- Ambulance image dataset preparation
- Manual number plate annotation using CVAT
- COCO 1.0 annotation export
- COCO-to-YOLO annotation conversion
- Train, validation, and test dataset split
- YOLO11n model training
- Number plate detection on input images
- Prediction image generation

## Tech Stack

- Python
- Ultralytics YOLO11n
- PyTorch
- OpenCV
- CVAT
- COCO JSON
- YOLO annotation format

## Dataset Preparation

The dataset was prepared using the following workflow:

1. Downloaded ambulance images from Roboflow.
2. Selected images containing visible ambulance number plates.
3. Uploaded images to CVAT.
4. Manually annotated number plates using the `number_plate` label.
5. Exported annotations in COCO 1.0 format.
6. Converted COCO annotations to YOLO format.
7. Split annotated images into training, validation, and test sets.

### Dataset Split

| Split | Images |
|---|---:|
| Training | 61 |
| Validation | 8 |
| Test | 8 |
| **Total annotated images** | **77** |

The split is approximately 80:10:10. The small dataset size should be considered when interpreting model performance.

## Project Structure

```text
Ambulance-Number-Plate-Detection/
├── train.py
├── test_model.py
├── convert_coco_to_yolo.py
├── requirements.txt
├── .gitignore
├── README.md
├── screenshots/
└── weights/
```

The dataset and training outputs are stored locally and are not included in the repository by default.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Ambulance-Number-Plate-Detection.git
cd Ambulance-Number-Plate-Detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Configuration

Create a `data.yaml` file in the dataset root folder:

```yaml
path: C:/path/to/Ambulance_YOLO_Annotated_Only
train: images/train
val: images/val
test: images/test

names:
  0: number_plate
```

Replace the example path with the actual dataset location on your computer.

## Model Training

The project uses the pretrained YOLO11n model as the starting point.

Run:

```bash
python train.py
```

The training script is configured for up to 50 epochs, an image size of 640 pixels, and a batch size of 4.

The best model is saved at:

```text
runs/ambulance_plate/weights/best.pt
```

The exact output location depends on the training configuration.

## Run Inference

Run:

```bash
python test_model.py
```

When prompted, enter the full path of an ambulance image.

The model predicts number plate bounding boxes and saves an annotated prediction image.

## Training Results

The training log reported the following validation metrics for the best model:

| Metric | Result |
|---|---:|
| Best epoch | 19 |
| Precision | 96.1% |
| Recall | 100% |
| mAP@50 | 99.5% |
| mAP@50–95 | 47.3% |

These metrics were obtained on a validation set of only 8 images. They should not be interpreted as a guarantee of performance on unseen real-world images. Testing on a larger, diverse dataset is recommended.

## Screenshots

Add project screenshots to the `screenshots/` directory and update the image paths below.

### 1. Training Results

```markdown
![YOLO Training Results](screenshots/training-results.png)
```

### 2. Number Plate Detection

```markdown
![Number Plate Detection](screenshots/number-plate-detection.png)
```

### 3. Prediction Output

```markdown
![Prediction Output](screenshots/prediction-output.png)
```

The screenshots will appear after you add the corresponding image files to the repository.

## Limitations

- The dataset currently contains only 77 annotated images.
- Validation metrics are based on 8 images.
- Detection performance may vary with image quality, lighting, distance, and viewing angle.
- The model detects number plate regions but does not read plate characters.
- An OCR system can be integrated as a future enhancement.

## Future Improvements

- Collect and annotate more ambulance images.
- Evaluate performance on a larger independent test set.
- Improve detection under different lighting and camera angles.
- Integrate OCR for reading number plate text.
- Develop a web interface for image and video inference.

## License

Choose an appropriate license after confirming that the dataset and any third-party assets can be redistributed under its terms.

## Author

**Nazish Shaikh**

GitHub: [Nazish-08](https://github.com/Nazish-08)
