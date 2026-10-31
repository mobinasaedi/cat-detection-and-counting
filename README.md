# Cat Detection and Counting

An image-based computer vision application that detects whether cats are present and estimates their count using a pretrained YOLO object-detection model.

## Features

- Detects cat instances in an image.
- Reports whether at least one cat was detected.
- Estimates the number of detected cats.
- Supports a configurable confidence threshold.
- Saves an annotated image with bounding boxes.

## Method

1. Load a pretrained YOLO model.
2. Run inference on the input image.
3. Filter predictions to the COCO `cat` class.
4. Count the remaining detections.
5. Save the annotated image.

The repository uses pretrained weights; it does not claim that the model was trained from scratch.

## Structure

```text
cat-detection-and-counting/
├── cat_counter.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

The first run downloads the pretrained `yolo11n.pt` weights automatically.

## Usage

```bash
python cat_counter.py --image path/to/image.jpg
```

Customize the confidence threshold and output path:

```bash
python cat_counter.py --image path/to/image.jpg --confidence 0.4 --output outputs/result.jpg
```

## Output

The terminal reports whether a cat was detected and the number of detections. The annotated image is saved to the selected output path.

## Evaluation Considerations

Results depend on image resolution, occlusion, lighting, object scale, and the model's training distribution. The reported count is the number of model detections, not a guaranteed ground-truth count. A research extension could evaluate precision, recall, and counting error on a labeled test set.

## Technologies

Python, Ultralytics YOLO, OpenCV, object detection, computer vision.
