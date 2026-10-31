
from pathlib import Path
import cv2
from ultralytics import YOLO

IMAGE_PATH = r"F:\computer vision projects\computer_vision_github_projects\cat-detection-and-counting\cat_detection.jpg"
OUTPUT_PATH = "cat_detection_result.jpg"
CONFIDENCE = 0.35

def detect_and_count_cats():
    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO("yolo11n.pt")

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        classes=[15],
        verbose=False
    )

    result = results[0]
    count = 0 if result.boxes is None else len(result.boxes)

    annotated_image = result.plot()

    cv2.imwrite(OUTPUT_PATH, annotated_image)

    if count > 0:
        print("Cat detected: Yes")
    else:
        print("Cat detected: No")

    print(f"Number of cats detected: {count}")
    print(f"Result saved to: {Path(OUTPUT_PATH).resolve()}")

if __name__ == "__main__":
    detect_and_count_cats()
