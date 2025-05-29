# Object Detection with YOLOv5 - Detection, Counting, and Reporting

This is a beginner-friendly Python project for detecting objects in images using a pre-trained YOLOv5 model. The project includes features for drawing bounding boxes and labels on detected objects, counting the number of each object class, and saving a detection report as a CSV file.

---

## Features

- Detect multiple objects in an input image using YOLOv5 pre-trained weights (`yolov5s`).
- Draw bounding boxes and class labels on detected objects.
- Count the number of detected objects per class.
- Save a CSV report summarizing detected object counts.
- Save the output image with annotations.
- Display the annotated image using matplotlib.

---

## Requirements

- Python 3.7 or higher
- PyTorch
- OpenCV
- Pandas
- Matplotlib

You can install dependencies with:

```bash
pip install -r requirements.txt
