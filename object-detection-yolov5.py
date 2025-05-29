import torch
import cv2
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def detect_and_count_objects(img_path, model, output_img_path='output.jpg', report_path='report.csv')
    # Load image
    img = cv2.imread(img_path)
    if img is None
        raise FileNotFoundError(fImage not found at {img_path})
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(img_rgb)
    
    # Parse results as pandas DataFrame
    df = results.pandas().xyxy[0]  # columns xmin, ymin, xmax, ymax, confidence, class, name
    
    # Count objects per class
    counts = Counter(df['name'])
    
    # Draw bounding boxes and labels on original image
    for _, row in df.iterrows()
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']
        
        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=2)
        
        # Label text with confidence
        text = f{label} {conf.2f}
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), (0,255,0), -1)  # filled rectangle
        cv2.putText(img, text, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    
    # Save annotated image
    cv2.imwrite(output_img_path, img)
    
    # Save counts report as CSV
    report_df = pd.DataFrame(counts.items(), columns=['Object', 'Count'])
    report_df.to_csv(report_path, index=False)
    
    # Print counts summary
    print(Detected objects count)
    for obj, cnt in counts.items()
        print(f{obj} {cnt})
    
    # Show image with matplotlib
    img_rgb_annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,8))
    plt.imshow(img_rgb_annotated)
    plt.axis('off')
    plt.title(Detected Objects with Bounding Boxes)
    plt.show()

# --- Main execution ---

if __name__ == '__main__'
    # Load pre-trained YOLOv5s model
    model = torch.hub.load('ultralyticsyolov5', 'yolov5s', pretrained=True)
    
    # Specify input image path
    input_image = 'your_image.jpg'  # Replace with your image file
    
    # Run detection, counting and save results
    detect_and_count_objects(
        img_path=input_image,
        model=model,
        output_img_path='output_detected.jpg',
        report_path='detection_report.csv'
    )
