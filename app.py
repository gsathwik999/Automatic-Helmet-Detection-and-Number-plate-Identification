import cv2
import math
from ultralytics import YOLO
import os
import imagehash
from PIL import Image

# Initialize video capture
video_path = "media/sample3.mp4"  
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
model = YOLO("weights/best2.pt")  

# Define class names
classNames = ['helmet', 'license_plate', 'motorcyclist']

# Create directories to save snapshots
bike_images_dir = "violations"
number_plate_images_dir = "numberplate"
os.makedirs(bike_images_dir, exist_ok=True)
os.makedirs(number_plate_images_dir, exist_ok=True)

# Load existing image hashes
def load_existing_hashes(directory):
    hashes = set()
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = Image.open(image_path)
                hash_value = imagehash.phash(image)
                hashes.add(str(hash_value))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return hashes

# Maintain a set of saved image hashes
saved_helmet_hashes = load_existing_hashes(bike_images_dir)
saved_plate_hashes = load_existing_hashes(number_plate_images_dir)

# Function to calculate image hash
def get_image_hash(image):
    return str(imagehash.phash(Image.fromarray(image)))

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

frame_count = 0  # To track frame numbers

while True:
    success, img = cap.read()
    if not success:
        print("End of video or unable to read frame.")
        break

    results = model(img, stream=True)
    frame_count += 1

    height, width, _ = img.shape  # Get image dimensions for boundary checks

    for r in results:
        boxes = r.boxes

        motorcyclist_boxes = []
        helmet_boxes = []
        number_plate_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence score
            cls = int(box.cls[0])  # Class index

            # Classify bounding boxes
            if classNames[cls] == 'motorcyclist' and conf > 0.75:
                motorcyclist_boxes.append((x1, y1, x2, y2, conf))
            elif classNames[cls] == 'helmet' and conf < 0.6:
                helmet_boxes.append((x1, y1, x2, y2, conf))
            elif classNames[cls] == 'license_plate':
                number_plate_boxes.append((x1, y1, x2, y2, conf))

        # Process motorcyclists
        for mx1, my1, mx2, my2, mconf in motorcyclist_boxes:
            # Check for helmets in the motorcyclist bounding box
            helmet_found = False
            for hx1, hy1, hx2, hy2, hconf in helmet_boxes:
                if hx1 >= mx1 and hy1 >= my1 and hx2 <= mx2 and hy2 <= my2:
                    helmet_found = True
                    break

            if not helmet_found:  # No helmet detected
                # Check for a license plate inside the motorcyclist bounding box
                for nx1, ny1, nx2, ny2, nconf in number_plate_boxes:
                    if nx1 >= mx1 and ny1 >= my1 and nx2 <= mx2 and ny2 <= my2:
                        # Save the motorcyclist image
                        bike_image = img[my1:my2, mx1:mx2]
                        bike_hash = get_image_hash(bike_image)

                        # Save only if not already stored
                        if bike_hash not in saved_helmet_hashes:
                            bike_save_path = os.path.join(bike_images_dir, f"bike_frame_{frame_count}.jpg")
                            cv2.imwrite(bike_save_path, bike_image)
                            saved_helmet_hashes.add(bike_hash)
                            print(f"Saved bike image: {bike_save_path}")

                        # Expand the number plate bounding box
                        margin = 10  # Adjust the margin size as needed
                        ex1 = max(nx1 - margin, 0)  # Ensure within image boundaries
                        ey1 = max(ny1 - margin, 0)
                        ex2 = min(nx2 + margin, width)
                        ey2 = min(ny2 + margin, height)

                        # Save the expanded number plate image
                        number_plate_image = img[ey1:ey2, ex1:ex2]
                        plate_hash = get_image_hash(number_plate_image)

                        # Save only if not already stored
                        if plate_hash not in saved_plate_hashes:
                            number_plate_save_path = os.path.join(number_plate_images_dir, f"plate_frame_{frame_count}.jpg")
                            cv2.imwrite(number_plate_save_path, number_plate_image)
                            saved_plate_hashes.add(plate_hash)
                            print(f"Saved number plate image: {number_plate_save_path}")

    # Display the processed frame
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
