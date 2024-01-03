import torch
import os
import cv2
from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weights = ResNet50_QuantizedWeights.DEFAULT
model = resnet50(weights=weights, quantize=True)
model.eval()

preprocess = weights.transforms()
folder_path = "hw4/python/data/box_imagenet/478_carton"
count = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img = read_image(os.path.join(folder_path, filename))
        batch = preprocess(img).unsqueeze(0)

        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        category_name = weights.meta["categories"][class_id]

        if category_name == "carton":
            count += 1

        print(f"{filename}: {category_name}")

total_accuracy = count / len(os.listdir(folder_path))
print(f"Total accuracy [Validation]: {100 * total_accuracy}%")

video_path = 'hw4/python/data/carton_video2.mp4'
output_directory = 'hw4/python/data/output_frames'
os.makedirs(output_directory, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_path = os.path.join(output_directory, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(output_path, frame)
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

count = 0
for filename in os.listdir(output_directory):
    if filename.endswith(".jpg"):
        img = read_image(os.path.join(output_directory, filename))
        batch = preprocess(img).unsqueeze(0)

        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        category_name = weights.meta["categories"][class_id]

        if category_name == "carton":
            count += 1

        print(f"{filename}: {category_name}")

total_accuracy = count / len(os.listdir(output_directory))
print(f"Total accuracy [Video]: {100 * total_accuracy}%")