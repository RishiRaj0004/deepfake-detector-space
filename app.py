import gradio as gr
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load model architecture and weights
model = models.resnext101_32x8d(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="RishiRaj0004/deepfake-detector-model",
    filename="resnext101_deepfake_faces.pth",
    cache_dir="./model"
)
state_dict = torch.load(model_path, map_location=device)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 2. Face-aware frame extraction
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def extract_faces_from_video(video_path, frame_count=10, output_size=(128,128)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // frame_count, 1)
    faces = []
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(dets) > 0:
            x, y, w, h = max(dets, key=lambda r: r[2]*r[3])
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, output_size)
            faces.append(face)
    cap.release()
    return faces

# 3. Prediction function
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def predict_deepfake(video_path):
    faces = extract_faces_from_video(video_path)
    if len(faces) == 0:
        return "No face detected"
    batch = [preprocess(face) for face in faces]
    batch = torch.stack(batch).unsqueeze(0).to(device)  # shape (1, N, C, H, W)
    # Flatten batch dimension
    batch = batch.view(-1, 3, 128, 128)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    avg = probs.mean(axis=0)
    label = "Fake" if avg[1] > avg[0] else "Real"
    return f"{label} ({avg.max():.1%})"

# 4. Gradio interface
iface = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Text(label="Prediction"),
    title="Deepfake Detector",
    description="Upload a video and see if it’s real or fake."
)

if __name__ == "__main__":
    iface.launch()