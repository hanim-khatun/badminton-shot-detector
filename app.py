
import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import cv2
import tempfile
import base64

st.set_page_config(page_title="Badminton Shot Detector", page_icon="🏸", layout="centered")
st.title("🏸 Badminton Shot Detector")
st.write("Upload a video — AI will detect the shot type!")

SHOTS = ["Block", "Clear", "Drive", "Drop", "Lift", "Net", "Smash"]

@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 7)
    model.load_state_dict(torch.load("badminton_model_7shots.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    counts = {shot: 0 for shot in SHOTS}
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor).argmax(1).item()
            counts[SHOTS[pred]] += 1
        frame_count += 1
    cap.release()
    return counts

uploaded = st.file_uploader("📁 Upload your video", type=["mp4"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    video_b64 = base64.b64encode(open(tmp_path, "rb").read()).decode()
    st.markdown(
        f"""
        <video width="100%" height="300" controls style="max-height:300px; object-fit:contain;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("AI is analyzing the video..."):
        counts = predict_video(tmp_path)

    total = sum(counts.values())
    result = max(counts, key=counts.get)

    st.success(f"## Detected Shot: {result} 🏸")
    st.write("### Shot Breakdown:")
    for shot, count in sorted(counts.items(), key=lambda x: -x[1]):
        if total > 0:
            pct = count/total
            st.write(f"**{shot}**: {count} frames ({pct*100:.1f}%)")
            st.progress(pct)
