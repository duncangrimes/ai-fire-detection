import cv2
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import shufflenetv2
import os
from dotenv import load_dotenv
import resend
from datetime import datetime
import sys

# Load environment variables from .env


# Load the shufflenetonfire model
def load_model(device):
    model = shufflenetv2.shufflenet_v2_x0_5(
        pretrained=False, 
        layers=[4, 8, 4],
        output_channels=[24, 48, 96, 192, 64],
        num_classes=1
    )
    model.load_state_dict(torch.load("weights/shufflenet_ff.pt", map_location=device))
    model.eval()
    model.to(device)
    return model

# Define image transformations
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# Preprocess a frame (resize, convert colors, apply transform)
def preprocess_frame(frame, transform, device):
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).float().unsqueeze(0).to(device)
    return tensor

# Send email notification
def send_email(user_email, timestamp):
    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
    
    subject = f"Fire Detected at {time_str} on {date_str}"
    html_content = "<strong>There is a fire on your camera.</strong>"

    params = {
        "from": "AI Fire Detection <ai-fire-detection@duncangrimes.com>",
        "to": [user_email],
        "subject": subject,
        "html": html_content,
    }
    email = resend.Emails.send(params)
    print(email)

# Run inference on a frame and return "fire" or "no fire"
def detect_fire(frame, model, transform, device, user_email, timestamp):
    tensor = preprocess_frame(frame, transform, device)
    output = model(tensor)
    prediction = torch.round(torch.sigmoid(output)).item()
    # Original logic: 0 -> fire, 1 -> no fire
    if prediction == 0:
        send_email(user_email, timestamp)
        print("🔥 Fire detected. Program terminating...")
        sys.exit(0) 
    else:
        return "no fire"

def main():
    if len(sys.argv) != 2:
        print("Usage: python detect-fire.py <your_email>")
        sys.exit(1)
    
    user_email = sys.argv[1]

    load_dotenv()
    resend.api_key = os.getenv("RESEND")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = get_transform()

    # Open webcam (0 = default camera) and set 1080p resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    last_capture_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        # Capture and process frame every 2 seconds
        if current_time - last_capture_time >= 2:
            result = detect_fire(frame, model, transform, device, user_email, current_time)
            print(result)
            last_capture_time = current_time

        # Display the live feed (press 'q' to quit)
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()