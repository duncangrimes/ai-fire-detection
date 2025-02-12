import cv2
import torch
import torchvision.transforms as transforms
import argparse
from PIL import Image
from models import shufflenetv2, nasnet_mobile_onfire

def load_model(model_name, weight_path, device):
    """ Load the model with the given weights. """
    if model_name == "shufflenetonfire":
        model = shufflenetv2.shufflenet_v2_x0_5(pretrained=False, layers=[4, 8, 4],
                                                output_channels=[24, 48, 96, 192, 64], num_classes=1)
    elif model_name == "nasnetonfire":
        model = nasnet_mobile_onfire.nasnetamobile(num_classes=1, pretrained=False)
    else:
        raise ValueError("Invalid model name. Choose 'shufflenetonfire' or 'nasnetonfire'.")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def get_transform(model_name):
    """ Define image transformations based on model type. """
    if model_name == 'shufflenetonfire':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif model_name == 'nasnetonfire':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def preprocess_image(image_path, transform, device):
    """ Load and transform an image for model inference. """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).float().unsqueeze(0).to(device)
    return image

def detect_fire(image_path, model, transform):
    """ Run fire detection on an image and return classification result. """
    image_tensor = preprocess_image(image_path, transform, device)
    output = model(image_tensor)
    prediction = torch.round(torch.sigmoid(output)).item()
    return "Fire" if prediction == 0 else "No Fire"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire Detection")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="shufflenetonfire", choices=["shufflenetonfire", "nasnetonfire"], help="Model selection")
    parser.add_argument("--weight", default="weights/shufflenet_ff.pt", help="Path to model weights")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    
    args = parser.parse_args()
    
    device = torch.device("cpu") if args.cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weight, device)
    transform = get_transform(args.model)
    
    result = detect_fire(args.image, model, transform)
    print(result)  # Outputs: "Fire" or "No Fire"