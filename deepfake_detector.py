import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

class DeepfakeDetector:
    def __init__(self, model_path='resnet50_deepfake_finetuned_continue.pth'):
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=False)
        # Modify the final layer to match the saved model structure
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def is_deepfake(self, image_url, threshold=0.5):
        try:
            # Download and open image
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Transform image
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                deepfake_prob = probabilities[0][1].item()
            
            return deepfake_prob > threshold, deepfake_prob
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            return False, 0.0

    def process_image_batch(self, image_urls):
        """Process a batch of images and return only the ones classified as deepfakes"""
        deepfake_results = []
        for url in image_urls:
            is_fake, confidence = self.is_deepfake(url)
            if is_fake:
                deepfake_results.append({
                    'url': url,
                    'confidence': confidence * 100  # Convert to percentage
                })
        return deepfake_results 