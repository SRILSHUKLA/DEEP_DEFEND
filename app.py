import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import tempfile
from name import detect_top_celebrity_name
from urls import get_google_image_urls, get_youtube_video_urls, get_vimeo_video_urls, get_dailymotion_video_urls
from flask_cors import CORS
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
import base64
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import traceback
import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs
import yt_dlp
import subprocess

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB max upload
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'celebrity_recognition_app')  # for session

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n=== Using device: {device} ===")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    # Set memory growth to prevent OOM errors
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# Default credentials path - no need to upload repeatedly
DEFAULT_CREDENTIALS_PATH = os.getenv('DEFAULT_CREDENTIALS_PATH', 'new_credentials.json')

# Default number of items to load
DEFAULT_NUM_IMAGES = 30
DEFAULT_NUM_VIDEOS = 30

# Custom model class to match the saved architecture
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        print("Initializing DeepfakeDetector model...")
        # Load pretrained ResNet50
        self.resnet = resnet50(weights='IMAGENET1K_V2')
        print("ResNet50 initialized with pretrained weights")
        # Remove the original fc layer
        self.resnet.fc = nn.Identity()
        # Add the custom fc layers to match the saved model structure
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),   # First linear layer
            nn.ReLU(),              # First activation
            nn.Linear(512, 2)       # Final classification layer
        )
        print("Model architecture created with matching fc layers")
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

# Initialize deepfake detection model
def load_deepfake_model():
    try:
        print("\n=== Starting Deepfake Model Loading ===")
        print("Creating model instance...")
        model = DeepfakeDetector()
        
        # Check if model file exists
        model_path = 'resnet50_deepfake_finetuned_continue.pth'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        print(f"Found model file at {model_path}")
        print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Load state dict with error handling
        try:
            print("Loading state dict...")
            state_dict = torch.load(model_path, map_location=device)
            print("Successfully loaded state dict")
            print("State dict keys:")
            for key in state_dict.keys():
                if key.startswith('fc'):
                    print(f"  {key}")
        except Exception as e:
            print(f"Error loading state dict: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return None
        
        # Handle potential state dict key mismatches
        if 'module.' in list(state_dict.keys())[0]:
            print("Converting state dict keys from DataParallel format")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            print("Loading state dict into model...")
            # Convert fc layer keys to match Sequential module format
            fc_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('fc.'):
                    # Convert fc.0.weight -> 0.weight, fc.3.weight -> 2.weight
                    new_key = k.replace('fc.', '')
                    if new_key.startswith('0.'):
                        fc_state_dict[new_key] = v
                    elif new_key.startswith('3.'):
                        fc_state_dict[new_key.replace('3.', '2.')] = v
            
            print("Converted fc layer keys:")
            for k in fc_state_dict.keys():
                print(f"  {k}")
                
            model.fc.load_state_dict(fc_state_dict)
            print("Successfully loaded fc layer weights")
        except Exception as e:
            print(f"Error loading state dict into model: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return None
            
        # Move model to GPU if available
        model = model.to(device)
        model.eval()
        print(f"Model moved to {device} and set to eval mode!")
        print("=== Deepfake Model Loading Complete ===\n")
        return model
    except Exception as e:
        print(f"Error in load_deepfake_model: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        return None

# Load the model at startup
print("\n=== Initializing Deepfake Detection Model ===")
deepfake_model = load_deepfake_model()
if deepfake_model is None:
    print("WARNING: Failed to load deepfake detection model. Deepfake detection will be skipped.")
else:
    print("Deepfake detection model loaded successfully!")
print("=== Model Initialization Complete ===\n")

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_deepfake(image_path):
    """
    Check if an image is a deepfake using the loaded model
    Returns True if the image is likely a deepfake (probability > 0.8)
    """
    if deepfake_model is None:
        print("Deepfake model not loaded, skipping detection")
        return False  # Skip detection if model failed to load
        
    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Move tensor to GPU if available
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = deepfake_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fake_probability = probabilities[0][1].item()  # Probability of being fake
            
        print(f"Deepfake probability: {fake_probability:.2f}")
        # Only return True if we're very confident it's a deepfake (probability > 0.8)
        return fake_probability > 0.8
    except Exception as e:
        print(f"Error in deepfake detection: {str(e)}")
        return False  # Skip detection on error

def verify_celebrity_and_deepfake(image_url, celebrity_name):
    """
    Verify if the celebrity is present in the image and if it's a deepfake
    Returns: (is_celebrity_present, is_deepfake)
    """
    try:
        # Download the image
        print(f"Downloading image: {image_url}")
        response = requests.get(image_url, timeout=5)
        if response.status_code != 200:
            print(f"Failed to download image: {image_url}, status code: {response.status_code}")
            return False, False
        
        # Save to temporary file
        temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_verify.jpg')
        with open(temp_img_path, 'wb') as f:
            f.write(response.content)
        
        try:
            # First check if celebrity is in the image
            print(f"Running face recognition on: {image_url}")
            detected_name = detect_top_celebrity_name(temp_img_path, DEFAULT_CREDENTIALS_PATH)
            print(f"Detected name: {detected_name}, Expected: {celebrity_name}")
            
            if detected_name != celebrity_name:
                print("Celebrity not found in image")
                return False, False
            
            # If celebrity is found, check if it's a deepfake
            print(f"Checking if image is a deepfake: {image_url}")
            is_fake = is_deepfake(temp_img_path)
            print(f"Deepfake detection result: {is_fake}")
            
            return True, is_fake
            
        finally:
            # Clean up
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
    except Exception as e:
        print(f"Error verifying image {image_url}: {str(e)}")
        return False, False

def extract_video_id(url):
    """Extract video ID from various video platform URLs"""
    if 'youtube.com' in url or 'youtu.be' in url:
        if 'youtube.com' in url:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        else:  # youtu.be
            return url.split('/')[-1]
    elif 'vimeo.com' in url:
        return url.split('/')[-1]
    elif 'dailymotion.com' in url:
        return url.split('/')[-1]
    return None

def download_video(video_url, output_path):
    """
    Download video using yt-dlp
    Returns True if successful, False otherwise
    """
    try:
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for faster processing
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return True
    except Exception as e:
        print(f"Error downloading video {video_url}: {str(e)}")
        return False

def download_video_frames(video_url, num_frames=100):
    """
    Download video and extract frames
    Returns list of (frame, timestamp) tuples
    """
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"Could not extract video ID from URL: {video_url}")
            return []

        # Create temporary file for video
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_video_{video_id}.mp4')
        
        # Download video
        print(f"Downloading video: {video_url}")
        if not download_video(video_url, temp_video_path):
            print(f"Failed to download video: {video_url}")
            return []
        
        # Extract frames
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {temp_video_path}")
            return []

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            print(f"Invalid video properties: frames={total_frames}, fps={fps}")
            return []
        
        # Calculate frame intervals
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                timestamp = idx / fps
                frames.append((pil_image, timestamp))
        
        cap.release()
        
        # Clean up
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from video: {str(e)}")
        return []

def is_video_deepfake(image_path):
    """
    Check if a video frame is a deepfake using the loaded model
    Returns True if the frame is likely a deepfake (probability > 0.7)
    """
    if deepfake_model is None:
        print("Deepfake model not loaded, skipping detection")
        return False  # Skip detection if model failed to load
        
    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Move tensor to GPU if available
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = deepfake_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fake_probability = probabilities[0][1].item()  # Probability of being fake
            
        print(f"Video frame deepfake probability: {fake_probability:.2f}")
        # Return True if we're confident it's a deepfake (probability > 0.7)
        return fake_probability > 0.7
    except Exception as e:
        print(f"Error in video frame deepfake detection: {str(e)}")
        return False  # Skip detection on error

def verify_video_frames(video_url, celebrity_name, num_frames=15):
    """
    Verify if the celebrity appears in multiple frames of the video
    Returns: (is_valid_video, is_deepfake, deepfake_timestamps)
    """
    video_is_deepfake = False
    deepfake_timestamps = []
    try:
        print(f"\nVerifying video: {video_url}")
        # Extract frames from video
        frames = download_video_frames(video_url, num_frames)
        if not frames:
            print(f"No frames extracted from video: {video_url}")
            return False, False, []
            
        print(f"Extracted {len(frames)} frames for verification")
        
        # First pass: Check all frames for celebrity presence
        print("\nFirst pass: Checking frames for celebrity presence...")
        frames_with_celebrity = []
        
        # Process frames in batches to avoid memory issues
        batch_size = 5  # Reduced batch size since we have fewer frames
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            print(f"Processing frames {i+1} to {min(i+batch_size, len(frames))} of {len(frames)}")
            
            for frame, timestamp in batch:
                # Save frame temporarily
                temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{timestamp}.jpg')
                frame.save(temp_frame_path)
                
                try:
                    # Check if celebrity is in frame
                    detected_name = detect_top_celebrity_name(temp_frame_path, DEFAULT_CREDENTIALS_PATH)
                    print(f"Frame at {timestamp:.2f}s: Detected {detected_name}, Expected {celebrity_name}")
                    
                    if detected_name == celebrity_name:
                        frames_with_celebrity.append((frame, timestamp))
                finally:
                    # Clean up
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
        
        # Check if we have enough frames with the celebrity (at least 30%)
        min_valid_frames = int(num_frames * 0.3)  # For 15 frames, this means at least 5 frames
        if len(frames_with_celebrity) < min_valid_frames:
            print(f"\nNot enough frames with celebrity: {len(frames_with_celebrity)}/{min_valid_frames} required")
            return False, False, []
            
        print(f"\nFound {len(frames_with_celebrity)} frames with celebrity ({len(frames_with_celebrity)/len(frames)*100:.1f}%)")
        
        # Second pass: Check only frames with celebrity for deepfakes
        print("\nSecond pass: Checking frames with celebrity for deepfakes...")
        deepfake_frames = 0
        total_frames_checked = 0
        
        for frame, timestamp in frames_with_celebrity:
            # Save frame temporarily
            temp_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{timestamp}.jpg')
            frame.save(temp_frame_path)
            
            try:
                # Check if frame is a deepfake using the video-specific threshold
                is_fake = is_video_deepfake(temp_frame_path)
                total_frames_checked += 1
                if is_fake:
                    deepfake_frames += 1
                    deepfake_timestamps.append(timestamp)
                    print(f"Frame at {timestamp:.2f}s: Confirmed deepfake")
                else:
                    print(f"Frame at {timestamp:.2f}s: Not a deepfake")
            finally:
                # Clean up
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
        
        # Video is deepfake if ANY frame is a deepfake
        video_is_deepfake = deepfake_frames > 0
        
        print(f"\nVideo verification results for {video_url}:")
        print(f"Total frames analyzed: {len(frames)}")
        print(f"Frames with celebrity: {len(frames_with_celebrity)}/{len(frames)} ({len(frames_with_celebrity)/len(frames)*100:.1f}%)")
        print(f"Deepfake frames: {deepfake_frames}/{total_frames_checked}")
        if deepfake_frames > 0:
            print("Deepfake frames found at timestamps:")
            for ts in deepfake_timestamps:
                print(f"- {ts:.2f} seconds")
        print(f"Final verdict: Valid=True, Deepfake={video_is_deepfake}\n")
        
        return True, video_is_deepfake, deepfake_timestamps
        
    except Exception as e:
        print(f"Error verifying video frames: {str(e)}")
        return False, False, []

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = None
    try:
        # Save the uploaded file to a temporary location
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        # Read the image file for base64 encoding before processing
        with open(temp_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode()
        
        # Process with name.py - detect celebrity
        print("Running initial face recognition on uploaded image")
        celebrity_name = detect_top_celebrity_name(temp_path, DEFAULT_CREDENTIALS_PATH)
        print(f"Initial celebrity detection result: {celebrity_name}")
        
        # Check if there was an error
        if celebrity_name.startswith("Error:"):
            raise Exception(celebrity_name)
        
        # Check if no matches were found
        if celebrity_name == "NO_MATCHES_FOUND":
            os.remove(temp_path)
            return jsonify({'error': 'No matches found'}), 404
        
        # Get results from urls.py functions
        print(f"Fetching images for: {celebrity_name}")
        all_images = get_google_image_urls(celebrity_name, total_num=DEFAULT_NUM_IMAGES * 2)
        print(f"Found {len(all_images)} images")
        
        # Get video URLs
        print(f"\nFetching videos for: {celebrity_name}")
        youtube_videos = get_youtube_video_urls(celebrity_name, max_results=DEFAULT_NUM_VIDEOS)
        vimeo_videos = get_vimeo_video_urls(celebrity_name, max_results=DEFAULT_NUM_VIDEOS)
        dailymotion_videos = get_dailymotion_video_urls(celebrity_name, max_results=DEFAULT_NUM_VIDEOS)
        
        print(f"Found {len(youtube_videos)} YouTube videos")
        print(f"Found {len(vimeo_videos)} Vimeo videos")
        print(f"Found {len(dailymotion_videos)} Dailymotion videos")
        
        # Verify images
        verified_images = []
        for image_url in all_images:
            is_celebrity, is_deepfake = verify_celebrity_and_deepfake(image_url, celebrity_name)
            if is_celebrity and is_deepfake:
                verified_images.append(image_url)
                print(f"Verified deepfake image {len(verified_images)}/10: {image_url}")
                result['images'] = verified_images
                if len(verified_images) >= 10:
                    break
        
        # Verify videos
        print("\nStarting video verification...")
        verified_youtube = []
        verified_vimeo = []
        verified_dailymotion = []
        
        print("\nVerifying YouTube videos...")
        for video_url in youtube_videos:
            print(f"\nProcessing YouTube video: {video_url}")
            is_valid, is_deepfake, timestamps = verify_video_frames(video_url, celebrity_name)
            if is_valid and is_deepfake:
                verified_youtube.append({
                    'url': video_url,
                    'timestamps': timestamps
                })
                print(f"Verified deepfake YouTube video: {video_url}")
                
        print("\nVerifying Vimeo videos...")
        for video_url in vimeo_videos:
            print(f"\nProcessing Vimeo video: {video_url}")
            is_valid, is_deepfake, timestamps = verify_video_frames(video_url, celebrity_name)
            if is_valid and is_deepfake:
                verified_vimeo.append({
                    'url': video_url,
                    'timestamps': timestamps
                })
                print(f"Verified deepfake Vimeo video: {video_url}")
                
        print("\nVerifying Dailymotion videos...")
        for video_url in dailymotion_videos:
            print(f"\nProcessing Dailymotion video: {video_url}")
            is_valid, is_deepfake, timestamps = verify_video_frames(video_url, celebrity_name)
            if is_valid and is_deepfake:
                verified_dailymotion.append({
                    'url': video_url,
                    'timestamps': timestamps
                })
                print(f"Verified deepfake Dailymotion video: {video_url}")
        
        print(f"\nVerification Summary:")
        print(f"Total verified deepfake images: {len(verified_images)}")
        print(f"Total verified deepfake videos:")
        print(f"- YouTube: {len(verified_youtube)}")
        print(f"- Vimeo: {len(verified_vimeo)}")
        print(f"- Dailymotion: {len(verified_dailymotion)}")
        
        # Create a result that matches the frontend's expected format
        result = {
            'id': str(hash(celebrity_name)),
            'imageUrl': f"data:image/jpeg;base64,{image_base64}",
            'confidence': 95.5,
            'isSelected': False,
            'timestamp': datetime.now().isoformat(),
            'celebrity': celebrity_name,
            'images': verified_images,
            'youtube_videos': verified_youtube,
            'vimeo_videos': verified_vimeo,
            'dailymotion_videos': verified_dailymotion
        }
        
        print(f"\nReturning result with:")
        print(f"- {len(verified_images)} verified deepfake images")
        print(f"- {len(verified_youtube)} verified YouTube videos")
        print(f"- {len(verified_vimeo)} verified Vimeo videos")
        print(f"- {len(verified_dailymotion)} verified Dailymotion videos")
        return jsonify([result])
        
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")

@app.route('/api/search', methods=['POST'])
def search_deepfakes():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = None
    try:
        # Save the uploaded file to a temporary location
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        # Read the image file for base64 encoding before processing
        with open(temp_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode()
        
        # Process with name.py - detect celebrity
        print("Running initial face recognition on uploaded image")
        celebrity_name = detect_top_celebrity_name(temp_path, DEFAULT_CREDENTIALS_PATH)
        print(f"Initial celebrity detection result: {celebrity_name}")
        
        # Check if there was an error
        if celebrity_name.startswith("Error:"):
            raise Exception(celebrity_name)
        
        # Check if no matches were found
        if celebrity_name == "NO_MATCHES_FOUND":
            os.remove(temp_path)
            return jsonify({'error': 'No matches found'}), 404
        
        # Get results from urls.py functions - reduced number for efficiency
        print(f"Fetching images for: {celebrity_name}")
        all_images = get_google_image_urls(celebrity_name + " deepfake", total_num=10)  # Reduced from DEFAULT_NUM_IMAGES * 2
        print(f"Found {len(all_images)} images")
        
        # Get videos with 'deepfake' in title - reduced number for efficiency
        search_query = f"{celebrity_name} deepfake"
        print(f"\nFetching videos with query: {search_query}")
        youtube_videos = get_youtube_video_urls(search_query, max_results=5)  # Reduced from DEFAULT_NUM_VIDEOS
        vimeo_videos = get_vimeo_video_urls(search_query, max_results=5)
        dailymotion_videos = get_dailymotion_video_urls(search_query, max_results=5)
        
        print(f"Found {len(youtube_videos)} YouTube videos")
        print(f"Found {len(vimeo_videos)} Vimeo videos")
        print(f"Found {len(dailymotion_videos)} Dailymotion videos")
        
        # Initialize empty results
        verified_images = []
        verified_video = None  # Store one video
        video_platform = None  # Track which platform the video is from
        
        # Create initial result object
        result = {
            'id': str(hash(celebrity_name)),
            'imageUrl': f"data:image/jpeg;base64,{image_base64}",
            'confidence': 95.5,
            'isSelected': False,
            'timestamp': datetime.now().isoformat(),
            'celebrity': celebrity_name,
            'images': verified_images,
            'youtube_videos': [],
            'vimeo_videos': [],
            'dailymotion_videos': []
        }

        # Process images first - find exactly 5 deepfake images
        print("\nProcessing images...")
        for image_url in all_images:
            if len(verified_images) >= 5:  # Stop after finding 5 deepfake images
                print("Found all 5 required deepfake images, stopping image search.")
                break
                
            is_celebrity, is_deepfake = verify_celebrity_and_deepfake(image_url, celebrity_name)
            if is_celebrity and is_deepfake:
                verified_images.append(image_url)
                print(f"Verified deepfake image {len(verified_images)}/5: {image_url}")
                result['images'] = verified_images

        if len(verified_images) < 5:
            print(f"Warning: Only found {len(verified_images)} deepfake images out of 5 required")

        # Process videos - find first deepfake video from any platform
        print("\nProcessing videos...")
        
        # Try Dailymotion first
        if not verified_video:
            print("\nVerifying Dailymotion videos...")
            for video_url in dailymotion_videos:
                print(f"\nProcessing Dailymotion video: {video_url}")
                is_valid, is_deepfake, timestamps = verify_video_frames(video_url, celebrity_name)
                if is_valid and is_deepfake:
                    verified_video = video_url
                    video_platform = 'dailymotion'
                    result['dailymotion_videos'] = [video_url]
                    print(f"Found deepfake Dailymotion video, stopping video search.")
                    break

        # Try Vimeo if no Dailymotion video found
        if not verified_video:
            print("\nVerifying Vimeo videos...")
            for video_url in vimeo_videos:
                print(f"\nProcessing Vimeo video: {video_url}")
                is_valid, is_deepfake, timestamps = verify_video_frames(video_url, celebrity_name)
                if is_valid and is_deepfake:
                    verified_video = video_url
                    video_platform = 'vimeo'
                    result['vimeo_videos'] = [video_url]
                    print(f"Found deepfake Vimeo video, stopping video search.")
                    break

        # Finally try YouTube if still no video found
        if not verified_video:
            print("\nVerifying YouTube videos...")
            for video_url in youtube_videos:
                print(f"\nProcessing YouTube video: {video_url}")
                is_valid, is_deepfake, timestamps = verify_video_frames(video_url, celebrity_name)
                if is_valid and is_deepfake:
                    verified_video = video_url
                    video_platform = 'youtube'
                    result['youtube_videos'] = [video_url]
                    print(f"Found deepfake YouTube video, stopping video search.")
                    break

        print(f"\nFinal Results Summary:")
        print(f"- {len(verified_images)} verified deepfake images found")
        if verified_video:
            print(f"- 1 verified deepfake {video_platform} video found")
        else:
            print("- No verified deepfake videos found")
        
        return jsonify([result])
        
    except Exception as e:
        print(f"Error in search_deepfakes: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up in finally block to ensure it happens
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part
        if file.filename == '':
            return redirect(request.url)
        
        # Get credentials file if provided, otherwise use default
        credentials_path = DEFAULT_CREDENTIALS_PATH
        if 'credentials' in request.files:
            cred_file = request.files['credentials']
            if cred_file.filename != '':
                # Save credentials file temporarily
                credentials_path = os.path.join(app.config['UPLOAD_FOLDER'], 'google_credentials.json')
                cred_file.save(credentials_path)
                # Save path in session for later use
                session['credentials_path'] = credentials_path
        elif 'credentials_path' in session:
            credentials_path = session['credentials_path']
        
        if file:
            # Save the uploaded file to a temporary location
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)
            
            try:
                # Process with name.py - detect celebrity
                celebrity_name = detect_top_celebrity_name(temp_path, credentials_path)
                
                # Check if there was an error
                if celebrity_name.startswith("Error:"):
                    raise Exception(celebrity_name)
                
                # Check if no matches were found
                if celebrity_name == "NO_MATCHES_FOUND":
                    # Clean up
                    os.remove(temp_path)
                    return render_template('no_matches.html')
                
                # Store celebrity name in session for load more functionality
                session['celebrity_name'] = celebrity_name
                
                # Get results from urls.py functions
                images = get_google_image_urls(celebrity_name, total_num=DEFAULT_NUM_IMAGES)
                youtube_videos = get_youtube_video_urls(celebrity_name, max_results=DEFAULT_NUM_VIDEOS)
                vimeo_videos = get_vimeo_video_urls(celebrity_name, max_results=DEFAULT_NUM_VIDEOS)
                dailymotion_videos = get_dailymotion_video_urls(celebrity_name, max_results=DEFAULT_NUM_VIDEOS)
                
                # Clean up
                os.remove(temp_path)
                
                # Render results
                return render_template('results.html', 
                                    celebrity=celebrity_name,
                                    images=images,
                                    youtube_videos=youtube_videos,
                                    vimeo_videos=vimeo_videos,
                                    dailymotion_videos=dailymotion_videos)
            except Exception as e:
                # Clean up in case of error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return render_template('index.html', error=str(e), has_default_credentials=os.path.exists(DEFAULT_CREDENTIALS_PATH))
    
    # Pass to template whether default credentials exist
    return render_template('index.html', has_default_credentials=os.path.exists(DEFAULT_CREDENTIALS_PATH))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Use port 5002 instead of 5001 (which is also in use)
    app.run(debug=True, port=5002) 