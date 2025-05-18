from google.cloud import vision
from google.oauth2 import service_account
import os

def detect_top_celebrity_name(image_path, credentials_path=None):
    # Check for credentials in different ways
    try:
        if credentials_path and os.path.exists(credentials_path):
            # Use explicit credentials if provided
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Fall back to environment variable
            client = vision.ImageAnnotatorClient()
            
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.web_detection(image=image)

        web_detection = response.web_detection

        if web_detection.web_entities:
            # Get the top entity
            top_entity = web_detection.web_entities[0]
            
            # Check if the entity score is high enough (confidence threshold)
            # And if there are enough matching images
            if (top_entity.score >= 0.5 and 
                (len(web_detection.partial_matching_images) > 2 or 
                 len(web_detection.full_matching_images) > 0)):
                return top_entity.description
            else:
                # Person not famous enough or not enough matching images
                return "NO_MATCHES_FOUND"
        else:
            # No entities found
            return "NO_MATCHES_FOUND"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # This is for direct script testing
    image_path = input("Enter the path to your image: ")
    credentials_path = input("Enter the path to your credentials file (or press Enter to use environment variable): ")
    
    if not credentials_path.strip():
        credentials_path = None
        
    result = detect_top_celebrity_name(image_path, credentials_path)
    print(f"Detected: {result}")