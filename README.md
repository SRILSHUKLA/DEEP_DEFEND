# Celebrity Recognition Web App

This web application uses Google Cloud Vision API to identify celebrities in uploaded images and then finds related content from various media sources.

## Features

- Image upload interface
- Celebrity recognition using Google Cloud Vision API
- Retrieval of related content from:
  - Google Images
  - YouTube
  - Vimeo
  - Dailymotion
- Responsive modern UI design

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Google Cloud Vision API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Vision API
4. Create service account credentials:
   - Go to IAM & Admin > Service Accounts
   - Create a new service account
   - Grant the role "Cloud Vision API User"
   - Create a key (JSON format)
   - Download the JSON key file

### 4. Set up Google Cloud credentials

Set the environment variable to point to your service account key file:

```bash
# For Mac/Linux
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"

# For Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your-service-account-key.json"
```

### 5. Run the application

```bash
python app.py
```

The application will be available at http://127.0.0.1:5000/

## Usage

1. Open the web app in your browser
2. Upload an image containing a celebrity
3. The system will identify the celebrity and display:
   - Images from Google
   - Videos from YouTube, Vimeo, and Dailymotion

## Note

The Google Custom Search API (used for images) and YouTube Data API have daily quota limits on free accounts. You might need to obtain your own API keys if the default ones reach their quota limits. 