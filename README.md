# DeepDefend - Advanced Deepfake Detection Platform

![image](https://github.com/user-attachments/assets/f01d742d-60b6-4cbe-b607-930cc85a8f23)
![Image](https://github.com/user-attachments/assets/73ba2948-7300-4e74-b9a3-b633461201e0)
![Image](https://github.com/user-attachments/assets/25d78196-cf33-494b-ac4e-2d6a9d8aa813)
![Image](https://github.com/user-attachments/assets/283b646d-f0ea-449b-9833-e2122a5b3690)
![Image](https://github.com/user-attachments/assets/6c9404e3-a265-4b40-b6f6-c6e24f0fe598)


DeepDefend is a cutting-edge web application designed to detect and report deepfake content using advanced AI technology. The platform provides real-time analysis of images and videos, helping users identify manipulated media with high accuracy.

## Features

- 🔍 **Advanced Deepfake Detection**

  - Real-time analysis of images and videos
  - High-accuracy detection using state-of-the-art AI algorithms
  - Support for multiple media formats (PNG, JPG, MP4)

- 🎯 **Comprehensive Search**

  - Search for similar deepfake content across the internet
  - URL-based search functionality
  - Advanced filtering and sorting options

- 📊 **Detailed Analysis**

  - Confidence scores for detected manipulations
  - Frame-by-frame analysis for videos
  - Facial landmark detection
  - Voice pattern analysis

- 🚨 **Reporting System**

  - Direct reporting to relevant authorities
  - Batch reporting capabilities
  - Detailed evidence compilation
  - Support for multiple reporting channels

- 🎨 **Modern UI/UX**
  - Responsive design
  - Real-time loading animations
  - Interactive visualizations
  - Dark mode interface

## Tech Stack

- **Frontend**

  - React.js
  - TypeScript
  - Tailwind CSS
  - Framer Motion
  - Lucide Icons

- **Backend**
  - Python
  - FastAPI
  - TensorFlow/PyTorch
  - OpenCV

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8+
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. Install frontend dependencies:

   ```bash
   cd frontend
   npm install
   ```

3. Install backend dependencies:

   ```bash
   cd ../backend
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

1. Start the backend server:

   ```bash
   cd backend
   python main.py
   ```

2. Start the frontend development server:

   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## API Documentation

The backend API is available at `http://localhost:5002` with the following endpoints:

- `POST /api/search` - Search for deepfake content
- `POST /api/analyze` - Analyze uploaded media
- `POST /api/report` - Submit reports to authorities


## Acknowledgments

- Thanks to all contributors who have helped shape DeepDefend: @Samriddhi903, @SuryadeepSinh-Jadeja, @moKshagna-p, @Mayan10
- The deepfake detection model used in this project was trained using the Deepfake and Real Images Dataset by Manjil Karki. We extend our gratitude to the creators for making this dataset publicly available. https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images






