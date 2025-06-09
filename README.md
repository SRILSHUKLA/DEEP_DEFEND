# DeepDefend - Advanced Deepfake Detection Platform

![DeepDefend Logo](frontend/public/logo.png)

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
  - Flask
  - PyTorch
  - OpenCV
  - Gunicorn

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SRILSHUKLA/DEEP_DEFEND.git
   cd DEEP_DEFEND
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

1. Start the server:

   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5002`

## API Documentation

The API is available at `http://localhost:5002` with the following endpoints:

- `POST /api/search` - Search for deepfake content
- `POST /api/analyze` - Analyze uploaded media

## Deployment

The application can be deployed on Render:

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your repository
4. Set environment variables
5. Deploy!

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- GitHub: [@SRILSHUKLA](https://github.com/SRILSHUKLA)
- Project: [DEEP_DEFEND](https://github.com/SRILSHUKLA/DEEP_DEFEND)
