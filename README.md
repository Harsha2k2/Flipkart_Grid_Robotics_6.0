# Product Analysis System

A Flask-based web application that analyzes packaged products and fresh produce using computer vision and AI. The system provides real-time analysis of product freshness, expiry dates, and quality assessment.

🌐 Live Demo: [https://flipkart-grid-robotics-6-0.onrender.com/](https://flipkart-grid-robotics-6-0.onrender.com/)  

⚠️ Note: Due to free hosting plan, the website cold starts every 15 minutes of inactivity. Please open the link in a new tab while reading this README to allow for startup time.

📚 Documentation & Demo Materials: [Google Drive Folder](https://drive.google.com/drive/folders/1olTw5sWuBg6nC59jvMfUxkVYtJwYheIl?usp=sharing)
- Demo Video
- Technical Report
- Presentation Slides

## Features

- **Product Analysis**
  - Packaged Product Detection (Brand, Expiry Date)
  - Fresh Produce Assessment (Freshness Score, Shelf Life)
  - Item Counting
  - Quality Analysis

- **Dashboard**
  - Real-time Analysis Results
  - Historical Data Tracking
  - Statistics and Metrics

## High Accuracy Through Model Synergy

Our system achieves exceptional accuracy by combining specialized computer vision models with Google's Gemini Vision AI. This tag-team approach allows us to leverage both custom-trained models for specific tasks and Gemini's advanced visual understanding capabilities, resulting in robust and reliable product analysis.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables for gemini vision api:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   FLASK_DEBUG=True  # Optional
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Tech Stack

- Flask (Backend)
- SQLite (Database)
- Google Gemini API (Vision Analysis)
- TensorFlow & PyTorch (Model Integration)
- Bootstrap (Frontend)

## API Endpoints

- `/analyze/<type>` - Analyze products (branded/fresh/all)
- `/dashboard` - View statistics and recent scans
- `/reset/<type>` - Reset counters and history

## License

Not for commercial use.
