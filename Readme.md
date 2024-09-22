# Sentiment Analysis Application

This project is a sentiment analysis application that utilizes multiple sentiment analysis models, including VADER, SentiWordNet, and Keras. The application features a Streamlit UI for user interaction and a Flask backend for processing sentiment analysis requests.

## Project Structure

sentiment-analysis-app/
│
├── backend/                    # Flask backend service
│   ├── SentimentModelServices.py  # Main Flask application
│   ├── requirements.txt        # Python packages for backend
│   ├── model/                  # Directory for models
│   │   └── sentiment_model.keras  # Pre-trained Keras model
│   ├── utils/                  # Utility functions
│   │   ├── preprocess.py       # Preprocessing functions
│   │   └── sentiment_analysis.py  # Sentiment analysis logic
│   └── static/                 # Static files (if needed)
│       └── ...                 # CSS, JS, images, etc.
│
├── ui/                         # Streamlit UI
│   ├── SentiUIApp.py                  # Main Streamlit application
│   ├── requirements.txt        # Python packages for UI
│   ├── assets/                 # Assets for UI (images, styles, etc.)
│   │   └── ...                 # Other UI-related assets
│   └── README.md               # Documentation for UI
│
├── .gitignore                  # Git ignore file
└── README.md                   # Documentation for the entire project


## Installation

### Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend

2.Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.Install the required packages:

pip install -r requirements.txt

4.Run the Flask backend
python SentimentModelServices.py

### UI
1.Create and activate a virtual environment (optional):

cd ui
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

2. Install Required Packages
pip install -r requirements.txt

3.Run Streamlit App :
streamlit run SentiUIApp.py





