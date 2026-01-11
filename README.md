# Sentiment Analysis API 🚀

A high-performance REST API built with **FastAPI** that predicts the sentiment (Positive/Negative) of text using a pre-trained Machine Learning model. This project demonstrates how to deploy an ML model as a production-ready microservice.

## 📋 Features

* **Real-time Prediction:** Instant sentiment analysis of user inputs.
* **Text Preprocessing:** Built-in cleaning pipeline (removes HTML, URLs, and special characters) to match training data conditions.
* **Confidence Scoring:** Returns probability scores indicating how confident the model is in its prediction.
* **API Documentation:** Interactive Swagger UI and ReDoc included automatically.
* **Health Checks:** Dedicated endpoints to monitor API status.

## 🛠️ Tech Stack

* **Python 3.10+**
* **FastAPI:** For building the high-performance web API.
* **Uvicorn:** As the ASGI server.
* **Scikit-Learn:** For the underlying machine learning model.
* **Pickle:** For model serialization/loading.

## 📂 Project Structure

```bash
├── model/
│   └── sentiment_model.pkl    # Pre-trained ML model
├── sentiment_api/
│   ├── main.py                # Main application logic
│   └── schema.py              # Pydantic models for request/response
├── requirements.txt           # Python dependencies
└── README.md

```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/sentiment-analysis-api.git
cd sentiment-analysis-api

```


2. **Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```


3. **Install dependencies**
```bash
pip install -r requirements.txt

```


4. **Run the Server**
```bash
uvicorn sentiment_api.main:app --reload

```


The server will start at `http://127.0.0.1:8000`.

## 📖 API Documentation

Once the server is running, you can access the interactive API docs:

* **Swagger UI:** [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs) - Test endpoints directly from your browser.
* **ReDoc:** [http://127.0.0.1:8000/redoc](https://www.google.com/search?q=http://127.0.0.1:8000/redoc) - Alternative documentation view.

## 🔌 Usage Examples

### **1. Check API Status**

**Request:** `GET /`

```json
{
  "message": "Sentiment Analysis API is running!",
  "status": "healthy",
  "version": "0.v1"
}

```

### **2. Predict Sentiment**

**Request:** `POST /predict`

```json
{
  "text": "I absolutely loved the service! The team was fantastic."
}

```

**Response:**

```json
{
  "original_text": "I absolutely loved the service! The team was fantastic.",
  "cleaned_text": "i absolutely loved the service the team was fantastic",
  "sentiment": "positive",
  "confidence": 0.98,
  "timestamp": "2023-10-27T10:00:00.123456"
}

```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements.
