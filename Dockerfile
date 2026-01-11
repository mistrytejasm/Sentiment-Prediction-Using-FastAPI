FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model
COPY sentiment_api/ ./sentiment_api/
COPY model/ ./model/

EXPOSE 8000

# Run your FastAPI app
CMD ["uvicorn", "sentiment_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
