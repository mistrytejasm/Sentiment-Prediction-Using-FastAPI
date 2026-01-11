from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import pickle
import re

# ============================================================================
# STEP 1: Security Configuration
# ============================================================================

# SECRET KEY - Change this in production!
SECRET_KEY = "your-secret-key-change-this-in-production-12345"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ============================================================================
# STEP 2: Fake User Database (Replace with real DB in production)
# ============================================================================

# Password for 'testuser' is 'secret123'
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


# To generate new password hash:
# from passlib.context import CryptContext
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# print(pwd_context.hash("your_password"))

# ============================================================================
# STEP 3: Pydantic Models
# ============================================================================

class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class ReviewRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing!"
            }
        }


class PredictionResponse(BaseModel):
    original_text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    predicted_by: str
    timestamp: str


# ============================================================================
# STEP 4: Authentication Functions
# ============================================================================

def verify_password(plain_password, hashed_password):
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str):
    """Get user from database"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return user_dict
    return None


def authenticate_user(username: str, password: str):
    """Authenticate user with username and password"""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(username)
    if user is None:
        raise credentials_exception
    return User(**user)


# ============================================================================
# STEP 5: Load Model
# ============================================================================

print("Loading model...")
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✓ Model loaded successfully!")

# ============================================================================
# STEP 6: Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="Secure Sentiment Analysis API",
    description="Protected API with OAuth2 authentication",
    version="2.0.0"
)


# ============================================================================
# STEP 7: Text Preprocessing
# ============================================================================

def clean_text(text):
    """Clean text same as training"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# STEP 8: API Endpoints
# ============================================================================

@app.get("/")
def home():
    """Home endpoint"""
    return {
        "message": "Secure Sentiment Analysis API",
        "status": "healthy",
        "version": "2.0.0",
        "authentication": "OAuth2 required",
        "endpoints": {
            "login": "/token",
            "predict": "/predict (requires authentication)",
            "user_info": "/users/me",
            "docs": "/docs"
        }
    }


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint to get access token

    Default credentials:
    - username: testuser
    - password: secret123
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(data={"sub": user["username"]})

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(
        request: ReviewRequest,
        current_user: User = Depends(get_current_user)
):
    """
    Predict sentiment (PROTECTED ENDPOINT)

    Requires authentication token
    """
    try:
        # Clean the text
        cleaned = clean_text(request.text)

        if not cleaned or len(cleaned.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text is empty after preprocessing"
            )

        # Make prediction
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]

        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(max(probabilities))

        return PredictionResponse(
            original_text=request.text,
            cleaned_text=cleaned,
            sentiment=sentiment,
            confidence=confidence,
            predicted_by=current_user.username,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Health check (no authentication required)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# STEP 9: Run the Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  SECURE SENTIMENT ANALYSIS API")
    print("=" * 60)
    print("\nDefault Login Credentials:")
    print("  Username: testuser")
    print("  Password: secret123")
    print("\nAPI Documentation: http://127.0.0.1:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
