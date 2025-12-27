# 🚗 Car Price Prediction – Full Stack ML Project

A full-stack machine learning project that predicts used car prices based on real market data.  
The system allows users to **predict**, **correct**, and **continuously improve** the model over time.

---

## ✨ Features

- 📊 Machine Learning model (Random Forest Regressor)
- 🧠 Predict car price by:
  - Brand
  - Model
  - Year
  - Engine size
- 🧩 Dynamic dropdowns – only valid combinations are selectable
- ✍️ User feedback:
  - Users can submit a corrected price
  - Data is saved **only if the price differs from the prediction**
- 🔁 One-click model retraining
- 🌐 REST API built with Flask
- 💻 Frontend with HTML, CSS (futuristic UI), and Axios
- 📁 CSV-based dataset (easy to inspect and extend)

---

## 🗂 Project Structure

```
car-price-predictor/
│
├── backend/
│   ├── app.py              # Flask server
│   ├── train.py            # ML training logic
│   ├── cars_600.csv        # Dataset
│   ├── our_prediction.joblib
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│
└── README.md
```

---

## 🧠 Machine Learning

- **Model:** RandomForestRegressor  
- **Features used:**
  - Brand (encoded)
  - Model (encoded)
  - Year
  - Engine size
- **Target:** Car price  
- **Training data:** Realistic car models & prices (CSV)

The model improves over time as users submit corrected prices and retrain it.

---

## 🌐 API Endpoints

### Get available data

```
GET /brands
GET /models/<brand>
GET /years/<brand>/<model>
GET /engines/<brand>/<model>/<year>
```

### Predict price

```
POST /predict
{
  "brand": "Toyota",
  "model": "Corolla",
  "year": 2020,
  "engine_size": 1.6
}
```

### Learn from user input

```
POST /learn
{
  "brand": "Toyota",
  "model": "Corolla",
  "year": 2020,
  "engine_size": 1.6,
  "price": 14500
}
```

### Retrain model

```
POST /retrain
```

---

## 🎨 Frontend

- Clean futuristic UI
- Fully dynamic selects
- Axios-based communication with backend
- User-friendly learning flow

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/car-price-predictor.git
cd car-price-predictor
```

### 2️⃣ Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 3️⃣ Frontend

Open the frontend directly:

```
frontend/index.html
```

---

## 🧪 Dataset

The CSV file structure:

```
brand,model,year,engine_size,price
```

All dropdowns are populated strictly from this data to ensure consistency.

---

## 📈 Future Improvements

- Prevent duplicate samples
- Weighted averaging instead of row duplication
- Model performance metrics
- Database storage instead of CSV
- Deployment (Docker / Cloud)

---

## 👤 Author

**Michael Uzan**  
Full-Stack Developer | Machine Learning Enthusiast
