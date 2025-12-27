import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

CSV_PATH = 'cars_600.csv'
MODEL_PATH = 'our_prediction.joblib'

def train_model():
    df = pd.read_csv(CSV_PATH)

    brand_encoder = LabelEncoder()
    model_encoder = LabelEncoder()

    df['brand_enc'] = brand_encoder.fit_transform(df['brand'])
    df['model_enc'] = model_encoder.fit_transform(df['model'])

    X = df[['brand_enc', 'model_enc', 'year', 'engine_size']]
    y = df['price']

    model = DecisionTreeRegressor()
    model.fit(X, y)

    joblib.dump({
        'model': model,
        'brand_encoder': brand_encoder,
        'model_encoder': model_encoder
    }, MODEL_PATH)
