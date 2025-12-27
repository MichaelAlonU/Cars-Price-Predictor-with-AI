from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os
import pandas as pd
from train import train_model

app = Flask(__name__)
CORS(app)

CSV_PATH = 'cars_600.csv'
MODEL_PATH = 'our_prediction.joblib'

if not os.path.exists(MODEL_PATH):
    train_model()

data = joblib.load(MODEL_PATH)
model = data['model']
brand_encoder = data['brand_encoder']
model_encoder = data['model_encoder']


@app.route('/brands')
def brands():
    df = pd.read_csv(CSV_PATH)
    return jsonify(sorted(df['brand'].unique().tolist()))


@app.route('/models/<brand>')
def models(brand):
    df = pd.read_csv(CSV_PATH)
    return jsonify(sorted(df[df['brand'] == brand]['model'].unique().tolist()))


@app.route('/years/<brand>/<model_name>')
def years(brand, model_name):
    df = pd.read_csv(CSV_PATH)
    years = sorted(
        df[(df['brand'] == brand) & (df['model'] == model_name)]
        ['year'].unique().tolist()
    )
    return jsonify(years)


@app.route('/engines/<brand>/<model_name>/<int:year>')
def engines(brand, model_name, year):
    df = pd.read_csv(CSV_PATH)
    engines = sorted(
        df[
            (df['brand'] == brand) &
            (df['model'] == model_name) &
            (df['year'] == year)
        ]['engine_size'].unique().tolist()
    )
    return jsonify(engines)


@app.route('/predict', methods=['POST'])
def predict():
    d = request.json

    brand = d['brand']
    model_name = d['model']
    year = int(d['year'])
    engine_size = float(d['engine_size'])

    brand_enc = brand_encoder.transform([brand])[0]
    model_enc = model_encoder.transform([model_name])[0]

    price = model.predict([[brand_enc, model_enc, year, engine_size]])[0]
    return jsonify({'predicted_price': int(price)})


@app.route('/learn', methods=['POST'])
def learn():
    d = request.json
    user_price = float(d['price'])

    if user_price <= 0:
        return jsonify({'error': 'Invalid price'}), 400

    # חיזוי נוכחי
    brand_enc = brand_encoder.transform([d['brand']])[0]
    model_enc = model_encoder.transform([d['model']])[0]

    predicted = model.predict([[
        brand_enc,
        model_enc,
        int(d['year']),
        float(d['engine_size'])
    ]])[0]

    if int(predicted) == int(user_price):
        return jsonify({'status': 'same price – not saved'})

    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([{
        'brand': d['brand'],
        'model': d['model'],
        'year': int(d['year']),
        'engine_size': float(d['engine_size']),
        'price': user_price
    }])], ignore_index=True)

    df.to_csv(CSV_PATH, index=False)
    return jsonify({'status': 'saved'})


@app.route('/retrain', methods=['POST'])
def retrain():
    global model, brand_encoder, model_encoder
    train_model()
    data = joblib.load(MODEL_PATH)
    model = data['model']
    brand_encoder = data['brand_encoder']
    model_encoder = data['model_encoder']
    return jsonify({'status': 'retrained'})


if __name__ == '__main__':
    app.run(debug=True)
