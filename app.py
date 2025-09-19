from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

BIKE_COMPANIES = ['Honda', 'Royal Enfield', 'TVS', 'Ola Electric', 'Bajaj', 'Ather']

def clean_data(df):
    """A robust function to clean the dataset for the app."""
    # --- Clean numerical columns ---
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['kms_driven'] = df['kms_driven'].astype(str).str.replace(',', '').str.replace('kms', '').str.strip()
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    
    # --- START: CRITICAL FIX ---
    # Add the Price cleaning logic that was missing from this file
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    # --- END: CRITICAL FIX ---
    
    # Drop rows where essential data is missing
    df.dropna(subset=['year', 'kms_driven', 'Price'], inplace=True)
    df['year'] = df['year'].astype(int)
    df['kms_driven'] = df['kms_driven'].astype(int)

    # Standardize the 'Price' column to ensure all values are in Lakhs
    price_threshold = 1000  # Any price over 1000 is assumed to be in Rupees
    df['Price'] = df['Price'].apply(lambda x: x / 100000 if x > price_threshold else x)

    # Clean and fill all other feature columns
    cat_features = ['company', 'fuel_type', 'city', 'seller_type', 'name']
    for col in cat_features:
        # Reassign the column to avoid the warning
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype(str).str.strip()

    df.fillna({
        'battery_capacity_kwh': 0, 'range_km': 0, 'engine_cc': 0,
        'has_sunroof': False, 'has_adas': False
    }, inplace=True)
    
    # Create derived features
    df['vehicle_type'] = np.where(df['company'].isin(BIKE_COMPANIES), 'Two-Wheeler', 'Car')
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    return df

# --- Load and Clean Data and Model ---
try:
    raw_df = pd.read_excel('dataset_enriched.xlsx')
    # Create a copy to avoid chained assignment warnings
    full_df = clean_data(raw_df.copy())
    print(f"âœ… Enriched dataset loaded and cleaned. {len(full_df)} valid rows available.")
except Exception as e:
    full_df = None
    print(f"âŒ FATAL ERROR loading or cleaning dataset: {e}")

try:
    with open('model/vehicle_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    print("âœ… Final model pipeline loaded.")
except Exception as e:
    pipeline = None
    print(f"âŒ FATAL ERROR loading model: {e}")

# --- Helper function to get dropdown data (unchanged) ---
def get_form_data(vehicle_type):
    if full_df is None: return {}
    df = full_df[full_df['vehicle_type'] == vehicle_type]
    
    data = {
        'companies': sorted([str(item) for item in df['company'].unique()]),
        'years': sorted(df['year'].unique(), reverse=True),
        'cities': sorted([str(item) for item in df['city'].unique()])
    }
    if vehicle_type == 'Car':
        data['fuel_types'] = sorted([str(f) for f in df['fuel_type'].unique()])
    return data

# --- Routes (unchanged) ---
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/cars')
def car_form():
    return render_template('car_form.html', **get_form_data('Car'))

@app.route('/bikes')
def bike_form():
    return render_template('bike_form.html', **get_form_data('Two-Wheeler'))
    
@app.route('/buy')
def buyer_dashboard():
    if full_df is None or pipeline is None: return "Server Error: Data or model not loaded.", 500
    
    inventory_df = full_df.sample(n=15, random_state=42).copy()
    inventory_df['predicted_price'] = pipeline.predict(inventory_df)
    vehicles_list = inventory_df.to_dict(orient='records')

    # --- START: NEW DEAL ANALYSIS LOGIC ---
    for vehicle in vehicles_list:
        seller_price = vehicle['Price']
        predicted_price = vehicle['predicted_price']

        if seller_price < predicted_price * 0.95:  # If more than 5% below market
            vehicle['deal_status'] = 'Great Deal!'
            vehicle['deal_color'] = 'success'  # Green
        elif seller_price > predicted_price * 1.05: # If more than 5% above market
            vehicle['deal_status'] = 'Scope for Negotiation'
            vehicle['deal_color'] = 'warning'  # Yellow
        else:
            vehicle['deal_status'] = 'Fair Price'
            vehicle['deal_color'] = 'primary'   # Blue
    # --- END: NEW DEAL ANALYSIS LOGIC ---

    return render_template('buyer_dashboard.html', vehicles=vehicles_list)

@app.route('/dashboard')
def dashboard():
    if full_df is None:
        return "Server Error: Data not loaded.", 500

    # This part will now work correctly
    avg_price_by_company = full_df.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
    company_labels = avg_price_by_company.index.tolist()
    price_values = avg_price_by_company.values.tolist()

    fuel_type_counts = full_df['fuel_type'].value_counts()
    fuel_labels = fuel_type_counts.index.tolist()
    fuel_values = fuel_type_counts.values.tolist()

    return render_template(
        'dashboard.html',
        company_labels=company_labels,
        price_values=price_values,
        fuel_labels=fuel_labels,
        fuel_values=fuel_values
    )

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None or full_df is None: return "Model or data not loaded. Check server logs.", 500
    try:
        vehicle_type = request.form['vehicle_type']
        form_data = { 'company': request.form['company'], 'year': int(request.form['year']), 'kms_driven': int(request.form['kms_driven']), 'city': request.form['city'], 'vehicle_type': vehicle_type, 'seller_type': 'Individual', 'range_km': 0, 'has_sunroof': False, 'has_adas': False, 'name': 'N/A' }
        seller_offer = float(request.form.get('offer_price', 0))
        if vehicle_type == 'Car':
            form_data.update({'fuel_type': request.form['fuel_type'], 'battery_capacity_kwh': float(request.form.get('battery_capacity_kwh', 0)), 'engine_cc': 0})
            template, dropdowns = 'car_form.html', get_form_data('Car')
        else:
            form_data.update({'fuel_type': 'Petrol', 'engine_cc': int(request.form.get('engine_cc', 0)), 'battery_capacity_kwh': 0})
            template, dropdowns = 'bike_form.html', get_form_data('Two-Wheeler')
        input_df = pd.DataFrame([form_data])
        input_df['age'] = datetime.now().year - input_df['year']
        prediction = pipeline.predict(input_df)[0]
        prediction_text = f" {prediction:,.2f} Lakhs"
        deal_info = {'seller_price_text': f" {seller_offer:,.2f} Lakhs"}

        if seller_offer > 0:
            if seller_offer > prediction * 1.1:
                deal_info.update({'status': 'Priced Above Market', 'message': 'Your asking price is higher than the AI-predicted market value. You may want to lower it for a quicker sale.', 'color': 'warning' })
            elif seller_offer < prediction * 0.9:
                deal_info.update({'status': 'Priced Below Market', 'message': 'Your asking price is significantly below market value. You could likely sell for more!', 'color': 'danger'})
            else:
                deal_info.update({'status': 'Fair Market Price', 'message': 'Your asking price is competitive and in line with the AI-predicted market value.', 'color': 'success'})

        return render_template(template, prediction_text=prediction_text, deal=deal_info, **dropdowns)

    except Exception as e:
        print(f"ERROR in /predict: {e}")
        return f"An error occurred: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)