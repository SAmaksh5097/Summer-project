import pandas as pd

def enrich_data(original_file='dataset.xlsx', output_file='dataset_enriched.xlsx'):
    """
    Loads an existing vehicle dataset, adds new imaginary data,
    and saves it to a new file.
    """
    print(f"🔄 Loading original data from '{original_file}'...")
    try:
        original_df = pd.read_excel(original_file)
    except FileNotFoundError:
        print(f"❌ ERROR: Original file '{original_file}' not found. Please make sure it's in the same folder.")
        return

    # Define new, imaginary vehicle data for 2025
    new_vehicle_data = [
        # Modern Cars (Petrol/Diesel)
        {'company': 'Tata', 'name': 'Tata Nexon', 'year': 2024, 'Price': 12.5, 'kms_driven': 15000, 'fuel_type': 'Petrol', 'city': 'Gurugram', 'seller_type': 'Individual', 'has_sunroof': True, 'has_adas': True},
        {'company': 'Mahindra', 'name': 'Mahindra XUV700', 'year': 2023, 'Price': 22.0, 'kms_driven': 25000, 'fuel_type': 'Diesel', 'city': 'Bengaluru', 'seller_type': 'Dealer', 'has_sunroof': True, 'has_adas': True},
        {'company': 'Hyundai', 'name': 'Hyundai Creta', 'year': 2024, 'Price': 16.5, 'kms_driven': 8000, 'fuel_type': 'Petrol', 'city': 'Mumbai', 'seller_type': 'Individual', 'has_sunroof': True, 'has_adas': False},

        # Electric Cars (EVs)
        {'company': 'Tata', 'name': 'Tata Punch EV', 'year': 2024, 'Price': 13.0, 'kms_driven': 9000, 'fuel_type': 'Electric', 'city': 'Bengaluru', 'seller_type': 'Individual', 'battery_capacity_kwh': 35, 'range_km': 421, 'has_sunroof': True, 'has_adas': False},
        {'company': 'Tesla', 'name': 'Tesla Model 3', 'year': 2025, 'Price': 55.0, 'kms_driven': 5000, 'fuel_type': 'Electric', 'city': 'Gurugram', 'seller_type': 'Dealer', 'battery_capacity_kwh': 60, 'range_km': 513, 'has_sunroof': True, 'has_adas': True},

        # Two-Wheelers (Petrol)
        {'company': 'Royal Enfield', 'name': 'Royal Enfield Himalayan 450', 'year': 2024, 'Price': 2.8, 'kms_driven': 6000, 'fuel_type': 'Petrol', 'city': 'Bengaluru', 'seller_type': 'Individual', 'engine_cc': 452},
        {'company': 'Bajaj', 'name': 'Bajaj Pulsar N250', 'year': 2023, 'Price': 1.5, 'kms_driven': 11000, 'fuel_type': 'Petrol', 'city': 'Mumbai', 'seller_type': 'Dealer', 'engine_cc': 249},
        {'company': 'TVS', 'name': 'TVS Apache RTR 310', 'year': 2024, 'Price': 2.5, 'kms_driven': 4000, 'fuel_type': 'Petrol', 'city': 'Gurugram', 'seller_type': 'Individual', 'engine_cc': 312},
        
        # Electric Two-Wheelers
        {'company': 'Ola Electric', 'name': 'Ola S1 Pro Gen2', 'year': 2025, 'Price': 1.4, 'kms_driven': 3000, 'fuel_type': 'Electric', 'city': 'Bengaluru', 'seller_type': 'Individual', 'battery_capacity_kwh': 4, 'range_km': 195},
        {'company': 'Ather', 'name': 'Ather 450X', 'year': 2024, 'Price': 1.35, 'kms_driven': 7000, 'fuel_type': 'Electric', 'city': 'Mumbai', 'seller_type': 'Dealer', 'battery_capacity_kwh': 3.7, 'range_km': 150},
    ]

    print(f"✨ Generating {len(new_vehicle_data)} new vehicle entries...")
    new_data_df = pd.DataFrame(new_vehicle_data)

    # Combine the original data with the new data
    print("➕ Combining original and new data...")
    combined_df = pd.concat([original_df, new_data_df], ignore_index=True)

    # Save the enriched dataframe to a new file
    combined_df.to_excel(output_file, index=False)
    print(f"✅ Success! Enriched data saved to '{output_file}'. It now has {len(combined_df)} rows.")

if __name__ == "__main__":
    enrich_data()
