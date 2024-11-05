'''
const: constant added for modeling
price: The price of the real-state.
rooms: Number of rooms.
bathroom: Number of bathrooms.
lift: whether a building has an elevator (also known as a lift in some regions) or not
terrace: If it has a terrace or not.
square_meters: Number of square meters.
real_state: Kind of real-state.
neighborhood: Neighborhood
square_meters_price: Price of the square meter.

df5.head()
price	rooms	bathroom	lift	terrace	square_meters	square_meters_price	real_state_apartment	real_state_attic	real_state_flat	real_state_study	real_state_unknown	neighborhood_Sarria-Sant Gervasi	neighborhood_Les Corts	neighborhood_Eixample	neighborhood_Sant Martí	neighborhood_Ciutat Vella	neighborhood_Gràcia	neighborhood_Sants-Montjuïc	neighborhood_Horta- Guinardo	neighborhood_Sant Andreu	neighborhood_Nou Barris
0	750	3.0	1	1	0	60	12.500000	0	0	1	0	0	0	0	0	0	0	0	0	1	0	0
1	770	2.0	1	1	0	59	13.050847	0	0	1	0	0	0	0	0	0	0	0	0	0	1	0
2	1300	1.0	1	1	1	30	28.384073	0	0	1	0	0	0	0	0	0	0	1	0	0	0	0
3	2225	1.0	1	1	1	70	28.384073	0	0	1	0	0	0	0	0	0	1	0	0	0	0	0
4	720	2.0	1	1	0	44	16.363636	0	0	1	0	0	0	0	0	0	0	0	0	0	1	0
5 rows × 22 columns


X.head()
const	rooms	bathroom	lift	terrace	square_meters	square_meters_price	real_state_apartment	real_state_attic	real_state_flat	real_state_study	real_state_unknown	neighborhood_Sarria-Sant Gervasi	neighborhood_Les Corts	neighborhood_Eixample	neighborhood_Sant Martí	neighborhood_Ciutat Vella	neighborhood_Gràcia	neighborhood_Sants-Montjuïc	neighborhood_Horta- Guinardo	neighborhood_Sant Andreu	neighborhood_Nou Barris
0	1.0	0.625	0.0	1.0	0.0	0.370370	0.304200	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0
1	1.0	0.375	0.0	1.0	0.0	0.362963	0.328330	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
2	1.0	0.125	0.0	1.0	1.0	0.148148	1.000000	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0
3	1.0	0.125	0.0	1.0	1.0	0.444444	1.000000	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0
4	1.0	0.375	0.0	1.0	0.0	0.251852	0.473446	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
5 rows × 22 columns


y.head()
	price
0	0.225722
1	0.236220
2	0.514436
3	1.000000
4	0.209974
'''

import joblib
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

def create_house_price_calculator():
    # Valid options for each input
    VALID_OPTIONS = {
        'rooms': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        'bathroom': [1, 2, 3, 4, 5, 6, 7, 8],
        'lift': [True, False],
        'terrace': [True, False],
        'real_state': ['flat', 'attic', 'apartment', 'study'],
        'neighborhood': [
            'Horta- Guinardo', 'Sant Andreu', 'Gràcia', 'Ciutat Vella',
            'Sarria-Sant Gervasi', 'Les Corts', 'Sant Martí', 'Eixample',
            'Sants-Montjuïc', 'Nou Barris'
        ]
    }

    # Average square meter prices by neighborhood (replace with actual values)
    NEIGHBORHOOD_PRICES = {
        'Horta- Guinardo': 12.5,
        'Sant Andreu': 11.8,
        'Gràcia': 14.2,
        'Ciutat Vella': 13.9,
        'Sarria-Sant Gervasi': 16.5,
        'Les Corts': 15.3,
        'Sant Martí': 13.2,
        'Eixample': 15.8,
        'Sants-Montjuïc': 12.9,
        'Nou Barris': 11.2
    }

    def display_options(options, title):
        print(f"\n{title}:")
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        return {idx: option for idx, option in enumerate(options, 1)}

    def get_validated_input(prompt, valid_options=None, input_type=str, is_indexed=False):
        while True:
            try:
                if input_type == bool:
                    user_input = input(f"{prompt} (yes/no): ").lower()
                    if user_input in ['yes', 'y']:
                        return True
                    elif user_input in ['no', 'n']:
                        return False
                    raise ValueError
                
                if is_indexed:
                    choice = int(input(prompt))
                    if choice not in valid_options:
                        raise ValueError
                    return valid_options[choice]
                
                value = input_type(input(prompt))
                if valid_options is not None and value not in valid_options:
                    raise ValueError
                return value
                
            except ValueError:
                if is_indexed:
                    print(f"Invalid input. Please choose a number between 1 and {len(valid_options)}")
                elif valid_options:
                    print(f"Invalid input. Please choose from: {valid_options}")
                else:
                    print(f"Invalid input. Please enter a valid {input_type.__name__}")

    def create_feature_vector(inputs, neighborhood_prices):
        # Initialize feature vector with zeros
        feature_vector = np.zeros(21)
        
        # Set basic features
        feature_vector[0] = inputs['rooms']
        feature_vector[1] = inputs['bathroom']
        feature_vector[2] = int(inputs['lift'])
        feature_vector[3] = int(inputs['terrace'])
        feature_vector[4] = inputs['square_meters']
        feature_vector[5] = neighborhood_prices[inputs['neighborhood']]  # square_meters_price
        
        # Set real_state one-hot encoding (positions 6-9)
        real_state_map = {'flat': 6, 'attic': 7, 'apartment': 8, 'study': 9}
        if inputs['real_state'] in real_state_map:
            feature_vector[real_state_map[inputs['real_state']]] = 1
            
        # Set neighborhood one-hot encoding (positions 10-19)
        neighborhood_map = {
            'Horta- Guinardo': 10, 'Sant Andreu': 11, 'Gràcia': 12, 
            'Ciutat Vella': 13, 'Sarria-Sant Gervasi': 14, 'Les Corts': 15,
            'Sant Martí': 16, 'Eixample': 17, 'Sants-Montjuïc': 18, 
            'Nou Barris': 19
        }
        if inputs['neighborhood'] in neighborhood_map:
            feature_vector[neighborhood_map[inputs['neighborhood']]] = 1
            
        return feature_vector

    # Get user inputs
    print("\n=== House Price Calculator ===\n")
    
    # Get basic inputs
    inputs = {
        'rooms': get_validated_input("Number of rooms: ", VALID_OPTIONS['rooms'], int),
        'bathroom': get_validated_input("Number of bathrooms: ", VALID_OPTIONS['bathroom'], int),
        'lift': get_validated_input("Has lift/elevator?", None, bool),
        'terrace': get_validated_input("Has terrace?", None, bool),
        'square_meters': get_validated_input("Square meters: ", None, int),
    }
    
    # Display and get real_state selection
    real_state_options = display_options(VALID_OPTIONS['real_state'], "Select type of real estate")
    inputs['real_state'] = get_validated_input(
        "Enter the number of your choice: ", 
        real_state_options, 
        int, 
        is_indexed=True
    )
    
    # Display and get neighborhood selection
    neighborhood_options = display_options(VALID_OPTIONS['neighborhood'], "Select neighborhood")
    inputs['neighborhood'] = get_validated_input(
        "Enter the number of your choice: ", 
        neighborhood_options, 
        int, 
        is_indexed=True
    )

    # Create feature vector
    feature_vector = create_feature_vector(inputs, NEIGHBORHOOD_PRICES)
    
    try:
        # Load the model and scalers
        loaded_model = joblib.load("Random_Forest_Tuned.pkl")
        input_scaler = joblib.load("Input_scaler.pkl")
        output_scaler = joblib.load("output_scaler.pkl")

        # Create DataFrame with the feature vector
        new_data = pd.DataFrame([feature_vector], columns=input_scaler.feature_names_in_)

        # Scale the data
        scaled_data = input_scaler.transform(new_data)
        
        # Add constant
        scaled_data_with_constant = sm.add_constant(scaled_data, has_constant='add')
        
        # Make prediction
        prediction_scaled = loaded_model.predict(scaled_data_with_constant)
        
        # Inverse transform the prediction
        prediction_original_scale = output_scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1))[0]
        
        print("\nPredicted price (EUR):", f"{prediction_original_scale[0]:,.2f}")
        
    except FileNotFoundError:
        print("\nError: Required model files not found. Please check file paths.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    create_house_price_calculator()