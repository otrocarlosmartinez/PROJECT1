
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

# Set the working directory to where the .pkl files are located
os.chdir("C:\\Users\\otroc\\OneDrive\\Documents\\Carlos\\IMPELIA_PEA AIDS_PROJECT1\\PROJECT1")

# Load the model and scaler
loaded_model = joblib.load("Random_Forest_Tuned.pkl")
input_scaler = joblib.load("Input_scaler.pkl")

# Create new data as a DataFrame with the same column names as the training data
new_data = pd.DataFrame(
    [[3.0,1,1,0,60,12.5,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0]], 
    columns=input_scaler.feature_names_in_
)

# Scale the new data
scaled_data = input_scaler.transform(new_data)

# Add a constant to the scaled data
scaled_data_with_constant = sm.add_constant(scaled_data, has_constant='add')

# Predict using the loaded model
prediction_scaled = loaded_model.predict(scaled_data_with_constant)
#print("Scaled Prediction:", prediction_scaled)

# Apply inverse scaling to obtain the original scale
output_scaler = joblib.load("output_scaler.pkl")
prediction_original_scale = output_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0]
print("Predicted price (EUR):", prediction_original_scale)


