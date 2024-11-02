
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

price	rooms	bathroom	lift	terrace	square_meters	square_meters_price	real_state_attic	real_state_flat	real_state_study	neighborhood_Eixample	neighborhood_Gràcia	neighborhood_Horta- Guinardo	neighborhood_Les Corts	neighborhood_Nou Barris	neighborhood_Sant Andreu	neighborhood_Sant Martí	neighborhood_Sants-Montjuïc	neighborhood_Sarria-Sant Gervasi
0	750	3.0	1	1	0	60	12.500000	0	1	0	0	0	1	0	0	0	0	0	0
1	770	2.0	1	1	0	59	13.050847	0	1	0	0	0	0	0	0	1	0	0	0
2	1300	1.0	1	1	1	30	28.384073	0	1	0	0	1	0	0	0	0	0	0	0
3	2225	1.0	1	1	1	70	28.384073	0	1	0	0	0	0	0	0	0	0	0	0
4	720	2.0	1	1	0	44	16.363636	0	1	0	0	0	0	0	0	1	0	0	0

scaled data head():
price	rooms	bathroom	lift	terrace	square_meters	square_meters_price	real_state_attic	real_state_flat	real_state_study	neighborhood_Eixample	neighborhood_Gràcia	neighborhood_Horta- Guinardo	neighborhood_Les Corts	neighborhood_Nou Barris	neighborhood_Sant Andreu	neighborhood_Sant Martí	neighborhood_Sants-Montjuïc	neighborhood_Sarria-Sant Gervasi
0	0.225722	0.625	0.0	1.0	0.0	0.370370	0.304200	0.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0
1	0.236220	0.375	0.0	1.0	0.0	0.362963	0.328330	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
2	0.514436	0.125	0.0	1.0	1.0	0.148148	1.000000	0.0	1.0	0.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
3	1.000000	0.125	0.0	1.0	1.0	0.444444	1.000000	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
4	0.209974	0.375	0.0	1.0	0.0	0.251852	0.473446	0.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0


'''
import joblib
import os
import numpy as np
import pandas as pd

# Change the working directory to where the .pkl file is located
os.chdir("C:\\Users\\otroc\\OneDrive\\Documents\\Carlos\\IMPELIA_PEA AIDS_PROJECT1\\PROJECT1")

# Load the model and scaler
loaded_model = joblib.load("Random_Forest_Tuned.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Define the new data as a DataFrame with correct feature names
new_data = pd.DataFrame(
    [[3,1,1,0,60,12.5,0,1,0,0,0,1,0,0,0,0,0,0]], 
    columns=loaded_scaler.feature_names_in_
)

# Add the constant as in the training phase
new_data.insert(0, 'const', 1)

# Scale the data, excluding the constant column
new_data_scaled = new_data.copy()
new_data_scaled.iloc[:, 1:] = loaded_scaler.transform(new_data.iloc[:, 1:])

# Predict using the loaded model
prediction = loaded_model.predict(new_data_scaled)
print("Prediction (scaled):", prediction)

# Inverse transform the output 
predicted_price = loaded_scaler.inverse_transform(np.array([[prediction[0]]] * new_data.shape[1]))[0][0]
print("Predicted value (original scale):", predicted_price)
