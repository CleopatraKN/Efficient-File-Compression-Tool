/******************************************************************************

Welcome to GDB Online.
  GDB online is an online compiler and debugger tool for C, C++, Python, PHP, Ruby, 
  C#, OCaml, VB, Perl, Swift, Prolog, Javascript, Pascal, COBOL, HTML, CSS, JS
  Code, Compile, Run and Debug online from anywhere in world.

*******************************************************************************/
import tensorflow as tf
import numpy as np

# Simulated dataset: [solar generation, wind generation, EV charging demand, grid demand]
data = np.array([
    [50, 30, 20, 80], [40, 25, 30, 85], [60, 35, 25, 75], [55, 40, 20, 70],
    [65, 45, 35, 90], [70, 50, 40, 95], [75, 55, 45, 100]
])

# Inputs (renewables and EV demand) and output (grid distribution)
X = data[:, :3]
y = data[:, 3]

# deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)  # Predict grid demand
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=100, verbose=0)

# Predict grid load for new input
new_input = np.array([[55, 35, 25]])  # 55MW solar, 35MW wind, 25MW EV charging
predicted_grid_load = model.predict(new_input)
print(f"Predicted Grid Load: {predicted_grid_load[0][0]:.2f} MW")
