import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
file_path = r"D:\Users\Mahesh\Desktop\SOM Proj\SOM Proj\data\pre-owned cars.csv"  # or use forward slashes
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Print the column names to check for the correct names
print(df.columns.tolist())

# Preprocess the data
# Update the column names based on the actual dataset
df = df[['make_year', 'price', 'km_driven']].dropna()  # Use 'make_year', 'price', 'km_driven'

# Check the data type of km_driven
print(df['km_driven'].dtype)  # This will print the data type of the column

# If km_driven is not a string, convert it directly to float
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')  # Convert to numeric

# Drop rows with NaN values (if any)
df = df.dropna()

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Initialize and train the Self-Organizing Map (SOM)
som_x, som_y = 10, 10  # SOM grid size
som = MiniSom(x=som_x, y=som_y, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Initialize weights and train the SOM
som.random_weights_init(data_scaled)
som.train_random(data_scaled, 1000)

# Visualize the result
plt.figure(figsize=(10, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plot distance map
plt.colorbar()

# Plot markers for each data point on the SOM grid
markers = ['o']
colors = ['r']

for i, x in enumerate(data_scaled):
    w = som.winner(x)  # Get the winning neuron
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[0], markerfacecolor='None', markeredgecolor=colors[0],  markersize=10, markeredgewidth=2)

plt.title('Self-Organizing Map (SOM) for Cars Dataset')
plt.xlabel('SOM X-axis')
plt.ylabel('SOM Y-axis')
plt.show()
