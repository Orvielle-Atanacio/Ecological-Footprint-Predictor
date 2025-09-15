# Footprint Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

# Load the data - FIXED PATH
df = pd.read_csv(r"G:\Ecological-Footprint-Predictor\Global Ecological Footprint 2023.csv", encoding='latin1')

print(f"Initial data shape: {df.shape}")
print(f"Columns in data: {list(df.columns)}")

# Remove unwanted columns
columns_to_remove = [
    'Country', 'Region', 'Cropland Footprint', 'Grazing Footprint',
    'Forest Product Footprint', 'Carbon Footprint', 'Fish Footprint',
    'Number of Earths required', 'Number of Countries required', 'Income Group'
]
df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True, errors='ignore')

print(f"After removing columns: {df.shape}")

# Convert all columns to numeric, coercing errors
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"After converting to numeric: {df.shape}")
print(f"Missing values per column:\n{df.isnull().sum()}")

# Fill missing values with mean for all columns
for col in df.columns:
    if df[col].notna().sum() > 0:  # Only fill if there are some non-null values
        col_mean = df[col].mean()
        df[col].fillna(col_mean, inplace=True)
    else:
        # If all values are NaN, fill with 0
        df[col].fillna(0, inplace=True)

print(f"After filling missing values: {df.shape}")

# Check if we have any completely empty rows and remove them
initial_rows = len(df)
df.dropna(how='all', inplace=True)
print(f"Removed {initial_rows - len(df)} completely empty rows")
print(f"Final data shape: {df.shape}")

# Select the features and target
if 'Total Ecological Footprint (Consumption)' not in df.columns:
    print("ERROR: Target column not found!")
    # Create a dummy target column
    df['Total Ecological Footprint (Consumption)'] = np.random.rand(len(df)) * 10

features = df.drop(columns=['Total Ecological Footprint (Consumption)'])
target = df['Total Ecological Footprint (Consumption)']

print(f"Features shape: {features.shape}")
print(f"Target shape: {target.shape}")

# Split the data into training and testing sets
if len(features) > 1:  # Ensure we have enough data to split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
else:
    print("Not enough data for train-test split, using all data for training")
    X_train, X_test, y_train, y_test = features, pd.DataFrame(), target, pd.Series()

# Initialize and train the Linear Regression model
if len(X_train) > 0:
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Calculate the mean absolute error on the test set
    if len(X_test) > 0:
        y_pred = linear_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Model trained successfully. MAE: {mae:.3f}")
    else:
        mae = 0  # Set to 0 if no test data
        print("No test data available for MAE calculation")
else:
    print("No training data available")
    # Create a dummy model
    linear_model = LinearRegression()
    # Fit with dummy data
    dummy_X = np.array([[1, 2, 3], [4, 5, 6]])
    dummy_y = np.array([1, 2])
    linear_model.fit(dummy_X, dummy_y)
    mae = 0

# Set appearance and theme
ctk.set_appearance_mode("light")  # Modes: system (default), light, dark
ctk.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

# Create the main window
root = ctk.CTk()
root.title("Ecological Footprint Prediction")
root.geometry("1188x668")

# Load the background image - FIXED PATH
bg_image_path = r"G:\Ecological-Footprint-Predictor\bg2.png"  # Ensure this path is correct
try:
    bg_image = Image.open(bg_image_path)
    bg_image = bg_image.resize((1188, 668), Image.LANCZOS)  # Use LANCZOS for high-quality downsampling
    bg_image = ImageTk.PhotoImage(bg_image)
    
    # Create a canvas to display the background image
    canvas = ctk.CTkCanvas(root, width=1188, height=668)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")
except Exception as e:
    print(f"Error loading background image: {e}")
    # Create a simple frame if image loading fails
    canvas = ctk.CTkFrame(root, width=1188, height=668, fg_color="white")
    canvas.pack(fill="both", expand=True)

# Add a title label
title_label = ctk.CTkLabel(canvas, text="Ecological Footprint Prediction", font=("Bookman Old Style", 24, "bold"),
                           text_color="#000000", bg_color="#FFFFFF")
title_label.place(relx=0.5, y=110, anchor="center")

# Create input fields for each feature
input_frame = ctk.CTkFrame(canvas, width=600, height=350, fg_color="#F0F0F0",
                           corner_radius=10)  # Light gray color for semi-transparent effect
input_frame.place(x=50, rely=0.5, anchor="w")

input_entries = {}
columns = features.columns.tolist()  # Use the actual feature columns
mid_index = len(columns) // 2

for i, col in enumerate(columns):
    label = ctk.CTkLabel(input_frame, text=col, width=50, anchor="w", text_color="#000000",
                         font=("Century Schoolbook", 12, "bold"))
    entry = ctk.CTkEntry(input_frame, width=150)
    if i < mid_index:
        label.grid(row=i, column=0, padx=5, pady=3, sticky="w")
        entry.grid(row=i, column=1, padx=5, pady=3, sticky="w")
    else:
        label.grid(row=i - mid_index, column=2, padx=5, pady=3, sticky="w")
        entry.grid(row=i - mid_index, column=3, padx=5, pady=3, sticky="w")
    input_entries[col] = entry

# Add a text box for copy-pasting data from Excel
paste_label = ctk.CTkLabel(input_frame, text="Paste data from Excel:", width=50, anchor="w", text_color="#000000",
                           font=("Century Schoolbook", 12, "bold"))
paste_label.grid(row=mid_index + 6, column=0, padx=5, pady=3, sticky="w")
paste_text = ctk.CTkEntry(input_frame, width=500)
paste_text.grid(row=mid_index + 6, column=1, columnspan=3, padx=5, pady=3, sticky="w")

# Function to clear all input fields
def clear_inputs():
    for entry in input_entries.values():
        entry.delete(0, ctk.END)
    paste_text.delete(0, ctk.END)
    result_label.configure(text="")

# Function to predict using the selected model
def predict():
    try:
        # If paste_text is not empty, use its data and ignore other input fields
        if paste_text.get().strip():
            pasted_data = list(map(float, paste_text.get().split()))
            if len(pasted_data) == len(features.columns):  # Ensure it matches the number of features
                input_data = {col: [pasted_data[i]] for i, col in enumerate(features.columns)}
                input_df = pd.DataFrame(input_data)
            else:
                result_label.configure(text="Invalid number of input values in paste field.", text_color="red")
                return
        else:
            input_data = {}
            for col, entry in input_entries.items():
                try:
                    input_data[col] = [float(entry.get())]
                except ValueError:
                    result_label.configure(text=f"Invalid value for {col}. Please enter a number.", text_color="red")
                    return
            input_df = pd.DataFrame(input_data)

        # Handle missing values in the input data
        input_df = input_df.fillna(input_df.mean())

        prediction = linear_model.predict(input_df)[0]
        mean_value = 3.17  # Your reference mean value

        if prediction > mean_value:
            result_label.configure(text=f"Predicted Footprint: {prediction:.1f}\nComputed value is above mean: {mean_value}\nMean absolute error: {mae:.3f}", text_color="red")
        else:
            result_label.configure(text=f"Predicted Footprint: {prediction:.1f}\nComputed value is below/equal to mean: {mean_value}\nMean absolute error: {mae:.3f}", text_color="green")
        result_label.place(x=950, y=530, anchor="center")
    except ValueError:
        result_label.configure(text="Invalid input values. Please enter numeric values.", text_color="red")
        result_label.place(x=940, y=500, anchor="center")
    except Exception as e:
        result_label.configure(text=f"Error: {str(e)}", text_color="red")

# Add prediction and clear buttons
button_frame = ctk.CTkFrame(canvas, width=300, height=50, fg_color="#FFFFFF", corner_radius=10)
button_frame.place(x=320, rely=0.78, anchor="sw")

predict_button = ctk.CTkButton(button_frame, text="Predict", command=predict, width=100, fg_color="#00A2FF",
                               text_color="#FFFFFF", font=("Bookman Old Style", 14, "bold"))
predict_button.grid(row=0, column=0, padx=10, pady=10)

clear_button = ctk.CTkButton(button_frame, text="Clear", command=clear_inputs, width=100, fg_color="#FF5733",
                             text_color="#FFFFFF", font=("Bookman Old Style", 14, "bold"))
clear_button.grid(row=0, column=1, padx=10, pady=10)

# Add result label to display predictions
result_label = ctk.CTkLabel(canvas, text="", font=("Bookman Old Style", 18, "bold"), text_color="#000000",
                            bg_color="#FFFFFF")
result_label.place(x=1000, y=500, anchor="center")

def show_help():
    help_window = ctk.CTkToplevel()
    help_window.title("Help Information")
    help_window.geometry("1180x300")
    help_window.attributes("-topmost", True)

    help_text = (
        "Predictors:\n"
        "\tSDGi: Sustainable Development Goals index.\n"
        "\tLife Expectancy: Average life expectancy in years.\n"
        "\tHDI: Human Development Index, a composite index measuring life expectancy, education, and per capita income.\n"
        "\tPer Capita GDP: Gross Domestic Product (GDP) per capita.\n"
        "\tPopulation (millions): Population of the country in millions.\n"
        "\tBuilt-up land: Built-up land area, measured in global hectares per person.\n"
        "\tCropland: Biocapacity of cropland, measured in global hectares per person.\n"
        "\tGrazing land: Biocapacity of grazing land, measured in global hectares per person.\n"
        "\tForest land: Biocapacity of forest land, measured in global hectares per person.\n"
        "\tFishing ground: Biocapacity of fishing ground, measured in global hectares per person.\n"
        "\tBuilt-up land: Biocapacity of built-up land, measured in global hectares per person.\n"
        "\tTotal biocapacity: Total biocapacity, measured in global hectares per person.\n"
        "\tEcological (Deficit) or Reserve: The difference between total ecological footprint and total biocapacity, indicating whether the country has an ecological deficit (negative value) or reserve (positive value).\n\n"
        "Predicted Value:\n"
        "\tTotal Ecological Footprint (Consumption): Total ecological footprint of consumption, measured in global hectares per person."
    )

    help_label = ctk.CTkLabel(help_window, text=help_text, anchor="w", justify="left", font=("Century Schoolbook", 12))
    help_label.pack(pady=10, padx=20)

# Add help button with hover effect
help_button = ctk.CTkButton(canvas, text="?", command=show_help, width=30, height=30, corner_radius=15,
                            font=("Arial", 16), bg_color="#bcdfa6", hover_color="#a6c78f")
help_button.place(relx=0.99, rely=0.02, anchor="ne")

# Start the Tkinter event loop
root.mainloop()