# 🌍 Footprint Predictor

A desktop application that predicts the **Ecological Footprint (Consumption)** of a country using **Linear Regression** trained on the *Global Ecological Footprint 2023 dataset*.  

The app is built with:
- **Python** (pandas, scikit-learn, tkinter/customtkinter, Pillow)
- **Machine Learning** (Linear Regression)
- **GUI** (CustomTkinter with styled inputs and background)

---

## ✨ Features
- Predict **Total Ecological Footprint (Consumption)** based on socio-economic and ecological indicators.
- Input values manually or **paste data directly from Excel**.
- Provides model accuracy via **Mean Absolute Error (MAE)**.
- Modern UI with **light theme** and background image.
- Built-in **Help panel** explaining all predictors.

---

## 📊 Dataset
The model uses the [Global Ecological Footprint 2023](https://www.kaggle.com/datasets/jainaru/global-ecological-footprint-2023) dataset.  
Unnecessary columns (e.g., Country, Region, etc.) are dropped, and missing values are filled with feature means.

---

## 🛠️ Installation

1. **Clone the repository**
Run `git clone https://github.com/Orvielle-Atanacio/footprint-predictor.git` and then `cd footprint-predictor`.

Example requirements.txt:
```
pandas
scikit-learn
customtkinter
pillow
```
2. Prepare dataset

- Place the CSV file Global Ecological Footprint 2023.csv inside the project folder.
- Update the file path in the script if needed.

## 📸 Footprint Predictor in Action
![UI_Sample](https://github.com/Orvielle-Atanacio/Ecological-Footprint-Predictor/blob/main/assets/ui.png)
