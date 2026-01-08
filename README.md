# ⚡ Power Consumption Prediction System

An AI-powered electricity consumption prediction and recommendation system that helps households optimize their energy usage, reduce costs, and minimize environmental impact.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a sophisticated **Neural Network-based prediction system** for household electricity consumption. Using deep learning and time-series analysis, the system provides:

- **Accurate predictions** of electricity consumption with 96.6% accuracy
- **Real-time monitoring** and alerts for excessive energy usage
- **Personalized recommendations** for energy-saving strategies
- **Cost analysis** and potential savings calculations
- **Interactive dashboard** for easy visualization and insights

The system is designed for homeowners, energy consultants, and utility companies to better understand and optimize electricity consumption patterns.

## ✨ Features

### 🔮 Prediction Engine
- **High-accuracy forecasting**: 99.6% R² score, 3.4% MAPE
- **25 engineered features** including temporal patterns, historical data, and appliance usage
- **Real-time predictions** based on current household parameters
- **Cyclical time encoding** for better temporal pattern recognition

### 💡 Smart Recommendations
- **Severity-based alerts**: Critical, High, Medium, and Low priority recommendations
- **Context-aware suggestions**: Personalized based on time of day, day of week, and usage patterns
- **Appliance-specific tips**: Targeted advice for kitchen, laundry, and HVAC systems
- **Actionable guidance**: Specific steps to reduce consumption

### 📊 Analytics Dashboard
- **Interactive visualizations** using Plotly
- **Consumption gauge** with color-coded thresholds
- **Cost breakdown** and savings potential
- **Model performance metrics** and comparisons
- **Energy-saving tips** database with expected savings percentages

### 💰 Cost Analysis
- **Hourly, daily, and monthly** cost projections
- **Savings calculator** with reduction target simulation
- **ROI estimates** for implementing recommendations
- **CO₂ reduction** impact tracking

## 📈 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **R² Score** | 0.9964 | Model explains 99.64% of variance |
| **MAPE** | 3.40% | Average prediction error of only 3.4% |
| **MAE** | 0.024 kW | Mean absolute error |
| **RMSE** | 0.054 kW | Root mean squared error |

**Training Data**: 1,639,404 samples  
**Test Data**: 409,852 samples  
**Total Epochs**: 47  
**Batch Size**: 128

## 🛠 Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow 2.15**: Deep learning framework
- **Keras**: High-level neural network API
- **Streamlit**: Interactive web application framework

### Data Science & ML
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Feature scaling and metrics
- **Joblib**: Model serialization

### Visualization
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Fayedolla/Power-Consumption-.git
cd Power-Consumption-
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
Ensure all required files are present:
- `neural_network_model.h5` or `neural_network_model.keras`
- `scaler_X.pkl` and `scaler_y.pkl`
- `feature_info.json` and `model_metadata.json`

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Prediction System

1. **Navigate to "Predict Consumption"** page
2. **Enter household parameters**:
   - Time information (hour, day, month)
   - Current usage (voltage, current intensity)
   - Appliance usage (kitchen, laundry, HVAC)
   - Recent history (past 1h, 2h, 3h, 6h, 12h, 24h)
3. **Click "Predict Consumption"**
4. **Review results**:
   - Predicted vs actual consumption
   - Severity-based recommendations
   - Cost analysis and potential savings

### Exploring Model Performance

Navigate to **"Model Performance"** page to view:
- Detailed metrics (MAE, RMSE, R², MAPE)
- Neural network architecture
- Hyperparameters
- Model comparison charts

### Energy-Saving Tips

Visit **"Energy Tips"** page for:
- Category-wise energy-saving strategies
- Expected savings percentages
- Savings calculator
- Annual cost reduction projections

## 📁 Project Structure

```
Power-Consumption-/
│
├── app.py                          # Main Streamlit application
├── main.py                         # Entry point (optional)
├── requirements.txt                # Python dependencies
│
├── neural_network_model.h5         # Trained model (H5 format)
├── neural_network_model.keras      # Trained model (Keras format)
├── saved_model/                    # SavedModel format
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   └── variables/
│
├── scaler_X.pkl                    # Feature scaler
├── scaler_y.pkl                    # Target scaler
├── feature_info.json               # Feature metadata
├── model_metadata.json             # Model performance metrics
├── training_history.csv            # Training logs
│
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🧠 Model Architecture

### Neural Network Design

```
Input Layer (25 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (30%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (20%)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (20%)
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear)
```

### Hyperparameters

- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Batch Size**: 128
- **Epochs**: 47
- **Validation Split**: 20%

## 📊 Dataset

The model is trained on the **Individual Household Electric Power Consumption Dataset** containing:

- **2+ million measurements** of household electricity consumption
- **4-year time period** (2006-2010)
- **1-minute sampling rate**
- **Features include**:
  - Global active power
  - Global reactive power
  - Voltage
  - Global intensity
  - Sub-metering (kitchen, laundry, HVAC)

### Feature Engineering

**25 engineered features** including:
- **Temporal features**: hour, day, month, day_of_week, quarter
- **Cyclical encoding**: hour_sin, hour_cos, day_sin, day_cos
- **Categorical**: is_weekend, time_of_day
- **Historical lags**: 1h, 2h, 3h, 6h, 12h, 24h
- **Rolling statistics**: mean_3h, mean_6h, std_6h
- **Physical measurements**: voltage, global_intensity, sub_metering_1/2/3

## 🔧 How It Works

### 1. Data Collection
User inputs current household parameters through the web interface.

### 2. Feature Engineering
The system automatically generates 25 features including:
- Temporal patterns (time of day, day of week)
- Cyclical encodings for better time representation
- Rolling statistics from historical data

### 3. Prediction
The neural network processes scaled features and outputs predicted consumption.

### 4. Analysis
The system compares predicted vs actual consumption and calculates:
- Excess consumption percentage
- Cost implications
- Potential savings

### 5. Recommendations
Based on consumption patterns, the system generates:
- Severity-based alerts
- Time-specific recommendations
- Appliance-specific tips
- Cost-saving strategies

## 🖼 Screenshots

### Home Dashboard
Interactive dashboard showing model performance and key metrics.

### Prediction Interface
User-friendly form for inputting household parameters and receiving instant predictions.

### Recommendations
Severity-based recommendations with actionable steps to reduce consumption.

### Analytics
Detailed visualizations of consumption patterns and savings potential.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Fayedolla** - [GitHub Profile](https://github.com/Fayedolla)

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository - Individual Household Electric Power Consumption
- Built with TensorFlow, Keras, and Streamlit
- Inspired by sustainable energy initiatives

## 📧 Contact

For questions, suggestions, or collaborations:
- **GitHub**: [@Fayedolla](https://github.com/Fayedolla)
- **Repository**: [Power-Consumption-](https://github.com/Fayedolla/Power-Consumption-.git)

---

<div align="center">
  <p><strong>⚡ Save Energy • 💰 Save Money • 🌍 Save the Planet</strong></p>
  <p>Made with ❤️ for a sustainable future</p>
</div>
