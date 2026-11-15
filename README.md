# 🌍 Climate Change Analyzer

A Python-based climate data analysis project that uses real-world historical temperature anomaly data from **NASA GISS** to explore global temperature trends through mathematical and computational techniques.

The project demonstrates the practical application of **Vectors, Matrices, Linear Regression, Interpolation, and Extrapolation** to analyze historical climate patterns and estimate temperature anomalies.

## Project Overview

Climate change is one of the major environmental challenges of our time. Understanding historical temperature variations helps identify long-term climate trends.

This project analyzes annual global temperature anomaly data obtained from NASA's GISS Surface Temperature Analysis (GISTEMP) dataset. It applies numerical and mathematical methods to examine temperature variations, calculate trends, estimate intermediate values, and extend historical trends into future years.

## Key Features

- **NASA Data Integration:** Downloads and processes historical global temperature anomaly data directly from NASA GISS.
- **Vector Operations:** Calculates year-to-year temperature changes, annual rates of change, average annual rates, and vector magnitude.
- **Matrix Operations:** Constructs climate data matrices and applies matrix multiplication and inverse operations.
- **Linear Regression:** Uses the Least Squares Method to determine the best-fitting linear temperature trend.
- **Interpolation:** Estimates temperature anomalies between two known historical observations.
- **Extrapolation:** Estimates future temperature anomalies by extending the calculated historical linear trend.
- **Data Visualization:** Generates graphs representing historical temperature trends, annual variations, interpolation, and extrapolation.
- **Interactive Interface:** Provides a menu-driven terminal interface for accessing different analysis features.

## Mathematical Concepts Implemented

| Concept | Application |
|---|---|
| Vector Differences | Calculates year-to-year temperature changes |
| Annual Rate of Change | Measures temperature variation per year |
| Vector Magnitude | Calculates the Euclidean norm of the change vector |
| Matrix Operations | Organizes climate data and performs mathematical computations |
| Linear Regression | Determines the best-fitting historical temperature trend using Least Squares |
| Linear Interpolation | Estimates values between two known observations |
| Linear Extrapolation | Estimates values beyond the available historical data |

### Core Mathematical Formulas

**1. Vector Difference**

ΔTᵢ = Tᵢ₊₁ − Tᵢ

**2. Linear Regression (Least Squares Method)**

β = (XᵀX)⁻¹XᵀY

**3. Linear Interpolation**

T = T₁ + [(y − y₁)/(y₂ − y₁)] × (T₂ − T₁)

**4. Linear Extrapolation**

T = a + b(y − y₀)

Where `a` represents the intercept, `b` represents the regression slope, and `y₀` represents the first year in the dataset.

## Visualizations

The project generates four visualizations:

1. **Historical Temperature Trend:** Displays annual temperature anomalies over time.
2. **Year-to-Year Temperature Changes:** Illustrates annual increases and decreases in temperature anomalies.
3. **Interpolation Graph:** Visualizes the estimated temperature anomaly between two known observations.
4. **Extrapolation Graph:** Displays the historical trend extended to estimate future temperature anomalies.

## Technology Stack

- **Programming Language:** Python
- **NumPy:** Vector operations, matrix computations, and numerical calculations.
- **Pandas:** Dataset loading, cleaning, and preprocessing.
- **Matplotlib:** Graph generation and data visualization.
- **Data Source:** NASA GISS GISTEMP.

## Project Structure

```text
Climate-Change-Analyzer/
│
├── mfad_climate_analyzer.py
│
└── README.md
```

*Note: The Python filename can be adjusted to match the actual filename in the repository.*

## ⚙️ Installation and Execution

### Prerequisites

- Python 3.x
- Internet connection (to retrieve the NASA dataset)

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Climate-Change-Analyzer
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib
```

### 4. Run the Program

```bash
python climate_change_analyzer.py
```

## 📋 Available Operations

The interactive terminal menu provides the following options:

| Option | Operation |
|---|---|
| 1 | Display climate data and vector/matrix operations |
| 2 | Look up an actual NASA observation |
| 3 | Interpolate between two known years |
| 4 | Extrapolate a future temperature anomaly |
| 5 | Plot historical temperature trends |
| 6 | Plot year-to-year temperature changes |
| 7 | Plot interpolation |
| 8 | Plot extrapolation |
| 9 | Exit the program |

## Dataset and References

**NASA GISS – GISTEMP Surface Temperature Analysis**

Dataset: [Global-mean monthly, seasonal, and annual means](https://data.giss.nasa.gov/gistemp/)

The project uses the `GLB.Ts+dSST.csv` dataset and its `J-D` column, representing annual global temperature anomalies in degrees Celsius.

Reference period: **1951–1980**.

## Limitations

- Interpolation assumes a linear relationship between the two selected observations.
- Extrapolation assumes that the historical linear trend continues beyond the available data.
- The extrapolated values are mathematical estimates, not official climate forecasts.
- The analysis does not account for all atmospheric, oceanic, environmental, and future emission-related factors influencing global temperatures.

## Learning Outcomes

Through this project, we explored how mathematical concepts can be implemented in Python to solve real-world analytical problems. It strengthened our understanding of numerical computation, data preprocessing, matrix algebra, trend analysis, and scientific data visualization.

---

**Project Domain:** Climate Data Analysis | Numerical Computing | Applied Mathematics

**Built using Python, NumPy, Pandas, and Matplotlib.**
