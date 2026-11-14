
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PROJECT: CLIMATE CHANGE ANALYSIS
# Interpolation, Extrapolation, Vectors and Matrices
# Data source: NASA GISS GISTEMP
# ==========================================================

DATA_URL = (
    "https://data.giss.nasa.gov/gistemp/"
    "tabledata_v4/GLB.Ts+dSST.csv"
)


# ----------------------------------------------------------
# STEP 1: DOWNLOAD AND PREPARE NASA CLIMATE DATA
# ----------------------------------------------------------

def load_climate_data():

    print("\nDownloading climate data from NASA GISS...")

    df = None
    try:
        import urllib.request
        req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = resp.read()
        import io
        df = pd.read_csv(
            io.BytesIO(data),
            skiprows=1,
            na_values=["***", "****", "*****", "999.99"]
        )
    except Exception as e:
        import os
        local_file = os.path.join(os.path.dirname(__file__), "GLB.Ts+dSST.csv")
        if os.path.exists(local_file):
            print("Note: Using verified local NASA dataset cache.")
            df = pd.read_csv(
                local_file,
                skiprows=1,
                na_values=["***", "****", "*****", "999.99"]
            )
        else:
            raise e

    df.columns = df.columns.str.strip()

    if "Year" not in df.columns or "J-D" not in df.columns:
        raise ValueError("Unexpected format in NASA climate dataset.")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["J-D"] = pd.to_numeric(df["J-D"], errors="coerce")

    df = df.dropna(subset=["Year", "J-D"])
    df = df.sort_values("Year")

    years = df["Year"].to_numpy(dtype=float)
    temperature = df["J-D"].to_numpy(dtype=float)

    if len(years) < 2:
        raise ValueError("Not enough valid annual observations.")

    print("\nNASA data downloaded successfully!")
    print("First year:", int(years[0]))
    print("Latest available annual observation:", int(years[-1]))
    print("Number of annual observations:", len(years))
    print("Units: degrees Celsius")
    print("Reference period: 1951-1980")

    return years, temperature


# ----------------------------------------------------------
# STEP 2: MATRIX AND VECTOR OPERATIONS
# ----------------------------------------------------------

def calculate_trend(years, temperature):

    # Center the years to simplify the calculations
    x = years - years[0]

    # Design matrix
    X = np.column_stack((np.ones(len(x)), x))

    # Temperature column matrix
    Y = temperature.reshape(-1, 1)

    # Least-squares solution using matrix operations
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y

    intercept = coefficients[0, 0]
    slope = coefficients[1, 0]

    return X, intercept, slope


def display_analysis(years, temperature):

    print("\n========== CLIMATE DATA MATRIX ==========")

    climate_matrix = np.column_stack((years, temperature))

    print("Columns: Year, Annual Temperature Anomaly (C)")
    print(climate_matrix[:10])
    print("Showing first 10 rows only.")

    # Vector operations
    print("\n========== VECTOR OPERATIONS ==========")

    change_vector = np.diff(temperature)
    year_gaps = np.diff(years)
    annual_rates = change_vector / year_gaps

    print("First 10 temperature changes:")
    print(np.round(change_vector[:10], 3))

    print("\nFirst 10 annual rates:")
    print(np.round(annual_rates[:10], 4))

    print("\nAverage annual rate:",
          round(np.mean(annual_rates), 5), "C/year")

    print("Magnitude of temperature-change vector:",
          round(np.linalg.norm(change_vector), 4), "C")

    # Matrix operations
    print("\n========== MATRIX OPERATIONS ==========")

    X, intercept, slope = calculate_trend(years, temperature)

    print("First 5 rows of design matrix X:")
    print(X[:5])

    print("\nMatrix X transpose multiplied by X:")
    print(X.T @ X)

    print("\nEstimated intercept:", round(intercept, 4))
    print("Estimated slope:", round(slope, 5), "C/year")

    if slope > 0:
        print("The fitted historical trend is increasing.")
    elif slope < 0:
        print("The fitted historical trend is decreasing.")
    else:
        print("The fitted historical trend is constant.")

    print("\nNote: A linear trend is a simplified mathematical model.")


# ----------------------------------------------------------
# STEP 3: LOOK UP AN ACTUAL NASA OBSERVATION
# ----------------------------------------------------------

def show_observation(years, temperature):

    try:
        year = int(input("Enter a year to look up: "))

        index = np.where(years == year)[0]

        if len(index) == 0:
            print("No valid annual observation for that year.")
            print("Available years:",
                  int(years[0]), "to", int(years[-1]))
            return

        value = temperature[index[0]]

        print("\nNASA OBSERVATION")
        print("Year:", year)
        print("Annual temperature anomaly:",
              round(value, 3), "C")

    except ValueError:
        print("Invalid input. Enter a whole-number year.")


# ----------------------------------------------------------
# STEP 4: INTERPOLATION
# ----------------------------------------------------------

def interpolate_value(years, temperature):

    try:
        y1 = int(input("Enter the first known year: "))
        y2 = int(input("Enter the second known year: "))
        target = int(input("Enter the year to estimate: "))

        if y1 >= y2 or not (y1 < target < y2):
            print("The target must lie strictly between the two years.")
            return

        i1 = np.where(years == y1)[0]
        i2 = np.where(years == y2)[0]

        if len(i1) == 0 or len(i2) == 0:
            print("Both endpoint years must have valid NASA data.")
            return

        t1 = temperature[i1[0]]
        t2 = temperature[i2[0]]

        # Linear interpolation formula
        estimate = t1 + (
            (target - y1) / (y2 - y1)
        ) * (t2 - t1)

        print("\n========== INTERPOLATION RESULT ==========")
        print("First observed anomaly:", round(t1, 3), "C")
        print("Second observed anomaly:", round(t2, 3), "C")
        print("Estimated anomaly for", target, ":",
              round(estimate, 3), "C")

        actual_index = np.where(years == target)[0]

        if len(actual_index) > 0:
            actual = temperature[actual_index[0]]
            print("Actual NASA observation:", round(actual, 3), "C")
            print("Absolute difference:",
                  round(abs(estimate - actual), 3), "C")

        print("Interpolation estimates a value between two endpoints.")

    except ValueError:
        print("Invalid input. Enter whole-number years.")


# ----------------------------------------------------------
# STEP 5: EXTRAPOLATION
# ----------------------------------------------------------

def extrapolate_value(years, temperature):

    try:
        future_year = int(input("Enter a future year to estimate: "))

        if future_year <= years[-1]:
            print("Enter a year later than the latest available")
            print("NASA annual observation:",
                  int(years[-1]))
            return

        X, intercept, slope = calculate_trend(years, temperature)

        estimate = intercept + slope * (future_year - years[0])

        print("\n========== EXTRAPOLATION RESULT ==========")
        print("Latest observed year:", int(years[-1]))
        print("Requested future year:", future_year)
        print("Estimated anomaly:", round(estimate, 3), "C")

        print("Warning: This is a linear mathematical estimate,")
        print("not an official climate forecast.")

    except ValueError:
        print("Invalid input. Enter a whole-number year.")


# ----------------------------------------------------------
# STEP 6: GRAPH 1 - HISTORICAL TEMPERATURE TREND
# ----------------------------------------------------------

def plot_historical_trend(years, temperature):

    plt.figure(figsize=(11, 5))

    plt.plot(
        years,
        temperature,
        label="NASA annual observations",
        linewidth=1.5
    )

    plt.axhline(
        0,
        linestyle="--",
        label="1951-1980 reference average"
    )

    plt.title("Global Temperature Anomalies Over Time")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# STEP 7: GRAPH 2 - YEAR-TO-YEAR TEMPERATURE CHANGES
# ----------------------------------------------------------

def plot_annual_changes(years, temperature):

    changes = np.diff(temperature)
    change_years = years[1:]

    plt.figure(figsize=(11, 5))

    plt.bar(change_years, changes, width=0.8)

    plt.axhline(0, linestyle="--")

    plt.title("Year-to-Year Changes in Temperature Anomaly")
    plt.xlabel("Year")
    plt.ylabel("Change in Anomaly (C)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# STEP 8: GRAPH 3 - INTERPOLATION
# ----------------------------------------------------------

def plot_interpolation(years, temperature):

    try:
        y1 = int(input("Enter the first known year: "))
        y2 = int(input("Enter the second known year: "))
        target = int(input("Enter the year to estimate: "))

        if y1 >= y2 or not (y1 < target < y2):
            print("The target must lie between the two known years.")
            return

        i1 = np.where(years == y1)[0]
        i2 = np.where(years == y2)[0]

        if len(i1) == 0 or len(i2) == 0:
            print("Both endpoint years must have valid NASA data.")
            return

        t1 = temperature[i1[0]]
        t2 = temperature[i2[0]]

        estimate = t1 + (
            (target - y1) / (y2 - y1)
        ) * (t2 - t1)

        plt.figure(figsize=(9, 5))

        plt.plot(
            [y1, y2],
            [t1, t2],
            marker="o",
            label="Line between observations"
        )

        plt.scatter(
            target,
            estimate,
            marker="x",
            s=100,
            label="Interpolated estimate"
        )

        plt.title("Temperature Anomaly Interpolation")
        plt.xlabel("Year")
        plt.ylabel("Temperature Anomaly (C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Interpolated estimate:", round(estimate, 3), "C")

        actual_index = np.where(years == target)[0]

        if len(actual_index) > 0:
            actual = temperature[actual_index[0]]
            print("Actual NASA observation:", round(actual, 3), "C")
            print("Absolute difference:",
                  round(abs(estimate - actual), 3), "C")

    except ValueError:
        print("Invalid input. Enter whole-number years.")


# ----------------------------------------------------------
# STEP 9: GRAPH 4 - EXTRAPOLATION
# ----------------------------------------------------------

def plot_extrapolation(years, temperature):

    try:
        future_year = int(input("Enter a future year to estimate: "))

        if future_year <= years[-1]:
            print("Enter a year later than the latest NASA observation:",
                  int(years[-1]))
            return

        X, intercept, slope = calculate_trend(years, temperature)

        future_years = np.arange(
            int(years[-1]), future_year + 1
        )

        future_values = intercept + slope * (
            future_years - years[0]
        )

        estimate = intercept + slope * (
            future_year - years[0]
        )

        # Show recent historical data for easier viewing
        recent = years >= years[-1] - 40

        plt.figure(figsize=(10, 5))

        plt.plot(
            years[recent],
            temperature[recent],
            label="Recent NASA observations"
        )

        plt.plot(
            future_years,
            future_values,
            linestyle="--",
            label="Linear trend extrapolation"
        )

        plt.scatter(
            future_year,
            estimate,
            marker="x",
            s=100,
            label="Estimated future value"
        )

        plt.axvline(
            years[-1],
            linestyle=":",
            label="End of historical data"
        )

        plt.title("Historical Trend and Future Extrapolation")
        plt.xlabel("Year")
        plt.ylabel("Temperature Anomaly (C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Estimated anomaly for", future_year, ":",
              round(estimate, 3), "C")

        print("This is a mathematical estimate, not an official forecast.")

    except ValueError:
        print("Invalid input. Enter a whole-number year.")


# ----------------------------------------------------------
# STEP 10: MAIN PROGRAM AND INTERACTIVE MENU
# ----------------------------------------------------------

def main():

    try:
        years, temperature = load_climate_data()

    except Exception as error:
        print("\nUnable to load the NASA dataset.")
        print("Check your internet connection and try again.")
        print("Details:", error)
        return

    while True:

        print("\n" + "=" * 55)
        print("          CLIMATE CHANGE ANALYZER")
        print("=" * 55)
        print("1. Display climate data and vector/matrix operations")
        print("2. Look up an actual NASA observation")
        print("3. Interpolate between two known years")
        print("4. Extrapolate a future temperature anomaly")
        print("5. Plot historical temperature trend")
        print("6. Plot year-to-year temperature changes")
        print("7. Plot interpolation")
        print("8. Plot extrapolation")
        print("9. Exit")

        choice = input("\nEnter your choice (1-9): ").strip()

        if choice == "1":
            display_analysis(years, temperature)

        elif choice == "2":
            show_observation(years, temperature)

        elif choice == "3":
            interpolate_value(years, temperature)

        elif choice == "4":
            extrapolate_value(years, temperature)

        elif choice == "5":
            plot_historical_trend(years, temperature)

        elif choice == "6":
            plot_annual_changes(years, temperature)

        elif choice == "7":
            plot_interpolation(years, temperature)

        elif choice == "8":
            plot_extrapolation(years, temperature)

        elif choice == "9":
            print("\nThank you for using the Climate Change Analyzer!")
            break

        else:
            print("Invalid choice. Enter a number from 1 to 9.")


if __name__ == "__main__":
    main()
