import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# File locations
DATA_FILE = "data/india_climate_2000_2024.csv"
RESULTS_DIR = "results"


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df


# -------------------------------------------------
# Validate and Preprocess Dataset
# -------------------------------------------------
def preprocess_data(df):
    required_columns = [
        "Year",
        "Temperature_C",
        "Rainfall_mm",
        "CO2_Emissions_tonnes"
    ]

    # Check required columns
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    print("\n===== DATA VALIDATION =====")

    # Check for missing values
    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values:")
    print(missing_values)

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print("\nDuplicate rows:", duplicates)

    # Check numeric columns
    numeric_cols = [
        "Year",
        "Temperature_C",
        "Rainfall_mm",
        "CO2_Emissions_tonnes"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid rows
    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)

    print("\nInvalid rows removed:", before - after)

    # Sort data by year
    df = df.sort_values("Year").reset_index(drop=True)

    print("Valid observations:", len(df))
    print("Year range:", int(df["Year"].min()), "-", int(df["Year"].max()))

    return df


    # Check required columns
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Convert numeric columns
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=required_columns)

    # Sort by year
    df = df.sort_values("Year")

    return df


# -------------------------------------------------
# View Dataset
# -------------------------------------------------
def view_dataset(df):
    print("\n===== DATASET =====")
    print(df.to_string(index=False))


# -------------------------------------------------
# Descriptive Statistics
# -------------------------------------------------
def descriptive_statistics(df):
    print("\n===== DESCRIPTIVE STATISTICS =====")

    columns = [
        "Temperature_C",
        "Rainfall_mm",
        "CO2_Emissions_tonnes"
    ]

    print(df[columns].describe())


# -------------------------------------------------
# Correlation Analysis
# -------------------------------------------------
def correlation_analysis(df):
    print("\n===== CORRELATION ANALYSIS =====")

    columns = [
        "Temperature_C",
        "Rainfall_mm",
        "CO2_Emissions_tonnes"
    ]

    correlation = df[columns].corr()

    print(correlation)

    # Save heatmap
    plt.figure(figsize=(7, 5))
    plt.imshow(correlation, cmap="coolwarm")

    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.yticks(range(len(columns)), columns)

    plt.colorbar(label="Correlation")

    for i in range(len(columns)):
        for j in range(len(columns)):
            plt.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Climate Variables Correlation")
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/correlation_heatmap.png"
    )

    plt.show()


# -------------------------------------------------
# Temperature Trend
# -------------------------------------------------
def temperature_trend(df):
    print("\n===== TEMPERATURE TREND =====")

    print(
        f"Temperature in {int(df['Year'].iloc[0])}: "
        f"{df['Temperature_C'].iloc[0]:.4f} °C"
    )

    print(
        f"Temperature in {int(df['Year'].iloc[-1])}: "
        f"{df['Temperature_C'].iloc[-1]:.4f} °C"
    )

    change = (
        df["Temperature_C"].iloc[-1]
        - df["Temperature_C"].iloc[0]
    )

    print(f"Total temperature change: {change:.4f} °C")

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Year"],
        df["Temperature_C"],
        marker="o"
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Trend in India")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/temperature_trend.png"
    )

    plt.show()


# -------------------------------------------------
# Rainfall Trend
# -------------------------------------------------
def rainfall_trend(df):
    print("\n===== RAINFALL TREND =====")

    print(
        f"Minimum rainfall: "
        f"{df['Rainfall_mm'].min():.2f} mm"
    )

    print(
        f"Maximum rainfall: "
        f"{df['Rainfall_mm'].max():.2f} mm"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Year"],
        df["Rainfall_mm"],
        marker="o"
    )

    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.title("Rainfall Trend in India")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/rainfall_trend.png"
    )

    plt.show()


# -------------------------------------------------
# CO2 Emissions Trend
# -------------------------------------------------
def co2_trend(df):
    print("\n===== CO2 EMISSIONS TREND =====")

    first = df["CO2_Emissions_tonnes"].iloc[0]
    last = df["CO2_Emissions_tonnes"].iloc[-1]

    percentage_change = ((last - first) / first) * 100

    print(f"CO2 emissions in first year: {first:.2f}")
    print(f"CO2 emissions in last year: {last:.2f}")
    print(f"Percentage increase: {percentage_change:.2f}%")

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["Year"],
        df["CO2_Emissions_tonnes"],
        marker="o"
    )

    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions (tonnes)")
    plt.title("CO2 Emissions Trend in India")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/co2_trend.png"
    )

    plt.show()


# -------------------------------------------------
# Temperature Regression
# -------------------------------------------------
def temperature_regression(df):
    print("\n===== TEMPERATURE REGRESSION =====")

    X = df[["Year"]]
    y = df["Temperature_C"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["Year"],
        y,
        label="Actual"
    )

    plt.plot(
        df["Year"],
        predictions,
        label="Regression Line"
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Linear Regression")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/temperature_regression.png"
    )

    plt.show()


# -------------------------------------------------
# Generate All Visualizations
# -------------------------------------------------
def generate_all(df):
    temperature_trend(df)
    rainfall_trend(df)
    co2_trend(df)
    correlation_analysis(df)
    temperature_regression(df)

    print("\nAll visualizations generated successfully.")


# -------------------------------------------------
# Final Interpretation
# -------------------------------------------------
def final_interpretation(df):

    print("\n===== FINAL INTERPRETATION =====")

    temperature_change = (
        df["Temperature_C"].iloc[-1]
        - df["Temperature_C"].iloc[0]
    )

    co2_change = (
        (
            df["CO2_Emissions_tonnes"].iloc[-1]
            - df["CO2_Emissions_tonnes"].iloc[0]
        )
        / df["CO2_Emissions_tonnes"].iloc[0]
    ) * 100

    temperature_rainfall = df[
        "Temperature_C"
    ].corr(df["Rainfall_mm"])

    temperature_co2 = df[
        "Temperature_C"
    ].corr(df["CO2_Emissions_tonnes"])

    print(
        f"Temperature change: "
        f"{temperature_change:.4f} °C"
    )

    print(
        f"CO2 percentage change: "
        f"{co2_change:.2f}%"
    )

    print(
        f"Temperature-Rainfall correlation: "
        f"{temperature_rainfall:.3f}"
    )

    print(
        f"Temperature-CO2 correlation: "
        f"{temperature_co2:.3f}"
    )

    print("\nThe dataset shows the overall climate trends")
    print("for India during the analyzed period.")

    print(
        "\nNote: Correlation does not establish causation."
    )


# -------------------------------------------------
# Main Menu
# -------------------------------------------------
def main():

    print("==============================================")
    print(" CLIMATE CHANGE TREND ANALYSIS - INDIA")
    print("==============================================")

    try:
        df = load_data()
        df = preprocess_data(df)

        print(
            f"\nDataset loaded successfully."
            f" Valid observations: {len(df)}"
        )

    except Exception as e:
        print(f"\nError: {e}")
        return

    while True:

        print("\nProject: Climate Change Trend Analysis - India")
        print("Data Period: 2000-2024")
        print("Valid Observations:", len(df))
        
        print("\n========== MENU ==========")
        print("1. View Dataset")
        print("2. Descriptive Statistics")
        print("3. Correlation Analysis")
        print("4. Temperature Trend")
        print("5. Rainfall Trend")
        print("6. CO2 Emissions Trend")
        print("7. Temperature Regression Analysis")
        print("8. Generate All Visualizations")
        print("9. Final Interpretation")
        print("0. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            view_dataset(df)

        elif choice == "2":
            descriptive_statistics(df)

        elif choice == "3":
            correlation_analysis(df)

        elif choice == "4":
            temperature_trend(df)

        elif choice == "5":
            rainfall_trend(df)

        elif choice == "6":
            co2_trend(df)

        elif choice == "7":
            temperature_regression(df)

        elif choice == "8":
            generate_all(df)

        elif choice == "9":
            final_interpretation(df)

        elif choice == "0":
            print("\nThank you!")
            break

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()