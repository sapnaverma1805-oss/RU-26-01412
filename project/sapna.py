import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def load_weather_data(filename):
    """Loads CSV data using Pandas, supporting files with or without headers."""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found. Please enter a valid file name.")
        return None

    # Peek at the first line to check if a header row exists
    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().lower()

    if "date" in first_line and "temp" in first_line:
        df = pd.read_csv(filename)
        df.columns = ["Date", "Temp", "Humidity", "Rainfall"]
    else:
        df = pd.read_csv(
            filename, header=None, names=["Date", "Temp", "Humidity", "Rainfall"]
        )

    # Ensure numeric columns are properly typed
    for col in ["Temp", "Humidity", "Rainfall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_data(df, verbose=False):
    """Removes duplicate rows and fills missing values using column means."""
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df_cleaned)

    # Calculate mean temperature using NumPy before filling missing values
    valid_temps = df_cleaned["Temp"].dropna().to_numpy()
    mean_temp = float(np.mean(valid_temps)) if len(valid_temps) > 0 else 0.0

    missing_temp_count = int(df_cleaned["Temp"].isna().sum())
    df_cleaned["Temp"] = df_cleaned["Temp"].fillna(mean_temp)

    # Fill any missing Humidity or Rainfall values with their respective means
    for col in ["Humidity", "Rainfall"]:
        if df_cleaned[col].isna().any():
            col_mean = float(np.mean(df_cleaned[col].dropna().to_numpy()))
            df_cleaned[col] = df_cleaned[col].fillna(col_mean)

    if verbose:
        msg_parts = [f"Duplicates removed: {duplicates_removed}"]
        if missing_temp_count > 0:
            msg_parts.append(f"Missing Temp filled with mean: {mean_temp:.2f}")
        print(" | ".join(msg_parts))
        print(
            f"Cleaned rows: {len(df_cleaned)} | Line chart 'Temperature Trend' displayed"
        )

    return df_cleaned


def show_summary(df):
    """Calculates and displays summary statistics using NumPy."""
    cleaned_df = clean_data(df, verbose=False)

    temps = cleaned_df["Temp"].to_numpy()
    rainfall = cleaned_df["Rainfall"].to_numpy()

    avg_temp = np.mean(temps)
    max_temp = np.max(temps)
    min_temp = np.min(temps)
    std_temp = np.std(temps)
    total_rain = np.sum(rainfall)

    if len(cleaned_df) == 1:
        print(
            f"Average Temp: {avg_temp:.2f} C | Max: {int(max_temp)} | "
            f"Min: {int(min_temp)} | Std Dev: {std_temp:.2f}"
        )
    else:
        max_date = cleaned_df.loc[cleaned_df["Temp"].idxmax(), "Date"]
        min_date = cleaned_df.loc[cleaned_df["Temp"].idxmin(), "Date"]
        print(
            f"Average Temp: {avg_temp:.2f} C | Max: {int(max_temp)} ({max_date}) | "
            f"Min: {int(min_temp)} ({min_date}) | Total Rainfall: {int(total_rain)} mm"
        )


def plot_charts(df):
    """Generates a Line Chart for Temperature Trend and a Bar Chart for Rainfall."""
    # 1. Line Chart: Temperature Trend
    plt.figure(figsize=(8, 4))
    plt.plot(df["Date"], df["Temp"], marker="o", color="tab:red", linewidth=2)
    plt.title("Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. Bar Chart: Daily Rainfall
    plt.figure(figsize=(8, 4))
    plt.bar(df["Date"], df["Rainfall"], color="tab:blue")
    plt.title("Daily Rainfall")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    print("=== Weather Data Analyzer ===")
    filename = input("Enter CSV file name (e.g., weather.csv): ").strip()

    df = load_weather_data(filename)
    if df is None:
        return

    print("\nMenu:")
    print("1. Summary")
    print("2. Clean & Plot")
    choice = input("Select an option (1 or 2): ").strip()

    if choice == "1":
        show_summary(df)
    elif choice == "2":
        cleaned_df = clean_data(df, verbose=True)
        plot_charts(cleaned_df)
    else:
        print("Invalid option selected. Please choose 1 or 2.")


if __name__ == "__main__":
    main()

