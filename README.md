# Seasonality-in-RUS-casualties
Attempting to describe seasonal impact in the Ukraine war


# Introduction
This project aims to analyze and visualize the losses of Russian equipment and personnel over time. It leverages various statistical and machine learning techniques to preprocess, analyze, and visualize the data, providing insights into trends and patterns.  It aims to be a first step in casualty analysis of the RUS-UKR war by splitting the war into seasons, and attemping to quantify the impact of generalized weather with regards to equipment and personnel loss rates.
Additionally this serves as an experiment in applying SARIMA models on a dataset where very few external variables are known to us.

# Dataset
dataset sourced from https://www.kaggle.com/datasets/piterfm/2022-ukraine-russian-war, which is updated weekly from UKR MOD numbers

# Usage
To run the analysis, execute the Rus_ukr_evaluation.py script:

python Rus_ukr_evaluation.py
Ensure the data files are located in the ../data/ directory.

# Features
Data Loading and Preprocessing: Reads and preprocesses data from CSV files.


Statistical Analysis: Performs Dickey-Fuller test to check stationarity, calculates daily differences, and creates seasonal dataframes.


Visualization: Plots various time series data and residuals to visualize trends and patterns.


Modeling: Utilizes ARIMA and SARIMAX models for time series forecasting.

# Dependencies
The project requires the following Python libraries:

numpy
pandas
matplotlib
statsmodels
sklearn
datetime
scipy
pmdarima
Configuration
Ensure the following configuration is in place:

Data files (russia_losses_equipment.csv and russia_losses_personnel.csv) should be in the ../data/ directory.
Update any file paths as necessary in the Rus_ukr_evaluation.py script.

# Documentation
Functions in eval_functions.py

check_stationarity(timeseries): Performs the Dickey-Fuller test to check stationarity.

create_diff_columns(df_name, column_name): Creates columns for daily differences.


plot_diff_columns(df_name, column_name): Plots difference columns.

return_seasonal_dataframes(df_name, column_name): Returns seasonal dataframes.

# Rus_ukr_evaluation.py
Loads and preprocesses equipment and personnel loss data.
Creates difference columns and datetime-related columns.
Visualizes various aspects of the data.

# Examples
Run the analysis and visualize the results:


# Troubleshooting


# Limitations
Obviously with a dataset that is in essence just losses over time, there's a strong limit to how valuable this is on its own.  UKR MOD sources are widely regarded by western governments to overestimate Russian losses, but without a similar daily or weekly casualty update dataset from a Western source, this is about as good as we can get in the fog of war.  The SARIMA auto-arima is VERY computationally expensive and requires a large amount of memory.  Personally i've seen okay looking results from a s value of 90 from the SARIMA model, but with how expensive the search algorithm is, you may get better results looking through the PAC and autocorrection plots and doing it by hand.

# Future development
Next step would be find more datasets to better account for other variables at play.  any kind of geolocational casualty data, even if not completely consistent may help.  pulling weather data across known axis of advance would be another good start.

# Contributors
Hcyoun1
