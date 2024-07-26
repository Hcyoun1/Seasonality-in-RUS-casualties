import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from datetime import datetime
from scipy import stats
from collections import defaultdict
from eval_functions import *
from pmdarima.arima import auto_arima


if __name__ == "__main__":
    ru_equip_deaths_df = pd.read_csv("../data/russia_losses_equipment.csv")
    ru_pers_deaths_df = pd.read_csv("../data/russia_losses_personnel.csv")
    
    # resort for clarity
    ru_equip_deaths_df = ru_equip_deaths_df.sort_index(ascending=False)
    ru_equip_deaths_df.reset_index(inplace=True)
    ru_pers_deaths_df = ru_pers_deaths_df.sort_index(ascending=False)
    ru_pers_deaths_df.reset_index(inplace=True)

    for i in range(65):
        if pd.isna(ru_equip_deaths_df["vehicles and fuel tanks"].iloc[i]):
            ru_equip_deaths_df["vehicles and fuel tanks"].iloc[i] = ru_equip_deaths_df["fuel tank"].iloc[i] \
            + ru_equip_deaths_df["military auto"].iloc[i] 
    
    create_diff_columns(ru_equip_deaths_df, "vehicles and fuel tanks")
    create_diff_columns(ru_equip_deaths_df, "tank")
    create_diff_columns(ru_equip_deaths_df, "APC")
    create_diff_columns(ru_equip_deaths_df, "field artillery")
    create_diff_columns(ru_pers_deaths_df, "personnel")

    # Create columns for a Datetime object and for the int month and int year for calculations
    #these are only made once for each csv
    ru_equip_deaths_df["Dt_OBJ"] = ru_equip_deaths_df['date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))
    ru_equip_deaths_df["Int_month"] = ru_equip_deaths_df["Dt_OBJ"].dt.month
    ru_equip_deaths_df["Int_year"] = ru_equip_deaths_df["Dt_OBJ"].dt.year

    ru_pers_deaths_df["Dt_OBJ"] = ru_pers_deaths_df['date'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d'))
    ru_pers_deaths_df["Int_month"] = ru_pers_deaths_df["Dt_OBJ"].dt.month
    ru_pers_deaths_df["Int_year"] = ru_pers_deaths_df["Dt_OBJ"].dt.year

    # For season calculations we fake the year december is in so visualizations won't be a problem
    #actual datetime value will not change or interfere with data
    for i in range(len(ru_equip_deaths_df)):
            if ru_equip_deaths_df["Int_month"].iloc[i] == 12:
                ru_equip_deaths_df.iloc[i,-1] += 1 

    print("Initial cleaning of csv's done. enter diff for standard graphs,auto for auto arima,sarima for modeling, trend for trendlines")
    user_input = input()
    if user_input == "diff":
         plot_diff_columns(ru_equip_deaths_df,"vehicles and fuel tanks")
         plot_diff_columns(ru_equip_deaths_df, "tank")
         plot_diff_columns(ru_equip_deaths_df, "APC")
         plot_diff_columns(ru_equip_deaths_df, "field artillery")
         plot_diff_columns(ru_pers_deaths_df, "personnel")
    
    if user_input == "auto":
        second_input = input("input column name")
        if (second_input in ru_equip_deaths_df.columns):
            auto_arima_call(ru_equip_deaths_df,f"{second_input} diff")
        if (second_input in ru_pers_deaths_df.columns):
            auto_arima_call(ru_pers_deaths_df,f"{second_input} diff")


    if user_input == "sarima":
        print("Ensure that you have run auto first!")
        second_input = input("input column name")
        if (second_input in ru_equip_deaths_df.columns):
            forecaster = sarima_gen(ru_equip_deaths_df,f"{second_input} diff")
            y_train, y_test = test_train_split(ru_equip_deaths_df,f"{second_input} diff" )
        if (second_input in ru_pers_deaths_df.columns):
            forecaster = sarima_gen(ru_pers_deaths_df,f"{second_input} diff")
            y_train, y_test = test_train_split(ru_pers_deaths_df,f"{second_input} diff" )
        
        plt.plot()
        y_train.plot.line()
        plt.plot(forecaster.index,forecaster.values)
        y_test.plot.line()
        plt.savefig(f"../images/RU_SARIMAvActual_{second_input}.png")

    if user_input == "trend":
        trend_line_w_outliers(ru_equip_deaths_df, "vehicles and fuel tanks",draw_data=True, draw_trend=False )
        trend_line_w_outliers(ru_equip_deaths_df, "tank",draw_data=True, draw_trend=False )
        trend_line_w_outliers(ru_equip_deaths_df, "APC",draw_data=True, draw_trend=False )
        trend_line_w_outliers(ru_equip_deaths_df, "field artillery",draw_data=True, draw_trend=False )
        trend_line_w_outliers(ru_pers_deaths_df, "personnel",draw_data=True, draw_trend=False )


