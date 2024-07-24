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
import pickle

def check_stationarity(timeseries): 
    # Perform the Dickey-Fuller test 
    result = adfuller(timeseries, autolag='AIC') 
    p_value = result[1] 
    print(f'ADF Statistic: {result[0]}') 
    print(f'p-value: {p_value}') 
    print('Stationary' if p_value < 0.05 else 'Non-Stationary') 

def create_diff_columns(df_name, column_name):
    """
    Create a new dataframe column for the daily differences in loss rates 
    args: pandas dataframe, str name for column
    """
    df_name[f"{column_name} diff"] = df_name[f"{column_name}"].diff(1)
    df_name[f"{column_name} diff"][0] = df_name[f"{column_name}"][0]

def plot_diff_columns(df_name, column_name):
    df_name[f"{column_name} diff"].plot.line()
    plt.savefig(f"../images/RU_casualties_{column_name}.png")
    plt.close()

def return_seasonal_dataframes(df_name, column_name):
    '''Only return season dataframes
    Args: dataframe, str for column
    returns four dataframes in order spring, summer, fall, winter
    '''
    springdf = df_name[(df_name["Int_month"] == 3) | (df_name["Int_month"] == 4) |
                                                (df_name["Int_month"] == 5)]
    summerdf = df_name[(df_name["Int_month"] == 6) | (df_name["Int_month"] == 7) |
                                                (df_name["Int_month"] == 8)]
    falldf = df_name[(df_name["Int_month"] == 9) | (df_name["Int_month"] == 10) |
                                                (df_name["Int_month"] == 11)]
    winterdf = df_name[(df_name["Int_month"] == 12) | (df_name["Int_month"] == 1) |
                                                (df_name["Int_month"] == 2)]
    
    return springdf,summerdf,falldf,winterdf

def create_seasonal_comparisons(df_name, column_name):
    # split the given dataframe into seasons, plot the dataframes
    # run T-Tests against spring and print results
    #df_name["Int_month"] = df_name["Dt_OBJ"].dt.month
    ru_deaths_summerdf = df_name[(df_name["Int_month"] == 6) | (df_name["Int_month"] == 7) |
                                                (df_name["Int_month"] == 8)]
    ru_deaths_falldf = df_name[(df_name["Int_month"] == 9) | (df_name["Int_month"] == 10) |
                                                (df_name["Int_month"] == 11)]
    ru_deaths_winterdf = df_name[(df_name["Int_month"] == 12) | (df_name["Int_month"] == 1) |
                                                (df_name["Int_month"] == 2)]
    ru_deaths_springdf = df_name[(df_name["Int_month"] == 3) | (df_name["Int_month"] == 4) |
                                                (df_name["Int_month"] == 5)]
    
    ru_deaths_summerdf[f"{column_name}"].plot()
    ru_deaths_springdf[f"{column_name}"].plot()
    ru_deaths_falldf[f"{column_name}"].plot()
    ru_deaths_winterdf[f"{column_name}"].plot()
    print("Summer mean:", ru_deaths_summerdf[f"{column_name}"].mean())
    print("Spring mean:" ,ru_deaths_springdf[f"{column_name}"].mean())
    print("Fall mean:" ,ru_deaths_falldf[f"{column_name}"].mean())
    print("Winter mean:" ,ru_deaths_winterdf[f"{column_name}"].mean())
    
    print("Winter vs Spring:" , stats.ttest_ind(ru_deaths_winterdf[f"{column_name}"]
                                                ,ru_deaths_springdf[f"{column_name}"]))
    print("Summer vs Spring:" ,stats.ttest_ind(ru_deaths_summerdf[f"{column_name}"]
                                                ,ru_deaths_springdf[f"{column_name}"]))
    print("Fall vs Spring:" ,stats.ttest_ind(ru_deaths_falldf[f"{column_name}"]
                                                ,ru_deaths_springdf[f"{column_name}"]))
    
def trend_line_w_outliers(df_name, column_name, draw_data=False,color=("blue","yellow","green","grey"), draw_trend=True ):
    ''' slightly misleading, but calls for draw trend line do all the work
    split DF into seasons, with options to draw the data to a plot, the trend line, or both
    color expects a 4-tuple of string color names for plotting
    Args: dataframe, str, boolean, 4-len tuple, boolean
    '''
    
    ru_deaths_summerdf = df_name[(df_name["Int_month"] == 6) | (df_name["Int_month"] == 7) |
                                                (df_name["Int_month"] == 8)]
    ru_deaths_falldf = df_name[(df_name["Int_month"] == 9) | (df_name["Int_month"] == 10) |
                                                (df_name["Int_month"] == 11)]
    ru_deaths_winterdf = df_name[(df_name["Int_month"] == 12) | (df_name["Int_month"] == 1) |
                                                (df_name["Int_month"] == 2)]
    ru_deaths_springdf = df_name[(df_name["Int_month"] == 3) | (df_name["Int_month"] == 4) |
                                                (df_name["Int_month"] == 5)]
    plt.figure(figsize=(12, 6))

    summer_std = ru_deaths_summerdf[f"{column_name}"].std()
    fall_std = ru_deaths_falldf[f"{column_name}"].std()
    winter_std = ru_deaths_winterdf[f"{column_name}"].std()
    spring_std = ru_deaths_springdf[f"{column_name}"].std()
    if draw_data:
        plt.plot(df_name["Dt_OBJ"],df_name[f"{column_name}"])
    
    print("Spring:")
    draw_trend_line(ru_deaths_springdf,column_name,color[0],draw_trend)
    print("Summer:")
    draw_trend_line(ru_deaths_summerdf,column_name,color[1],draw_trend)
    print("Fall:")
    draw_trend_line(ru_deaths_falldf,column_name,color[2],draw_trend)
    print("Winter:")
    draw_trend_line(ru_deaths_winterdf,column_name,color[3],draw_trend)

def draw_trend_line(df_name, column_name, color_name, draw_trend=True):
    #Attach december to the next year for plotting purposes
    outlier_counter = 0


    yearly_data = df_name.groupby(df_name["Int_year"])
    df_name_mean = df_name[f"{column_name}"].median()
    
    for year, data in yearly_data:
        # print("yearly", year, data.head())
        df_name_std = data[f"{column_name}"].std()
        df_name_mean = data[f"{column_name}"].mean()
        # print(f"{year}:STD : {df_name_std}")
        # print(f"{year}:mean : {df_name_mean}")
        
        first_point = data.iloc[0]
        last_point = data.iloc[-1]
        if draw_trend:
            plt.plot([first_point["Dt_OBJ"], last_point["Dt_OBJ"]], 
                    [first_point[f'{column_name}'], last_point[f'{column_name}']], 
                    marker='o', color=color_name)
        
        for points in range(len(data)):
            
            if (data[f"{column_name}"].iloc[points] > (df_name_std * 2) + df_name_mean):
                #or (df_name[f"{column_name}"].iloc[points] < df_name_mean - df_name_std)
                # print(" Outlier at ",data["Int_month"].iloc[points],  data[f"{column_name}"].iloc[points])
                outlier_counter += 1
                plt.plot(data["Dt_OBJ"].iloc[points],data[f"{column_name}"].iloc[points], marker='x',color="black")

    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Trend Lines for Each Month')
    plt.legend()
    plt.grid(True)
    print(f"Total season outliers: {outlier_counter}")

def seasonal_decomposition(df_name, column_name):

    equip_analysis = df_name.loc[0::,["Dt_OBJ", f"{column_name} diff"]]
    equip_analysis.set_index("Dt_OBJ", inplace=True)
    decompose_result = seasonal_decompose(equip_analysis,period=90)
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    resid_season = decompose_result.resid
    decompose_result.plot()
    return trend, seasonal, resid_season, decompose_result

def auto_arima_call(df_name, column_name):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    #Split the train and test
    X = df_name[f"{column_name}"]
    tscsv = TimeSeriesSplit()
    for i, (train_index, test_index) in enumerate(tscsv.split(X)):
        X_train = df_name.iloc[train_index]
        y_train = df_name.iloc[train_index]["APC diff"]
        X_test = df_name.iloc[test_index]
        y_test = df_name.iloc[test_index]["APC diff"]
    
    Arima_model= auto_arima(y_train, start_p=1, 
                        start_q=1, 
                        max_p=8, 
                        max_q=8, 
                        start_P=0, 
                        start_Q=0, 
                        max_P=8, 
                        max_Q=8,
                        m=30, 
                        seasonal=True, 
                        trace=True, 
                        d=1, D=1, 
                        error_action='warn', 
                        suppress_warnings=True, 
                        random_state = 20, 
                        n_fits=30)
    with open('../data/arima.pkl', 'wb') as pkl:
        pickle.dump(Arima_model, pkl)
    

def sarima_gen():
    with open('../data/arima.pkl', 'rb') as pkl:
        pickle.dump(Arima_model, pkl)
    