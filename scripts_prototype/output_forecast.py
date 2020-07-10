import numpy as np
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

################################################################################
# Constants
################################################################################

DEMAND = pd.read_csv('order_history_mandatory.csv', sep=';')
CLIENT_FORECAST= pd.read_csv('client_prediction.csv', sep=';')
FIGSIZE = (18, 8)


################################################################################
# Functions
################################################################################

def plot_time_series(demand, forecast=None, moving_average=None,
                     predict=None, vline=None, title=None, xlabel=None,
                     ylabel=None, filename=None, n_steps=1):
    """
    Plots time series, KAM forecast, moving average and predictions.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    df = pd.DataFrame()
    if vline is not None:
        plt.axvline(x=vline, color='grey', linestyle='--')

    # Plot data points
    demand.plot(ax=ax, style='o-', label='Observed')
    df['demand'] = demand

    #Plot forecast (client)
    if forecast is not None:
        forecast.plot(ax=ax, style='o', label='Client Forecast')
        df['forecast'] = forecast

    # Plot predictions
    if predict is not None:
        predict.plot(ax=ax, style='o--',
                     label=str(n_steps) + '-step-ahead forecast')
        df[str(n_steps) + '-step-ahead-forecast - train'] = predict.loc[:vline]
        df[str(n_steps) + '-step-ahead-forecast - test'] = predict.loc[vline:][
                                                           1:]

    # Plot data moving average
    if type(moving_average) is not None:
        moving_average.plot(ax=ax, style='x--', label='Moving Average')
        df['moving_average'] = moving_average

    plt.legend()
    if filename:
        plt.savefig(filename + '.png')
        df.to_excel(filename + '.xlsx')
    plt.show()


def plot_residuals(prediction_model, filename, title, sku):
    """
    Presents residuals plots as available in SARIMAX.
    """
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set(title=title)
        ax.set_axis_off()
        prediction_model.plot_diagnostics(fig=fig, figsize=FIGSIZE)
        if filename:
            plt.savefig(filename + '_residuals.png')
    except Exception:
        print(sku, 'residuals error!')