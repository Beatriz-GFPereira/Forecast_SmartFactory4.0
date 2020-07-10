from scripts_prototype.sarimax import *
from scripts_prototype.Prophet import *
from data_processing_prototype.data_processor import *
import multiprocessing as mp
from functools import partial
import itertools


# Max error target threshold for best selection defined in implementation phase
ERROR_TARGET = None


# List of all forecasting models enabled, possibilities: 'sarimax', 'prophet','prophet+sarimax', 'lstm'
MODELS_ENABLED = ['sarimax', 'prophet']
# Dictionary: key - Model name, item - run() or get_forecast() function
models_dict = {'sarimax': partial(sarimax),
               'prophet': partial(prophet),
               'prophet+sarimax': partial(sarimax)}

sku_dict = DataProcess().master_dict_resampled
periods = DataProcess().periods

# Initialize dictionary to store best performing model for each sku
best_model_sku = {}

# Dataframe to store redistributed out-of-sample forecasts for each sku
output_ppo = pd.DataFrame(columns=['sku', 'forecast_qty', 'start_date', 'end_date'])

for sku, df in sku_dict.items():
    freq = pd.infer_freq(df.index)
    # TODO: manage frequency in data-processor
    # TODO: add model obj and exogenous features to run get_forecasts
    results = pd.DataFrame(index=['model'], columns=['forecast', 'error'])
    for model in MODELS_ENABLED:
        if model == 'prophet+sarimax':
            if not results['model'].get('prophet', False):
                best, target_reached = prophet(data=df, freq=freq, periods=periods,
                                                     error_target=ERROR_TARGET).run()
                if 'prophet' in MODELS_ENABLED:
                    results.loc['prophet'] = best['forecast'], best['error_predictions']
                    if target_reached: break
                df = df['y'] - best['forecast']['yhat']
            else:
                df = df['y'] - results.loc['prophet']['forecast']['yhat']
        if model == 'prophet' and 'prophet' in results.index: continue
        best, target_reached = models_dict[model](data=df, freq=freq, periods=periods, error_target = ERROR_TARGET).run()
        results.loc[model] = best['forecast'], best['error_predictions']
        if target_reached: break
    best_model_sku[sku] = results.sort_values(by='error_predictions', ascending=True).iloc[0]


    ############## WARNING: CODE NOT RUNNING WITH GET_FORECASTS ##############
    # TODO: save best_model_sku and use it to upload best model already trained
    out_forecast, forecast_ppo = models_dict[best_model_sku[sku]['model']](data=df, freq=freq, periods=periods) \
        .get_forecasts(model=best_model_sku[sku]['model_obj'])

    # Assign sku to df of redistributed out-of-sample forecast output
    forecast_ppo['sku'] = sku
    # Append redistributed out-of-sample forecast for each sku to dataframe
    output_ppo = output_ppo.append(forecast_ppo, ignore_index=True)

# Save
output_ppo.to_csv("orders_forecast.csv", sep=";", decimal=",", index=False)






"""
# INPUTS
# Possibilities: 'sarimax', 'prophet', 'lstm'
MODELS_ENABLED = ['sarimax', 'prophet']
models_dict = {'sarimax': partial(sarimax(data=DEMAND, freq=freq, periods=periods).run()),
               'prophet': partial(prophet(data=DEMAND, freq=freq, periods=periods).run())}
model_inputs = [[1]]

n_cpu = mp.cpu_count() - 1
n_models = len(MODELS_ENABLED)

# RUN ALL MODELS IN PARALLEL
with mp.Pool(processes=min(n_cpu, n_models)) as pool:
    results = [pool.starmap(models_dict[m], model_inputs) for m in MODELS_ENABLED]
    for res in results:
        print(res)
"""