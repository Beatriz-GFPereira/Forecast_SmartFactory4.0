# Forecast_SmartFactory4.0
Prototype of demand forecast module from SMART FACTORY 4.0 project.

# Best model Algorithm formulation and implementation
The main algorithm routine is implemented in the best_model file.
### Variables:
ERROR_TARGET - Max error target threshold for best selection defined in implementation phase.

MODELS_ENABLED - List of all forecasting models defined by the user.

sku_dict - Dictionary with all SKUs and his data (from DataProcess class).

periods - Period to predict in the future defined by the user.

freq - Data frequency

## Models
The SARIMAX is implemented in the sarimax class, and Prophet in the prophet class.

This classes have a main function in common run() wich will return the best result from a grid search.

If you want to run just one model with fix parameters use run_model() method.

The get_forecasts() method will return the predictions for future dates and will resample data to match with PPO (Production Plan Optimization) module frequency.
