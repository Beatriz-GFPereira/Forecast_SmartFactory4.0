from fbprophet import Prophet
import numpy as np
import pandas as pd
import itertools

class prophet:

    def __init__(self, data, freq=None, business=False, periods=None, error_target=None, market_share=None):
        self.data = data # Pandas Dataframe with all the data
        self.demand = data[:-periods]  # Pandas Dataframe with all variables from history (y and exog features).
        self.freq = freq  # Any valid frequency for pd.date_range, such as 'D' or 'M'.
        self.periods = periods  # Int number of periods to forecast forward.
        self.business = business  # True if data is only from business days.
        self.market_share = market_share # Pandas Dataframe with capacity to specify the maximum achievable point
        # due to the business scenarios or constraints. In this case, the market size.
        if error_target is None:
            self.error_target = -1
        else:
            self.error_target = error_target # Float positive number gave by the client or from a past model.

    # Process impact features
    @staticmethod
    def holidays_format(data):
        """
         Pandas DataFrame with columns holiday (string) and ds (date type)
         and optionally columns lower_window and upper_window which specify a
         range of days around the date to be included as holidays.
        """
        # TODO: NOT INCLUDED ON PROTOTYPE
        return data

    # Prophet Model
    ##################################################################################
    def run_model(self, trend_growth='linear', trend_flexibility=0.05, changepoints=None,
                  holidays=None, additional_features=None):
        """
            Main function for training and testing.

            :param trend_growth: String 'linear' or 'logistic' to specify a linear or logistic
            trend. And 'flat' for no trend.
            :param trend_flexibility: Int number between 0 and 1. Helps adjust the strength of the trend.
             Increasing the value will make the trend more flexible.
            :param changepoints: List of dates at which to include potential changepoints. If
            not specified, potential changepoints are selected automatically.
            :param holidays: Pandas Dataframe. The holidays feature allows inputs of customized
             recurring events which occur on irregular schedules.
            :param additional_features: List of external features to be considered as a regressor.
            :return Dataframe with predictions, model object and the mean error.
        """

        # Prepare data to Prophet requirements
        ###################################################################
        # Create column ds with index (demand date)
        demand = self.demand
        demand['ds'] = demand.index

        # Adding additional features to the model as a regressor
        if additional_features is not None:
            # Select only the exog features in the list
            demand = demand[['y','ds'] + list(additional_features)]

        # Check holidays
        if holidays is not None:
            holidays = self.holidays_format(holidays)

        # Specify the carrying capacity (maximum) in a column 'cap'
        if self.market_share is not None:
            demand['cap'] = self.market_share[:len(demand)]
        # Saturating Minimum in a column 'floor'
        demand['floor'] = 0

        # Train model
        #####################################################################
        if self.freq is not 'D' or 'B':
            model = Prophet(growth=trend_growth, changepoint_prior_scale=trend_flexibility,
                       changepoints=changepoints, holidays=holidays, weekly_seasonality=False)
        else:
            model = Prophet(growth=trend_growth, changepoint_prior_scale=trend_flexibility,
                       changepoints=changepoints, holidays=holidays)

        # Adding regressors to the linear part of the model
        if additional_features is not None:
            for feature in additional_features:
                model.add_regressor(feature)

        # Fit
        model.fit(demand)

        # Predict Test
        ####################################################################

        # Dataframe to predict (including all history)
        # predict() from Prophet only receives dates as a column 'ds' (and other external variables)
        demand_predict = demand.drop(columns=['y'])

        # Dataframe with yhat (y predicted) and the components of the model
        forecast = model.predict(demand_predict)
        forecast.set_index('ds', inplace=True) # Set dates as index again

        # Diagnostics - Time series cross validation to measure forecast error using historical data
        train_period = int(len(demand) * 0.5) # Initialize with half of the sample
        test_ratio = 0.3
        horizon = int(train_period * test_ratio)

        error = self.evaluation(model, training_period=train_period, forecast_horizon=horizon)

        return forecast['yhat'], model, error

    def model_components(self,forecast):
        """
        Give the values of the model components on prediction dates.

        :param forecast: Prediction dataframe.
        :return Dataframe with trend and seasonal components.
        """

        # Remove yhat and components not predicted (with only zeros)
        components = forecast.drop(columns=['yhat','yhat_lower','yhat_upper'])
        components = components.loc[:, (components != 0).any(axis=0)]

        return components

    # Errors and the model robustness
    def evaluation(self, model, forecast_horizon, training_period=None, prediction_period=None):
        """
        Measure errors based on cross validation (SHF).

            :param model: Prophet model.
            :param training_period: String with pd.Timedelta compatible style.
            The size of the initial training period.
            :param prediction_period: String with pd.Timedelta compatible style.
            Spacing between cutoff dates.
            :param forecast_horizon: String with pd.Timedelta compatible style.
            :return Mean of the error from each fold
        """

        # Passing periods to string with Timedelta format
        initial = pd.Timedelta(training_period, unit=self.freq)
        horizon = pd.Timedelta(forecast_horizon, unit=self.freq)

        # Cross validation
        from fbprophet.diagnostics import cross_validation
        df_cv = cross_validation(model, initial=initial, period=prediction_period, horizon=horizon)
        print(df_cv['cutoff'].unique())
        from fbprophet.diagnostics import performance_metrics
        df_p = performance_metrics(df_cv)
        print(df_p.head())

        # WAPE per fold
        def ewm_error(fold):
            ape = np.abs(fold.y - fold.yhat) / fold.y
            ape.replace(to_replace=float('inf'), value=1, inplace=True)
            wape = sum(ape * fold.yhat) / sum(fold.yhat)
            return wape
        errors_fold = df_cv.groupby('cutoff').apply(ewm_error)

        # Weighted mean of fold errors
        mean_wape = errors_fold.ewm(alpha=0.1, adjust=True).mean()[-1:].iloc[0]

        std = np.std(errors_fold)

        return mean_wape

    def check_best(self, df):
        """
        Find min error.

        :param df: DataFrame with grid search results.
        :return: Dataframe with best model.
        """
        df_best = pd.DataFrame(columns=df.columns)
        idxmin = df['error_predictions'].idxmin()
        if np.isnan(idxmin):
            idxmin = df.index[0]
        row = df.loc[idxmin]
        df_best = df_best.append(row)

        return df_best

    def run(self):
        """
        Main function to get best model

        :return: Dataframe with 'forecast' (Dataframe), 'model' (prophet object),
        'error_predictions' (float), 'additional_features' (string).
        Boolean: True if error target reached.
        """

        # In case there's no exogenous
        if len(self.demand.columns) == 1:
            forecast, model, error = self.run_model()
            best = pd.DataFrame([[forecast, model, error, None]],
                                columns=['forecast', 'model', 'error_predictions','additional_features'])

        else:
            # Define external variables to be iteratively included
            # Possibilities: 'sales', 'marketing', 'macroeconomics', 'stock',...
            exogenous = self.demand.drop(columns="y")
            features_list = list(exogenous.columns)

            # Build all possible combinations of external features. Order is not relevant
            features_combinations = []
            for i in range(1, len(features_list) + 1):
                features_combinations += list(itertools.combinations(features_list, i))

            # Create a model per set of external features
            results = list()
            for f in features_combinations:
                forecast, model, error = self.run_model(additional_features=f)
                results.append((forecast, model, error, '_'.join(f)))

            results_df = pd.DataFrame(results)
            results_df.columns = ['forecast', 'model', 'error_predictions','additional_features']
            best = self.check_best(results_df)

        return best.iloc[0], best['error_predictions'].values[0] < self.error_target


    # Future Forecast
    #######################################################################

    # Process future forecast to PPO module (if Forecast freq != PPO freq)
    @staticmethod
    def redistribute_forecasts(freq, forecasts, business=None, method=None):
        """
        Distributes out-of-sample forecast values over number of days present in the frequency used for
        forecasting, to be processed by PPO module. Handles business day frequency as well.
        :param str freq: frequency of demand data used for forecasting represented as pd.DateOffset
        :param pd.Series forecasts: array of out-of sample forecast values
        :param bool business: whether or not business-day frequency has to be used in transformation.
                              If None, default value is used (Default: False)
        :param str method: name of method to distribute forecast values over number of days in the frequency in
                           use. if None, default value is used (Default: "uniform")
        :return: Dataframe of re-distributed forecast values with dates and empty column relative to sku
        :rtype pd.DataFrame
        """
        # Sanity check on possible inputs for business frequency and transformation method
        method_options = ["uniform"]
        if method is not None:
            assert method in method_options, f"{method} must be a string among: {method_options}"
        else:
            method = "uniform"
        # Check if business day frequency is needed or not
        if business is not None:
            assert business in [True, False], f"{business} must be a boolean value"
        else:
            print("NO daily frequency specified for re-distribution of forecasts:'D' is going to be used")
            business = False
        if business:
            daily_freq = "B"
        else:
            daily_freq = "D"

        # Initialize dataframe structure to be transferred to PPO module
        forecast_ppo = pd.DataFrame(columns=["sku", "forecast_qty", "start_date", "end_date"])

        # If the frequency of forecasting is day or business day, no transformation needed
        if freq == "D" or freq == "B":
            forecast_ppo["forecast_qty"] = [round(v) for v in forecasts.values]
            forecast_ppo["start_date"] = forecasts.index
            forecast_ppo["end_date"] = forecasts.index + pd.to_timedelta(1, unit="d")

        else:
            # Uniformly divide forecasts values by the number of days in the forecasting frequency
            if method == "uniform":
                # Initialize aux variables to collect new dates and values
                aux_date = list()
                aux_qty = list()

                # Weekly forecast values
                if freq in ["W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
                    for value in range(len(forecasts)):
                        # Create number of days in each month STARTING FROM forecast date
                        start = forecasts.index[value]
                        end = start + pd.DateOffset(days=7)
                        n_days = list(pd.date_range(start=start, end=end, freq=daily_freq, closed="left"))
                        # Concatenate each month in auxiliary lis
                        aux_date += n_days
                        for day in range(len(n_days)):
                            # Divide each forecast value by number of days in each month
                            aux_qty.append(round(forecasts.iloc[value] / len(n_days)))

                # Monthly forecast values
                elif freq == "MS" or freq == "M":
                    for value in range(len(forecasts)):
                        # Create number of days in each month STARTING FROM forecast date
                        if freq == "MS":
                            start = forecasts.index[value]
                            end = start + pd.DateOffset(months=1)
                            n_days = list(pd.date_range(start=start, end=end, freq=daily_freq, closed="left"))
                        # Create number of days in each month ENDING ON forecast date (month-end)
                        elif freq == "M":
                            end = forecasts.index[value]
                            m = end.month
                            y = end.year
                            # Leap year case
                            if (y % 4 == 0 or y % 400 == 0) and m == 2:
                                d = 2
                            # February normal year
                            elif m == 2:
                                d = 3
                            # April, june, september and november
                            elif m in [4, 6, 9, 11]:
                                d = 1
                            # All other months
                            else:
                                d = 0
                            start = end - pd.DateOffset(months=1) + pd.DateOffset(days=d)
                            n_days = list(pd.date_range(start=start, end=end, freq=daily_freq, closed="right"))
                        # Concatenate each month in auxiliary list
                        aux_date += n_days
                        for day in range(len(n_days)):
                            # Divide each forecast value by number of days in each month
                            aux_qty.append(round(forecasts.iloc[value] / len(n_days)))

                # Half-monthly forecast values:

                # Store values and dates in DataFrame
                forecast_ppo["forecast_qty"] = aux_qty
                forecast_ppo["start_date"] = aux_date
                forecast_ppo["end_date"] = forecast_ppo["start_date"] + pd.to_timedelta(1, unit="d")
            else:
                # TODO add transformation method different from uniform
                pass

        return forecast_ppo


    def get_forecasts(self, model, exog_features=None):
        """
        Get future Forecast.

        :param model: Model object
        :param exog_features: String with exogenous features
        :return out_forecast: Dataframe with forecast,
                forecast_ppo: Dataframe with forecast (matching freq with PPO module)
        """

        # Out of sample forecasting
        ############################################################################
        # Days to extend into the future
        future = model.make_future_dataframe(periods=self.periods, freq=self.freq, include_history=False)

        # Select only the exog features in the list
        if exog_features is not None:
            exog_features = exog_features.split('_')
            print(exog_features)
            exogenous = self.data.loc[:, exog_features]
            exogenous = exogenous[-self.periods:].reset_index(drop=True)
            future = future.join(exogenous)

        if self.market_share is not None:
            market_share = self.market_share[-self.periods:].reset_index(drop=True)
            future['cap'] = market_share
        future['floor'] = 0
        
        # Prediction
        out_forecast = model.predict(future)

        forecast_ppo = self.redistribute_forecasts(freq=self.freq, forecasts=out_forecast,
                                                   business=self.business)

        return round(out_forecast), forecast_ppo