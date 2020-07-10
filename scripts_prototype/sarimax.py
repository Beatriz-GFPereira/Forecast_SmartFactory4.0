import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import SimpleExpSmoothing
import itertools
from tqdm import tqdm

class sarimax:

    def __init__(self, data, freq=None, business=False, periods=None, error_target=None):
        # Remove yearly seasonality if possible.
        self.demand = self.yearly_seasonality(data['y'][:-periods])  # Pandas Dataframe with endog variable from history.
        self.exogenous = data.drop(columns="y")  # Pandas Dataframe with exog variables.
        self.freq = freq  # Any valid frequency for pd.date_range, such as 'D' or 'M'.
        self.business = business # True if data is only from business days.
        self.periods = periods  # Int number of periods to forecast forward.
        if error_target is None:
            self.error_target = -1
        else:
            self.error_target = error_target  # Float positive number

    def yearly_seasonality(self, data, period=365.25, fourier_order=10):
        """
        Provides Fourier series components with the specified frequency
        and order. Remove it from data if there is >=2 years of history.
        (Similar code to Fourier series from Prophet,
        see https://github.com/facebook/prophet/blob/16e632a6958bc1fbfcc67ed8628ba8c972df15db/python/fbprophet/forecaster.py)

        :param data: Pandas Dataframe
        :param period: Number of days of the period.
        :param fourier_order: Number of components. Can be increased when the seasonality
        needs to fit higher-frequency changes, and generally be less smooth.
        :return Pandas Dataframe without yearly seasonality
        """
        # TODO: fix to work also with frequencies other than days
        # Check 2 years (730 days) of history
        first = data.index[0]
        last = data.index[-1]
        if last - first < pd.Timedelta(days=730):
            seasonality = 0

        # Fourier series
        # else:
        #     # convert to days since epoch
        #     t = np.array(
        #         (data.index - datetime(1970, 1, 1)).total_seconds().astype(np.float)
        #     ) / (3600 * 24.)
        #     seasonality = np.column_stack([
        #         fun((2.0 * (i + 1) * np.pi * t / period))
        #         for i in range(fourier_order)
        #         for fun in (np.sin, np.cos)
        #     ])
        #     data = data - seasonality
        return data

    # SARIMAX Model
    #################################################################################

    def run_model (self, order, seasonal_order, exog_features=None):
        """
        Main function for training and testing.

        :param order: Tuple. (p, d, q) hyperparameters
        :param seasonal_order: (P, D, Q, S) hyperparameters
        :param exog_features: List of external features to be considered.
        :return: Predictions from SARIMAX and the baseline model,
        Dicitionary with models (SARIMAX, Simples Exponential Smoothing, Residuals),
        Dictionary with parameters (SARIMAX, Simples Exponential Smoothing)
        Errors from SARIMAX and Simples Exponential Smoothing on test
        """

        # Prepare data
        ############################################################################

        # Train set
        train_ratio = 0.7

        demand = self.demand
        cut = int(len(demand) * train_ratio)
        demand_train = demand[:cut]

        if exog_features is not None:
            # Define exogenous features dataset
            exogenous = self.exogenous[:-self.periods] # Exogenous from history (exlcude dates to forecast out of sample)
            exogenous = exogenous.loc[:, exog_features]
            exogenous_train = exogenous[:cut]  # Make training set
        else:
            exogenous = None
            exogenous_train = None

        # Train model
        ############################################################################
        # Fit model to training part of the series
        model_fit = SARIMAX(endog=demand_train,
                            exog=exogenous_train,
                            order=order,
                            seasonal_order=seasonal_order)
        # TODO: analyse other methods possibilities
        trained_model = model_fit.fit(method='lbfgs', disp=0)

        # Validate model
        ############################################################################
        # Apply trained model parameters to model for entire series
        model_predict = SARIMAX(endog=demand,
                                exog=exogenous,
                                order=order,
                                seasonal_order=seasonal_order)
        prediction_model = model_predict.filter(trained_model.params)

        # TODO: exclude negative predictions (transform data with Box-Cox?)

        # Use real observations for train
        train = prediction_model.get_prediction(start=demand.index[0], end=demand.index[cut-1],
                                               dynamic=False, exog=exogenous).predicted_mean

        # Residuals analysis
        #############################################################################
        residuals = trained_model.resid # y-yhat
        resid_model = None

        # Check autocorrelation
        if self.acorr_errors(residuals):
            # TODO: use a better model in the future
            resid_model = SimpleExpSmoothing(residuals).fit()
            resids_prediction = resid_model.fittedvalues
            train += resids_prediction

        # Testing
        #############################################################################
        # Use dynamic prediction for n steps on test
        n_steps = self.periods
        test = pd.Series()

        for i in range(demand[cut:].shape[0]):
            predictions = prediction_model.get_prediction(start=(cut - n_steps) + i, end=cut + i,
                                                          dynamic=True, exog=exogenous).predicted_mean[-1:]
            if resid_model is not None:
                predictions += resid_model.predict(start= cut + i, end=cut + i)
            test = test.append(predictions)

        predictions = train.append(test)

        # Post-processing
        ############################################################################
        # Baseline model
        train_exp_model = SimpleExpSmoothing(demand_train).fit()
        exp_model = SimpleExpSmoothing(demand).fit(smoothing_level=train_exp_model.params['smoothing_level'],
                                                   initial_level=train_exp_model.params['initial_level'])
        exp_smoothing = exp_model.fittedvalues

        # Create df with actual y and predictions to calcute the error
        forecast = pd.DataFrame()
        forecast['y'] = demand
        forecast['predictions'] = predictions
        forecast['exp_smoothing'] = exp_smoothing
        results_test = forecast[cut:]
        err = self.get_errors(results_test)

        # Save model object to forecast and params to future predicitons and to show on dashboards
        models = {'SARIMAX': prediction_model, 'exp_smoothing': exp_model, 'residuals': resid_model}
        params = {'SARIMAX': prediction_model.params, 'exp_smoothing': [exp_model.params.get(key) for key in ['smoothing_level', 'initial_level']]}

        return forecast['predictions'], forecast['exp_smoothing'], models, params, err

    def acorr_errors(self, resid, alpha=0.05):
        """
        Perform the Ljung-Box test to check autocorrelation of residuals.
        :param resid: Pandas Dataframe.
        :param alpha: Significance level.
        :return: Boolean. True if there's correlation.

        """
        # According to https://otexts.com/fpp2/residuals.html
        # the maximum lag being considered should be 2m (for seasonal data, where m = period)
        # Considering m as a month, we have
        max_month = pd.Timedelta(days=31)
        first = self.demand.index[0]
        for i in range(len(self.demand)):
            if (self.demand.index[i] - first) > max_month:
                m = i
                break
        maxlag = 2*(m+1)

        # For freq >= 'M'
        if maxlag < 10 : maxlag = 10 # Defined for non-seasonal data

        # The test is not good when the max lag is large so we limited to T/5
        if maxlag > len(self.demand)/5: maxlag = len(self.demand)/5

        # Ljung-Box test
        lb_test = acorr_ljungbox(resid, lags= maxlag, return_df=True)

        # Check significance level VS minimum p_value to reject Null HP
        check = alpha > lb_test["lb_pvalue"].min()

        return check

    def get_errors (self, df):
        """
        Measure errors based on predictions.

        :param df: Pandas DataFrame containing ['y', 'predictions',
            'exp_smoothing']
        :return: Pandas DataFrame.
        """
        # TODO: clarify denominator formula (provided from Navigator solution)
        # equivalent to WAPE?
        #denominator = df['y'].where(df['y'] > df['y'].mean() * 0.1, df['y'].mean() * 0.1)

        error = dict()

        # APE
        df['APE_pred'] = np.abs(df['y'] - df['predictions']) / df['y']
        df['APE_exp'] = np.abs(df['y'] - df['exp_smoothing']) / df['y']

        # if demand = 0 => APE = 1
        df['APE_pred'].replace(to_replace=float('inf'), value=1, inplace=True)
        df['APE_exp'].replace(to_replace=float('inf'), value=1, inplace=True)

        # WAPE with weighting of forecasts
        error['predictions'] = sum(df['APE_pred'] * df['predictions']) / sum(df['predictions'])
        error['exp_smoothing'] = sum(df['APE_exp'] * df['exp_smoothing']) / sum(df['exp_smoothing'])

        return error

    def grid_search (self, p, d, q, P, D, Q, S, exog_features=None):
        """
        Grid search SARIMAX hyperparameters.

        :param p: Order of the auto-regressive model.
        :param d: Degree of differencing.
        :param q: Order of the moving-average model.
        :param P: Seasonal order of the auto-regressive model.
        :param D: Seasonal degree of differencing.
        :param Q: Seasonal of the moving-average model.
        :param S: Seasonality number of periods.
        :param exog_features: List of external features to be considered.
        :return: List with results.
        """

        results = list()

        # Create list of hyperparemeter iterables
        # Exclude from all possible combinations
        orders = list(itertools.product(p, d, q))
        print(orders)
        seasonal_orders = list(itertools.product(P, D, Q, S))
        print(seasonal_orders)
        seasonal_orders = [(0, 0, 0, 0)] + [i for i in seasonal_orders if
                                            sum(i[:3]) != 0 and i[3] != 0]
        print(seasonal_orders)
        all_iterations = [i for i in list(itertools.product(orders, seasonal_orders))
                          if i != ((0, 0, 0), (0, 0, 0, 0))]
        print(all_iterations)

        for o, s in tqdm(all_iterations):
            forecast, forecast_es, models, params, err = self.run_model(o, s, exog_features=exog_features)
            if err['predictions'] > err['exp_smoothing']:
                results.append((*o, *s, forecast_es, {'exp_smoothing': models['exp_smoothing']}, params['exp_smoothing'], np.nan, err['exp_smoothing']))
                print("Worst predictions than Exponential Smoothing!")
                print()
            else:
                results.append((*o, *s, forecast, {'SARIMAX': models['SARIMAX'],'residuals': models['residuals']},
                                params['SARIMAX'], err['predictions'], err['exp_smoothing']))

        return results

    def check_best_hyperparameters(self,df):
        """
        Find best set of hyperparameters.

        :param df: DataFrame with grid search results.

        :return: Pandas DataFrame with best hyperparameters.
        """

        df_best = pd.DataFrame(columns=df.columns)
        idxmin = df['error_predictions'].idxmin()
        if np.isnan(idxmin):
            idxmin = df.index[0]
        row = df.loc[idxmin]
        df_best = df_best.append(row)

        return df_best

    def run (self):
        """
        Main function to run grid search and get the best model
        :return: Dataframe with 'forecast' (Dataframe), 'model' (dictionary),
        'error_predictions' (float),'error_exp_smoothing' (float), 'additional_features' (string).
        Boolean: True if error target reached.
        """

        # Set hyperparameter intervals
        # According to https: // www.rdocumentation.org / packages / forecast / versions / 8.12 / topics / auto.arima
        lst_p = range(0, 6)
        lst_d = range(0, 3)
        lst_q = range(0, 6)

        lst_P = range(0, 3)
        lst_D = [0, 1]
        lst_Q = range(0, 3)

        # TODO: expand for different frequencies
        if self.freq is 'D' or 'B':
            lst_S = [0, 7]
        else:
            lst_S = [0]

        if len(self.exogenous.columns) == 0:
            suffix = ''
            # Grid-search best hyperparameters
            results = self.grid_search(lst_p, lst_d, lst_q, lst_P, lst_D, lst_Q, lst_S)

            # Results post-processing
            results_df = pd.DataFrame(results)
            results_df.columns = ['p', 'd', 'q', 'P', 'D', 'Q', 'S', 'forecast',
                                  'model', 'params',
                                  'error_predictions',
                                  'error_exp_smoothing']
            best = self.check_best_hyperparameters(results_df)
            best['additional_features'] = None
            suffix = ''
            best_params = best.drop(columns=['forecast', 'model', 'additional_features'])
        else:
            # Define external variables to be iteratively included
            # Possibilities: 'sales', 'marketing', 'macroeconomics', 'stock',...
            features_list = list(self.exogenous.columns)

            # Build all possible combinations of external features. Order is not relevant
            features_combinations = []
            for i in range(1, len(features_list) + 1):
                features_combinations += list(itertools.combinations(features_list, i))

            # Grid-search best hyperparameters per set of external features
            best_results = list()
            for f in features_combinations:
                # Grid-search
                results = self.grid_search(lst_p, lst_d, lst_q, lst_P, lst_D, lst_Q, lst_S, exog_features=f)
                print(results)

                # Results post-processing
                results_df = pd.DataFrame(results)
                results_df.columns = ['p', 'd', 'q', 'P', 'D', 'Q', 'S', 'forecast',
                                      'model', 'params',
                                      'error_predictions',
                                      'error_exp_smoothing']

                best_result_df = self.check_best_hyperparameters(results_df)
                best_result_df['additional_features'] = '_'.join(f)
                best_results.append(best_result_df)

            best_results = pd.concat(best_results)
            print(best_results)
            best = self.check_best_hyperparameters(best_results)
            suffix = best['additional_features'].iloc[0]

            best_params = best.drop(columns=['forecast', 'model', 'additional_features'])

        # Save best parameters and realtive forecasting error to csv to be uploaded in future
        best_params.to_csv('img_results_' + suffix + '_best.csv', sep=";", index=False)

        # Store in-sample forecast error of best performing model
        if best['error_predictions'].values.sum() == 0:
            # Exponential Smoothing error as best error if no model is better than Exponential Smoothing
            error = best['error_exp_smoothing'].values[0]
        else:
            error = best['error_predictions'].values[0]

        best = best[['forecast', 'model', 'error_predictions','error_exp_smoothing', 'additional_features']]

        return best.iloc[0], error < self.error_target

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
                if freq in ["W","W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
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

    # Future Results
    def get_forecasts(self, model, exog_features = None):
        """
        Get future Forecast.

        :param model: Dictionary with the model(s) object(s)
        :param exog_features: String with exogenous features
        :return out_forecast: Dataframe with forecast,
                forecast_ppo: Dataframe with forecast (matching freq with PPO module)
        """

        demand = self.demand
        n_steps = self.periods - 1

        if exog_features is not None:
            exog_features = exog_features.split('_')
            exogenous = self.exogenous.loc[:, exog_features]
            exogenous = exogenous[-self.periods:]

        exogenous = None

        if len(model) > 1:
            forecast_model = model['SARIMAX']
            resids_model = model['residuals']
        else:
            forecast_model = model['exp_smoothing']
            resids_model = None


        # Out of sample forecasting
        out_forecast = forecast_model.predict(start=len(demand), end=len(demand) + n_steps)

        if resids_model is not None:
            out_forecast+= resids_model.predict(start=len(demand), end=len(demand) + n_steps)

        # Resampling freq of data to match PPO module
        forecast_ppo = self.redistribute_forecasts(freq=self.freq, forecasts=out_forecast,
                                                   business=self.business)

        return round(out_forecast), forecast_ppo