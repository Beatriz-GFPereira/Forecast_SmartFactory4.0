import pandas as pd
import numpy as np
import copy
import random
from data_processing_prototype.data_loader import DataLoad


class DataProcess:
    # TODO: insert possibility of estimating promotions from demand
    def __init__(self,
                 dataloader=DataLoad(), target_freq="W-WED", periods_to_forecast=2, corr_limits=None,
                 business=None, start=None, interp=None, custom=None, group_clients=None, est_promo=None,
                 use_stored_sku=True, use_uploaded_sku=False, save_new_list_sku=False,
                 update_demand=True, demand_vars=None,
                 use_sales=True, update_sales=True, save_new_sales=False, sales_vars=None,
                 use_ext_for=False, update_ext_for=True, save_new_ext_for=False, ext_for_vars=None,
                 use_stock=True, update_stock=True, save_new_stock=False, stock_vars=None,
                 use_mkting=True, update_mkting=True, save_new_mkting=False, mkting_vars=None,
                 use_macro=False, update_macro=True, save_new_macro=False, macro_vars=None, ext_macro=False,
                 use_other=False, update_other=True, save_new_other=False, other_vars=None, ext_other=False,
                 ):

        # INPUTS SANITY CHECK
        ############################################################
        # Target frequency
        freqs = ["D", "B", "W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "SM",
                 "SMS", "M", "BM", "MS", "BMS"]
        if target_freq is not None:
            assert target_freq in freqs, f"{target_freq} must be a string among: {freqs}"
        # Number of periods to forecast
        if periods_to_forecast is not None:
            assert isinstance(periods_to_forecast, int), "periods_to_forecast must be an integer"
        else:
            # Default value
            periods_to_forecast = 1

        # Features reduction criteria
        if corr_limits is not None:
            if isinstance(corr_limits, dict):
                if corr_limits.get("y_lim") is not None and corr_limits.get("f_lim") is not None:
                    if 0 < corr_limits["y_lim"] < 1 and 0 < corr_limits["f_lim"] < 1:
                        pass
                    else:
                        raise ValueError("Values of 'corr_limits' must be: 0 < value < 1")
                else:
                    raise KeyError("'corr_limits' must be a dictionary like: {'y_lim': float, 'f_lim': float}")
            else:
                raise TypeError("'corr_limits' must be a dictionary like: {'y_lim': float, 'f_lim': float}")

        # Booleans:
        misc = [business, start, interp, custom, group_clients, est_promo]
        sku = [use_stored_sku, use_uploaded_sku, save_new_list_sku]
        demand = [update_demand]
        sales = [use_sales, update_sales, save_new_sales]
        ext_for = [use_ext_for, update_ext_for, save_new_ext_for]
        stock = [use_stock, update_stock, save_new_stock]
        mkting = [use_mkting, update_mkting, save_new_mkting]
        macro = [use_macro, update_macro, save_new_macro, ext_macro]
        other = [use_other, update_other, save_new_other, ext_other]

        bools = [*misc, *sku, *demand, *sales, *ext_for, *stock, *mkting, *macro, *other]
        for b in bools:
            if b is not None:
                assert isinstance(b, bool), f"{b} must be a boolean"

        # Time series pre-processing miscellaneous parameters - Default values
        if business is None: business = False
        if start is None: start = False
        if interp is None: interp = False
        if custom is None: custom = False
        if group_clients is None: group_clients = False
        if est_promo is None: est_promo = False

        # Sku parameters - Default values
        # TODO: finish inputs sanity checks and default values

        # ATTRIBUTES ASSIGNMENT
        ############################################################

        # Create attributes for target freq and number of future periods for forecasting
        self.target_freq = target_freq
        self.periods = periods_to_forecast

        # Create attributes for business day frequency, periods measured at start, interpolation for downsampling of TS
        self.busiess = business
        self.start = start
        self.interp = interp

        # Create attribute for list of skus to forecast
        self.sku_list = dataloader.get_sku_list_updated(loaded=use_stored_sku, use_new=use_uploaded_sku,
                                                        save=save_new_list_sku)
        # TODO: attribute for ext_forecast data
        # create attributes storing data for signal and each set of exogenous variables based on user choices
        self.demand = dataloader.get_demand(update=update_demand, selected_vars=demand_vars)
        self.sales = dataloader.get_sales(active=use_sales, update=update_sales, save_new=save_new_sales,
                                          selected_vars=sales_vars)
        self.stock = dataloader.get_stock(active=use_stock, update=update_stock, save_new=save_new_stock,
                                          selected_vars=stock_vars)
        self.marketing = dataloader.get_marketing(active=use_mkting, update=update_mkting,
                                                  save_new=save_new_mkting, selected_vars=mkting_vars)
        self.macro = dataloader.get_macro(active=use_macro, ext=ext_macro, update=update_macro,
                                          save_new=save_new_macro, selected_vars=macro_vars)
        self.other = dataloader.get_other(active=use_other, ext=ext_other, update=update_other,
                                          save_new=save_new_other, selected_vars=other_vars)

        # create attribute for a unique DataFrame with signal and all selected variables
        self.master_tab = self.create_master(signal=self.demand, exog_sets=[self.sales, self.stock, self.marketing,
                                                                            self.macro, self.other])

        # create attribute with master table filtered by selected skus
        self.master_tab_filtered = self.filter_by_sku(sku_filter=self.sku_list, df_to_filt=self.master_tab)

        # create attribute with dict of sku: master_tab to forecast each one separately
        self.master_dict = self.get_master_dict_by_sku(sku_keys=self.sku_list, master_df=self.master_tab_filtered)

        # create attribute with dict of sku with cat variables encoded
        self.master_dict_encoded = self.encode_categorical(master_dict=self.master_dict, custom=custom,
                                                           group_clients=group_clients)

        # process signal and each exogenous set of variables for every df in dict of skus encoded (in order!)
        self.process_demand()
        self.process_macro()
        self.process_marketing()
        self.process_stock()
        self.process_sales()

        # create attribute for dict of skus with each df processed and with exogenous features reduced
        self.master_dict_reduced = self.reduce_features(master_dict_encoded=self.master_dict_encoded,
                                                        corr_limits=corr_limits)

        # create attribute for dict of skus with each df resampled according to target freq and n_steps added
        self.master_dict_resampled = self.resampling(business=self.busiess, start=self.start, interp=self.interp)

        # Create attribute for dict resampled with feauture values of exogenous variables:
        self.master_dict_ex = self.add_future_exog()

    @staticmethod
    def create_master(signal, exog_sets=None):
        """
        Creates master table combining signal data with all exogenous variables selected by user, if any
        :param DataFrame signal: DataFrame containing at least signal data
        :param list exog_sets: List of DataFrames to be iteratively merged with master table
        :return: DataFrame master table
        """
        master = signal
        if len(exog_sets) > 0:
            for m in exog_sets:
                if len(m) > 0:
                    master = master.merge(m, how="left")
                else:
                    continue
        return master

    @staticmethod
    def filter_by_sku(sku_filter, df_to_filt):
        """
        Filters input variables by skus contained in the shortlist and stores result in DataFrame
        :param df_to_filt: DataFrame with column 'sku', to which filtering is applied
        :param list sku_filter: list of skus to be used as filter
        :param: DataFrame df_variable: DataFrame containing variables to be filtered, must have a column "sku"
        :return: DataFrame filtered by shortlist of skus in use
        :rtype: pd.DataFrame
        """

        # to improve search speed set sku column as index
        df_to_filt.set_index("sku", inplace=True)

        # filter df of variables for skus in shortlist
        selection = df_to_filt.loc[df_to_filt.index.isin(sku_filter)]
        return selection

    @staticmethod
    def get_master_dict_by_sku(sku_keys, master_df):
        """
        Construct dictionary to associate each sku to the corresponding time series data to forecast
        :param sku_keys: list of skus for which to perform forecast
        :param DataFrame master_df: df collecting time series data of signal + selected exogenous vars for
        all selected skus together
        :return: dict like sku: df. With df collecting time series data of signal + selected exogenous vars for sku key
        :rtype: dict with string as keys and DataFrames as values
        """

        # set index type to str to access it
        master_df.index = master_df.index.astype('str')

        # create list of DataFrames for each sku in the list
        lst_of_master_tabs = [master_df.loc[master_df.index == sku] for sku in sku_keys]

        # set a datetime index in each DataFrame and drop sku index
        for df in lst_of_master_tabs:
            df.set_index(keys="order_date", drop=True, inplace=True)

        # create dictionary excluding skus in list which have an empty df associated
        master_dict = {k: v for k, v in zip(sku_keys, lst_of_master_tabs) if k is not None and len(v) > 0}

        # allow only skus for which actually there are time series data in filter list
        for sku in sku_keys:
            assert sku in master_dict.keys(), f"{sku} sku not associated to any data, please check sku list in use!"

        return master_dict

    @staticmethod
    def encode_categorical(master_dict, custom=False, group_clients=False):
        """
        For each sku in dictionary:
        Encodes all the categorical variables in use through one-hot encoding, if no custom encoding is defined.
        If in use, encodes the variable "promotion_target" with a 0, 0.5 or 1 system, to take into account, when the
        promotion has a target, if this latter is general or specific and, if this is the case, whether it corresponds
        or not to the client which effectively made the order.
        Drops the variable client_id is dropped if no grouping criteria is specified.
        :param dict master_dict: dictionary in the form sku: df. With df collecting time series data of signal and all
                                selected exogenous variables
        :param bool custom: Determines whether or not custom encoding is applied (Default: False)
        :param bool group_clients: Determines whether or not client_id is grouped before dropping (Default: False)
        :return: dict in the form sku: df. With all categorical variables encoded
        :rtype: dict with string as keys and DataFrames as values
        """
        # columns of all possible categorical variables excluding dates
        encode_vars = ["sales_channel", "sales_agent", "client_location", "market_brand", "product_family",
                       "promotion_type", "promotion_family"]

        for sku in master_dict.keys():
            # Encode promotion target based on efficiency (if target is == client which made the order or not)
            if "promotion_target" in master_dict[sku].columns:
                # Create temporary col to check if client_id in target or promotion
                temp = [str(s[0]) in str(s[1]) for s in zip(master_dict[sku]["client_id"],
                                                            master_dict[sku]["promotion_target"])]
                master_dict[sku].insert(loc=1, column="temp", value=temp)

                # encode with 0.5 for target == general, -0.5 for target != client_id and 1 for target == client_id
                master_dict[sku].loc[(master_dict[sku]["promotion_target"] != "general") &
                                     (master_dict[sku]["temp"] == True), "promotion_target"] = 1
                master_dict[sku].loc[(master_dict[sku]["promotion_target"] != "general") &
                                     (master_dict[sku]["temp"] == False), "promotion_target"] = 0
                master_dict[sku].loc[master_dict[sku]["promotion_target"] == "general", "promotion_target"] = 0.5

                master_dict[sku].drop(columns="temp", inplace=True)

            if custom:
                pass
                # TODO: customizable labeling and/or encoding for other vars (business dep)
            else:
                # encode all other cat variables excluding dates and clients with one-hot encoding
                master_dict[sku] = pd.get_dummies(data=master_dict[sku],
                                                  columns=[col for col in master_dict[sku].columns
                                                           if col in encode_vars])

            # check if custom grouping criteria for clients is provided
            if group_clients:
                # TODO: insert grouping scheme for client_id (business dep)
                # drop client_id column for each df
                master_dict[sku].drop(columns="client_id", inplace=True)
            else:
                # drop client_id column for each dataframe
                master_dict[sku].drop(columns="client_id", inplace=True)

        return master_dict

    @staticmethod
    def reduce_features(master_dict_encoded, corr_limits=None):
        """
        For each sku in dictionary AFTER processing:
        Reduces number of exogenous features, keeping only the ones with higest explanatory power for signal and lowest
        multicolinearity.
        :param dict master_dict_encoded: dict in the form sku: df. With all variables already processed
        :param dict corr_limits: dictionary of limits for correlation filter. first high-pass filter for corr of exogenous
                                 variables with signal ("y_lim": value); second low pass filter for corr among the
                                 exogenous variables remaining ("f_lim": value). (Default: {"y_lim": 0.7, "f_lim": 0.7})
        :return: dict in the form sku: df. With number of exogenous features reduced to the minimum
        :rtype: dict with string as keys and DataFrames as values
        """
        # All possible columns and encodings in exogenous sets AFTER PROCESSING
        # TODO: include other sets of exogenous missing (other) and any other addition
        sales_cols = ["delivery_delay", "time_to_delivery", "time_to_sale", "sale_diff", "unitary_price",
                      "payment_timing"]
        stock_cols = ["stockout_quantity"]
        marketing_cols = ["current_promo", "past_promo", "future_promo", "lag_promo_start", "lag_promo_end",
                          "promotion_amount", "promotion_target"]
        macro_cols = ['gdp', 'industrial_index', 'trade_values_balance', 'stock_market_index', 'commodity_price',
                      'unemployment_rate']
        sales_prefix = ["sales_channel_", "sales_agent_", "client_location_"]
        marketing_prefix = ["market_brand_", "product_family_", "promotion_type_"]
        macro_prefix = ["delta_"]

        # copy input master_dict_encoded to avoid overwriting instance
        master_dict = copy.deepcopy(master_dict_encoded)

        # access each master tab with categorical vars already encoded in dictionary of skus in use
        for sku in master_dict.keys():
            # collect all sales features in each master tab
            total_features = [*[col for col in master_dict[sku].columns if col in sales_cols],
                              *[col for col in master_dict[sku].columns if col in stock_cols],
                              *[col for col in master_dict[sku].columns if col in marketing_cols],
                              *[col for col in master_dict[sku].columns if col in macro_cols],
                              *[col for col in master_dict[sku].columns if sales_prefix[0] in col],
                              *[col for col in master_dict[sku].columns if sales_prefix[1] in col],
                              *[col for col in master_dict[sku].columns if sales_prefix[2] in col],
                              *[col for col in master_dict[sku].columns if marketing_prefix[0] in col],
                              *[col for col in master_dict[sku].columns if marketing_prefix[1] in col],
                              *[col for col in master_dict[sku].columns if marketing_prefix[2] in col],
                              *[col for col in master_dict[sku].columns if macro_prefix[0] in col],
                              ]
            # if no exogenous features, then pass to next df
            if len(total_features) == 0:
                continue
            else:
                # default thresholds for corr with signal (maximize) and between features (minimize)
                if corr_limits is None:
                    corr_limits = {"y_lim": 0.1, "f_lim": 0.9}

                # corr array between signal and all features in use
                corr_y = master_dict[sku][total_features].corrwith(master_dict[sku]["y"],
                                                                   axis=0, method="pearson").abs()

                # extract features more correlated with signal based on threshold
                relevant_features = [row for row in corr_y.loc[corr_y >= corr_limits["y_lim"]].index]

                # corr matrix matrix only for most correlated features
                corr_matrix = master_dict[sku][relevant_features].corr(method="pearson").abs()
                # Select lower triangle of correlation matrix
                lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(np.bool))

                # TODO: include method to drop cols most correlated with others, not just first one
                # filter out features highly correlated
                for col in lower.columns:
                    if any(lower[col] >= corr_limits["f_lim"]):
                        relevant_features.remove(col)
                    else:
                        continue
                # drop all columns for non relevant features
                to_drop = [col for col in total_features if col not in relevant_features]
                master_dict[sku].drop(columns=to_drop, inplace=True)
                print(f"SKU: {sku} - DROPPING FOLLOWING FEATURES BASED ON CORRELATION: {to_drop}")
                # TODO: insert renaming to link with models?

                # TODO: insert both an alternative default method and option for custom criteria
                #       ex: linear regression for each set of exog variables to reduce to 1 value, then test corr

        return master_dict

    @staticmethod
    def detect_frequency(df, business=False, start=False):
        """
        Detects frequency of Datetime Index of a DataFrame from daily up to quarterly frequency,
        counting in days. Redistributes the DataFrame according to the detected frequency, summing values of
        the same period and filling missing periods with 0s.
        :param pd.DataFrame df: DataFrame to detect frequency of, Index must be DatetimeIndex.
        :param bool business: Determines whether use business frequency in use for days, months or quarters. (Default: False)
        :param bool start: Determines whether semi-months, months and quarters are taken from start or not. (Default: False)
        :return: Detected frequency, DataFrame with DatetimeIndex uniformed to detected frequency
        :rtype: pd.DateOffset, pd.DataFrame
        """
        # collect differences between consecutive periods in days
        delta_days = []
        for i in range(len(df.index) - 1):
            delta_days.append(df.index[i + 1] - df.index[i])
        delta_days = pd.Series(delta_days).astype("timedelta64[D]")
        delta_days = [i for i in delta_days if i > 0]
        # take minimum
        min_freq = min(delta_days)

        # Detect corresponding DateOffset string based on number of days of min_freq
        # Resample df for lowest frequency to fill missing periods with 0

        # daily freq
        if min_freq == 1:
            if business:
                freq = "B"
                df = df.resample(rule=freq, closed="left").sum()
            else:
                freq = "D"
                df = df.resample(rule=freq, closed="left").sum()
        elif 1 < min_freq < 7:
            freq = f"{int(min_freq)}D"
            df = df.resample(rule=freq, closed="left").sum()
        # weekly freq
        elif min_freq == 7:
            freq = "W"
            df = df.resample(rule=freq, closed="right").sum()
        elif 7 < min_freq < 13:
            freq = f"{int(min_freq)}D"
            df = df.resample(rule=freq, closed="left").sum()
        # semi-monthly freq
        elif 13 <= min_freq <= 15:
            if start:
                freq = "SMS"
                df = df.resample(rule=freq, closed="left").sum()
            else:
                freq = "SM"
                df = df.resample(rule=freq, closed="right").sum()
        elif 15 < min_freq < 28:
            freq = f"{int(min_freq)}D"
            df = df.resample(rule=freq, closed="left").sum()
        # monthly freq
        elif 28 <= min_freq <= 31:
            if business:
                if start:
                    freq = "BMS"
                    df = df.resample(rule=freq, closed="left").sum()
                else:
                    freq = "BM"
                    df = df.resample(rule=freq, closed="right").sum()
            else:
                if start:
                    freq = "MS"
                    df = df.resample(rule=freq, closed="left").sum()
                else:
                    freq = "M"
                    df = df.resample(rule=freq, closed="right").sum()
        elif 31 < min_freq < 59:
            freq = f"{int(min_freq)}D"
            df = df.resample(rule=freq, closed="left").sum()
        # bi-monthly freq
        elif 59 <= min_freq <= 61:
            if business:
                if start:
                    freq = "2BMS"
                    df = df.resample(rule=freq, closed="left").sum()
                else:
                    freq = "2BM"
                    df = df.resample(rule=freq, closed="right").sum()
            else:
                if start:
                    freq = "2MS"
                    df = df.resample(rule=freq, closed="left").sum()
                else:
                    freq = "2M"
                    df = df.resample(rule=freq, closed="right").sum()
        elif 61 < min_freq < 89:
            freq = f"{int(min_freq)}D"
            df = df.resample(rule=freq, closed="left").sum()
            # quarterly freq
        elif 89 <= min_freq <= 91:
            if business:
                if start:
                    freq = "BQS"
                    df = df.resample(rule=freq, closed="left").sum()
                else:
                    freq = "BQ"
                    df = df.resample(rule=freq, closed="right").sum()
            else:
                if start:
                    freq = "QS"
                    df = df.resample(rule=freq, closed="left").sum()
                else:
                    freq = "Q"
                    df = df.resample(rule=freq, closed="right").sum()
        else:
            raise ValueError("NO FREQUENCY IN DATA!")

        return freq, df

    @staticmethod
    def seasonality_dummies(df, freq=None):
        """
        Extracts number of month and of day of the week and one/hot encodes them if the frequency of data is higher
        than monthly and or weekly.
        :param pd.DataFrame df: dataframe of data already resampled and with future periods added
        :param str freq: frequency of data as pd.DateOffset
        :return: DataFrame of data with added columns for month number and day of the week
        >:rtype: pd.DataFrame
        """
        # Month one hot encoding (Jan=1, Dec=12)
        df["month"] = df.index.month
        # # Dayofweek one hot encoding (Mon=0, Sun=6)
        df["dayofweek"] = df.index.weekday

        # Check if frequency is higher than monthly and or weekly, otherwise no info value from seasonality dummies
        if freq in [*["D", "B"], *[f"{n}D" for n in range(7)], *[f"{n}B" for n in range(7)]]:
            df = pd.get_dummies(data=df, prefix=["m", "wd"], columns=["month", "dayofweek"])

        elif freq in [*["W", "W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "SM","SMS"],
                      *[f"{n}D" for n in range(7,28)]]:
            df = pd.get_dummies(data=df, prefix=["m"], columns=["month"])
            df.drop(columns=["dayofweek"], inplace=True)
        else:
            df.drop(columns=["month", "dayofweek"], inplace=True)

        return df

    ################################## GENERAL METHODS #####################################

    def resampling(self, business=False, start=False, interp=False):
        """
        For each sku in dictionary AFTER features reduction:
        Redistributes the Dataframe according to the target frequency (max: Daily) and manipulates values of each
        column accordingly, taking into account if the original data are measured in business days and/or at the start
        or at the end of each period.
        Distinguishes between:
        No/Mock Resampling case, where the target frequency is the equal to the original one or differs just in terms of
        measurement (start vs end). In this case no real redistribution happens.
        Upsampling case, where the target frequency is lower than the original one. In this case the values of
        quantitative variables are summed, while the mean of the values is taken for the others.
        Downsampling case, where the target frequency is higher than the original one. In this case, new values for each
        variable have to be created between each pair of value along the index. This is done either by redistributing
        uniformly the values of original frequency over the target one, or by polynomial interpolation through the mean
        values of the original frequency.
        :param bool business: Determines whether original frequency based on business days or not (Default: False)
        :param bool start: Determines whether periods of original frequency measured at start or not. (Default: False)
        :param bool interp: Determines whether to use interpolation as downsampling method (Default: False)
        :return: dict in the form sku: df. With Index and variables redistributed according to target frequency
        :rtype: dict with string as keys and DataFrames as values
        """
        # Copy input master_dict_encoded to avoid overwriting instance
        master_dict = copy.deepcopy(self.master_dict_reduced)

        # Distinguish all possible columns with quantity variables from the others
        quant_cols = ["y", "sale_diff", "stockout_quantity"]

        # access each master tab with categorical vars already encoded in dictionary of skus in use
        for sku in master_dict.keys():

            # Target frequency to be used
            t_freq = self.target_freq

            # Detect frequency of data and force it on df
            d_freq, master_dict[sku] = self.detect_frequency(df=master_dict[sku], business=business, start=start)
            print(f"SKU: {sku} - DETECTED FREQUENCY: {d_freq}")

            # If target frequency not specified it is set equal to frequency in the data
            if t_freq is None:
                t_freq = d_freq

            # Auxiliary date ranges to compare detected and target frequencies (exclude first to avoid date overlap)
            # create range of dates with detected freq starting from latest date of index
            det = pd.date_range(start=master_dict[sku].index[-1], periods=7, freq=d_freq)[1:]
            # create range of dates with target freq starting from latest date of index
            tar = pd.date_range(start=master_dict[sku].index[-1], periods=7, freq=t_freq)[1:]
            compare = sum(det < tar)

            # NO RESAMPLING NEEDED
            #####################################################
            if t_freq == d_freq:
                print(f"SKU: {sku} - NO RESAMPLING NEEDED")
                # continue

            # MOCK RESAMPLING (Change base within same frequency)
            #####################################################
            # Cases where both detected and target freq are weekly, week day of target freq prevails
            elif d_freq == "W" and t_freq in ["W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
                master_dict[sku] = master_dict[sku].resample(rule=t_freq, closed="right").sum()

            # Cases where detected and target freq are equal and differ just for start/end and/or business
            # TODO: keep SM --> SMS and SMS --> SM cases in check when testing
            # From end to start
            elif (d_freq == "M" and t_freq == "MS") or (d_freq == "BM" and t_freq == "BMS") or \
                    (d_freq == "SM" and t_freq == "SMS"):
                master_dict[sku] = master_dict[sku].resample(rule=t_freq, closed="left").bfill()

            # From-start to end
            elif (d_freq == "MS" and t_freq == "M") or (d_freq == "BMS" and t_freq == "BM") or \
                    (d_freq == "SMS" and t_freq == "SM"):
                master_dict[sku] = master_dict[sku].resample(rule=t_freq, closed="right").ffill()

            # UPSAMPLING NEEDED
            #######################################################
            elif compare > 0:
                print(f"SKU: {sku} - UPSAMPLING DATA...")
                # TODO: insert dictionary with aggregation criteria for each possible variable if needed
                # TODO: explore using moving avergage as aggregation criteria for value features
                # Assign df to temp variable for manipulation
                temp = master_dict[sku]

                # Distinguish closing method between periods measured at start VS measured at end t to avoid data loss
                if t_freq == "B" or start:
                    closed = "left"
                else:
                    closed = "right"

                # Aggregate demand signal first and drop it from temp df
                res = temp["y"].resample(rule=t_freq, closed=closed).sum()
                temp.drop(columns="y", inplace=True)

                # Sum other quantity columns and average others, then concatenate results
                for col in temp.columns:
                    if col in quant_cols[1:]:
                        ex = temp[col].resample(rule=t_freq, closed=closed).sum()
                    else:
                        ex = temp[col].resample(rule=t_freq, closed=closed).mean()
                    res = pd.concat([res, ex], axis=1)
                # Reassign resulting df to sku dictionary
                master_dict[sku] = pd.DataFrame(res)

            # DOWNSAMPLING NEEDED
            elif compare == 0:
                print("DOWNSAMPLING DATA...")

                # Assign df to temp variable for manipulation
                temp = master_dict[sku]
                # Get Offset object relative to 1 period of detected frequency
                freq_per = temp.index.freq

                # Calculate number of periods of target freq within detected freq and assign it to temp column
                per = list()
                for i in range(len(temp.index)):
                    # For periods measured at start
                    if start:
                        x = pd.date_range(start=temp.index[i], end=temp.index[i] + freq_per, freq=t_freq)
                    else:
                        x = pd.date_range(start=temp.index[i] - freq_per, end=temp.index[i], freq=t_freq)
                    # Distinguish case of downsampling to week periods for calculation purposes
                    if t_freq in ["W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
                        periods = len(x)
                    else:
                        periods = len(x) - 1
                    per.append(periods)

                # Assign result to temporary column for dowsapling operations
                temp["per"] = per

                # Divide quantity columns by number of periods and drop periods column afterwards
                # TODO: explore additional downsampling criteria for non-quantity columns other than ffill()
                for col in temp.columns:
                    if col in quant_cols:
                        temp[col] = temp[col] // temp["per"]
                temp.drop(columns="per", inplace=True)

                # For periods measured at start
                if start:
                    # TODO: finish this case, last period of data gets lost when downsampling...
                    raise UserWarning("DOWNSAMPLING NOT POSSIBLE FOR FREQUENCIES WITH start = True!")
                # For periods measured at end
                else:
                    # Interpolation of quantity data through points already divided
                    if interp:
                        # Distinguish case of downsampling to week periods for calculation purposes
                        if t_freq in ["W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
                            print(f"SKU {sku} - TESTING BEST WEEKLY FREQUENCY FOR INTERPOLATION...")

                            # Test different starting days of week to find best one to begin interpolation
                            for f in ["W-SUN", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT"]:
                                t = temp["y"].resample(rule=f, closed="left").interpolate(method="spline", order=2)
                                if t.isna().sum() == 0:
                                    t_freq = f
                                    print(f"SKU {sku} - {t_freq} WILL BE USED AS FREQUENCY")
                                    break
                        # Interpolate signal column first
                        res = temp["y"].resample(rule=t_freq, closed="left").interpolate(method="spline", order=2)
                        temp.drop(columns="y", inplace=True)
                        for col in temp.columns:
                            # Interpolate other quantity columns and backfill others, then concatenate results
                            if col in quant_cols[1:]:
                                ex = temp[col].resample(rule=t_freq, closed="left").interpolate(method="spline",
                                                                                                order=2)
                                # Force eventual negative values resulting from interpolation to zero
                                ex.where(cond=ex >= 0, other=0, inplace=True)
                            else:
                                ex = temp[col].resample(rule=t_freq, closed="left").bfill()
                            res = pd.concat([res, ex], axis=1)
                    else:
                        # Instead of interpolation, fill periods uniformly with quantities divided by number of periods
                        res = temp.resample(rule=t_freq, closed="left").bfill()
                # Reassign resulting df to sku dictionary
                master_dict[sku] = pd.DataFrame(res)

        return master_dict

    def add_future_exog(self, custom=None):
        """
        For each sku dictionary ALREADY resampled with target frequency:
        Adds one row to time series for each period to forecast and fills it according to predefined criteria.
        Default: demand signal column --> NaN values, quantitative vars --> 0, other cols --> random value between max
                 and min of last 15 observations.
        Adds new exogenous features to model seasonality: 12 columns to encode months and 6 columns to encode weekdays
        :param custom:
        :return: dict in the form sku: df. With additional future periods as rows and seasonality columns
        :rtype: dict with string as keys and DataFrames as values
        """
        # Copy input master_dict_encoded to avoid overwriting instance
        master_dict = copy.deepcopy(self.master_dict_resampled)

        # Distinguish all possible columns with quantity variables from the others
        quant_cols = ["y", "sale_diff", "stockout_quantity"]
        sales_prefix = ["sales_channel_", "sales_agent_", "client_location_"]
        marketing_prefix = ["market_brand_", "product_family_", "promotion_type_"]


        # Access each master tab already resampled in dictionary of skus in use
        for sku in master_dict.keys():

            # Columns with binary values form encoding
            sales_chn = [col for col in master_dict[sku].columns if sales_prefix[0] in col]
            sales_agt = [col for col in master_dict[sku].columns if sales_prefix[1] in col]
            client_loc = [col for col in master_dict[sku].columns if sales_prefix[2] in col]
            mkt_brand = [col for col in master_dict[sku].columns if marketing_prefix[0] in col]
            prod_fam = [col for col in master_dict[sku].columns if marketing_prefix[1] in col]
            promo_type = [col for col in master_dict[sku].columns if marketing_prefix[2] in col]

            # Create array of future dates for forecasting to use as index
            fut_per = master_dict[sku].index[-self.periods:].shift(periods=self.periods)
            # Create empty df with number of rows == to number of periods to forecast and same columns as resampled df
            fut_val = pd.DataFrame(data=np.empty([self.periods, len(master_dict[sku].columns)]), index=fut_per,
                                   columns=[col for col in master_dict[sku].columns])
            # Fill future periods
            ###########################################################
            if custom:
                pass
            # TODO: implement filling method(s) based on combination of rules for each variable and external APis

            # Default criteria: NaNs for demand signal, 0 for quant vars, random value for bin encoded, last value or
            # the others
            else:
                recent = 15
                for col in fut_val.columns:
                    if col == "y":
                        fut_val[col] = np.nan
                    elif col in quant_cols[1:]:
                        fut_val[col] = 0
                    else:
                        mn = master_dict[sku][col][-recent:].min()
                        mx = master_dict[sku][col][-recent:].max()
                        fut_val[col] = random.randint(int(mn), int(mx))

            # Append future values to resampled df
            master_dict[sku] = master_dict[sku].append(fut_val)

            # Add seasonality columns for months and weekdays
            master_dict[sku] = self.seasonality_dummies(df=master_dict[sku], freq=self.target_freq)

        return master_dict

    ################################## DEMAND METHODS #####################################

    def process_demand(self, default_drop=None):
        """
        For each sku, in dictionary:
        Modifies demand instance selecting signal column to be used as signal and associated column from sales.
        Drops the other pair (quantity vs units).
        Renames selected demand column as "y".
        :param list of str default_drop: column names to be dropped (Defaut: ["order_units", "actual_sold_units"])
        """
        # define default columns to be dropped and sanity check
        if default_drop is None:
            default_drop = ["order_units", "actual_sold_units"]
        else:
            assert default_drop in [["order_units", "actual_sold_units"], ["order_quantity", "actual_sold_quantity"]]
        # possible columns for signal
        demand_cols = ["order_quantity", "order_units"]

        # access each master tab with categorical vars already encoded in dictionary of skus in use
        for sku, df in self.master_dict_encoded.items():
            # check that master df has both order quantity and units as columns for demand
            if all(col in df.columns for col in demand_cols):

                # list of additional signal columns to be dropped for each df
                drop_cols = [col for col in df.columns if col in default_drop]

                # check if quantity and units have a fixed ratio and drop additional signal columns if True
                if round(round(df["order_quantity"] / df["order_units"]).mean()) == \
                        round(df["order_quantity"].sum() / df["order_units"].sum()):
                    print(f"SKU: {sku} - ONLY ONE SIGNAL NEEDED! DROPPING COLUMNS: '{drop_cols}'")
                    df.drop(columns=drop_cols, inplace=True)

                else:
                    # force user to select only one column as signal to forecast
                    raise ArithmeticError(f"SKU: {sku} - SIGNAL COLUMNS '{demand_cols}'"
                                          f"NOT CONSISTENT, PLEASE SELECT ONLY ONE!")
            else:
                continue
            # rename remaining signal column as "y" for SARIMAX and PROPHET processing
            df.rename(columns={"order_quantity": "y", "order_units": "y"}, inplace=True, errors="ignore")
        print()

    ################################## SALES METHODS #####################################

    def process_sales(self):
        """
        For each sku in dictionary (if corresponding column is in use):
        Substitutes "exp_delivery_date" and "sale_date" with the difference between them. If only one of the two is
        present, computes the difference between it and "order_date" index.
        Substitutes "actual_sold_quantity" and "actual_sold_units" with their difference with demand signal respectively.
        """
        # possible dates columns in sales
        dates_cols = ["exp_delivery_date", "sale_date"]

        # access each master tab with categorical vars already encoded in dictionary of skus in use
        for sku, df in self.master_dict_encoded.items():
            # check dates columns and calculate time lags (days to int)
            if all(col in df.columns for col in dates_cols):

                # force user to provide dates columns with no NaNs
                # TODO: delete nans rows and raise warning
                for col in dates_cols:
                    assert df[col].isna().sum() == 0, f"SKU: {sku} - None VALUE NOT ACCEPTED FOR '{col}'," \
                                                      f"PLEASE UPDATE DATA OR EXCLUDE IT FROM SELECTION"

                # calculate eventual delay on expected delivery (max = 0)
                df.insert(loc=1, column="delivery_delay",
                          value=(df["sale_date"] - df["exp_delivery_date"]).astype("timedelta64[D]"))
                df.drop(columns=dates_cols, inplace=True)
                print(f"SKU: {sku} - NEW COLUMN: 'delivery_delay'. DROPPING COLUMNS: {dates_cols}")

            #: TODO potentially to remove these cases: poor info power as exogenous variable
            elif "exp_delivery_date" in df.columns:
                # if only sale date present calculate time between order and sale
                df.insert(loc=1, column="time_to_delivery",
                          value=(df["exp_delivery_date"] - df.index).astype("timedelta64[D]"))
                df.drop(columns="sale_date", inplace=True)
                print(f"SKU: {sku} - NEW COLUMN: 'time_to_delivery'. DROPPING COLUMN: 'expected_delivery_date'")

            elif "sale_date" in df.columns:
                # if only sale date present calculate time between order and sale
                df.insert(loc=1, column="time_to_sale",
                          value=(df["sale_date"] - df.index).astype("timedelta64[D]"))
                df.drop(columns="sale_date", inplace=True)
                print(f"SKU: {sku} - NEW COLUMN: 'time_to_sale'. DROPPING COLUMN: 'sale_date'")

            else:
                continue

            # check difference between order and sale quantity (or units)
            if "actual_sold_quantity" in df.columns:
                # calculate eventual difference of quantity between sales and orders (min = 0)
                df.insert(loc=2, column="sale_diff", value=df["y"] - df["actual_sold_quantity"])
                df.drop(columns="actual_sold_quantity", inplace=True)
                print(f"SKU: {sku} - NEW COLUMN: 'sale_diff'. DROPPING COLUMNS: 'actual_sold_quantity'")

            elif "actual_sold_units" in df.columns:
                # calculate eventual difference of units between sales and orders (min = 0)
                df.insert(loc=2, column="sale_diff", value=df["y"] - df["actual_sold_units"])
                df.drop(columns="actual_sold_units", inplace=True)
                print(f"SKU: {sku} - NEW COLUMN: 'sale_diff'. DROPPING COLUMN: 'actual_sold_units'")
            else:
                continue
        print()

    ################################## EXTERNAL FORECASTS METHODS #####################################

    def process_ext_forecast(self):
        # TODO: expand if ext_forecast included in master_tab
        pass

    ################################## STOCK METHODS #####################################

    def process_stock(self):
        """
        For each sku in dictionary (if corresponding column is in use):
        Fills null value of "stockout_quantity" with 0
        """
        # TODO: expand for case: stockout sku =! from order sku and stockout date != from order date
        for sku, df in self.master_dict_encoded.items():
            # check dates columns and calculate time lags (days to int)
            if "stockout_quantity" in df.columns:
                df["stockout_quantity"].fillna(value=0, inplace=True)

    ################################## MARKETING METHODS #####################################

    def process_marketing(self, days_past=90, days_fut=90):
        """
        For each sku in dictionary (if corresponding column is in use):
        Computes difference in days between "order_date" index and "promotion_date_start" and "promotion_date_end"
        respectively. If promotions too far behind in past, too far ahead in future, or NaNs sets difference to 0.
        Creates 3 additional columns with one-hot encoding "past_promo", "current_promo", "future_promo".
        Drops "promotion_date_start" and "promotion_date_end" columns
        :param int days_past: Number of days beyond which past promotions are not considered
        :param int  days_fut: Number of days beyond which future promotions are not considered
        """
        # possible date columns in marketing
        dates_cols = ["promotion_date_start", "promotion_date_end"]

        # set max lag to consider past and future promos
        # TODO: expand to account for custom lags based on inputs frequency
        max_past_lag = pd.to_timedelta(days_past, unit="d")
        max_fut_lag = pd.to_timedelta(days_fut, unit="d")

        # access each master tab with categorical vars already encoded in dictionary of skus in use
        for sku, df in self.master_dict_encoded.items():
            # check dates columns to define promos status and calculate time lags (days to int)
            if all(col in df.columns for col in dates_cols):

                df.insert(loc=1, column="past_promo", value=0)
                df.insert(loc=1, column="current_promo", value=0)
                df.insert(loc=1, column="future_promo", value=0)
                df.insert(loc=1, column="lag_promo_end", value=(df.index - df[dates_cols[1]]).astype("timedelta64[D]"))
                df.insert(loc=1, column="lag_promo_start",
                          value=(df.index - df[dates_cols[0]]).astype("timedelta64[D]"))

                # set lag start to 0 for promos exceeding time limits to consider them (and for NaNs)
                df.loc[(df.index - df[dates_cols[1]]) > max_past_lag, "lag_promo_start"] = 0
                df.loc[(df[dates_cols[0]] - df.index) > max_fut_lag, "lag_promo_start"] = 0
                df["lag_promo_start"].fillna(value=0, inplace=True)
                # set lag end to 0 for promos exceeding time limits to consider them (and for NaNs)
                df.loc[(df.index - df[dates_cols[1]]) > max_past_lag, "lag_promo_end"] = 0
                df.loc[(df[dates_cols[0]] - df.index) > max_fut_lag, "lag_promo_end"] = 0
                df["lag_promo_end"].fillna(value=0, inplace=True)

                # distinguish from above cases where order_date == promotion_date_start OR promotion_date_end
                df.loc[(df.index - df[dates_cols[0]]) == 0, "lag_promo_start"] = 1
                df.loc[(df.index - df[dates_cols[1]]) == 0, "lag_promo_end"] = 1

                # binary encoding for past, current and future promos
                df.loc[(df.index > df[dates_cols[1]]) & (df["lag_promo_end"] != 0), "past_promo"] = 1
                df.loc[((df.index >= df[dates_cols[0]]) & (df.index <= df[dates_cols[1]])) & (df["lag_promo_end"] != 0),
                       "current_promo"] = 1
                df.loc[(df[dates_cols[0]] > df.index) & (df["lag_promo_start"] != 0), "future_promo"] = 1

                # fill NaNs of promo amount column: all indirect promos (ads, marketing campaigns etc.)
                df["promotion_amount"].fillna(value=0, inplace=True)

                # drop original promo dates columns
                df.drop(columns=dates_cols, inplace=True)

            elif any(col in df.columns for col in dates_cols):
                # force user to pass both start_date and end_date if promos have to be used
                assert True == False, f"SKU: {sku} - BOTH 'promotion_date_start' AND 'promotion_date_end' NEEDED TO" \
                                      f" BE USED AS EXOGENOUS VARIABLES, PLEASE MODIFY CURRENT SELECTION!"

            else:
                continue
        print()

    ################################## MACROECONOMICS METHODS #####################################

    def process_macro(self):
        """
        For each sku in dictionary (if corresponding column is in use):
        Checks if there are clients from countries outside of EU, if not drops "exchange_rate" column.
        Computes difference between value of variable for the business and for client, if any. Assigns result to new
        "delta_" column.
        """
        # possible columns in macro
        macro_cols = ['gdp', 'industrial_index', 'trade_values_balance', 'stock_market_index', "unemployment_rate"]
        suffix = "_client"

        # access each master tab with categorical vars already encoded in dictionary of skus in use
        for sku, df in self.master_dict_encoded.items():

            # check if no clients with different currency to be considered
            if "exchange_rate" in df.columns and df["exchange_rate"].mean() == 1:
                df.drop(columns="exchange_rate", inplace=True)
                print(f"SKU: {sku} - NO FOREIGN CURRENCIES. DROPPING COLUMN: 'exchange_rate'")

            # Drop client location if same for all
            location = [col for col in df.columns if "client_location_" in col]
            if  len(location) == 1:
                df.drop(columns=location, inplace=True)
                print(f"SKU: {sku} - NO DIFFERENCE IN LOCATIONS. DROPPING COLUMN: {location[0]}")

            for col in reversed(macro_cols):
                if col in df.columns and (col + suffix) in df.columns:
                    df.insert(loc=1, column=("delta_" + col), value=(df[col] - df[(col + suffix)]))
                    df.drop(columns=[col + suffix], inplace=True)
                else:
                    continue
        print()

    ################################## OTHER METHODS #####################################

    def process_other(self):
        # TODO: expand if other included in master_tab
        pass


t = DataProcess()
r = list(t.master_dict_reduced.values())
print(r[0])

d = list(t.master_dict_resampled.values())
print(d[0])

e = list(t.master_dict_ex.values())
print(e[0])

# d[1].to_csv("Test.csv", sep=";", decimal=",")

#
# print(list(t.master_dict_reduced.values())[0].columns)
# # t.reduce_sales()
# s = list(t.master_dict_reduced.values())[0].resample(rule="D").interpolate()
# print(s)
# n = 3
#
# idx = s.index[-n:].shift(periods=n)
#
# print(idx)
