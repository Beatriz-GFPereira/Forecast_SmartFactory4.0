import os
import pandas as pd


class DataLoad:
    """
    Class collecting all necessary data for demand forecasting, including: demand (signal), all the sets of
    exogenous variables in use (sales, external demand forecasts, stock, marketing, macroeconomics, other), the list of
    product codes (sku) for which forecasting has to be performed and the forecasting model(s) to be applied. The
    current structure assumes a 2-folders structure: a database folder storing all historic data and a new_data folder,
    which receives the new data uploaded by the user. All data are stored and uploaded in CSV files. Depending on the
    inputs, the class gives the possibility to exclude some or all sets of exogenous variables as well as to use
    historical data only, new data only or to update historical data with new ones. This inputs are
    stored in variables of the class DataProcess, where all the methods of this class are actually called.

    Constructor parameters are the followings:

    :param str new_data_folder: path to folder containing new data files uploaded
    :param str file_new_orders: filename for new demand data
    :param str file_new_sales: filename for new sales data (if any)
    :param str file_new_ext_forecast: filename for new external demand forecast data (if any)
    :param str file_new_stock: filename for new stock data (if any)
    :param str file_new_marketing: filename for new marketing data (if any)
    :param str file_new_macro: filename for new macroeconomics data (if any)
    :param str file_new_other: filename for new other relevant data, depending on business activity (if any)
    :param str database_folder: path to folder containing historical data files
    :param str file_orders: filename for historical demand data
    :param str file_sales: filename for historical sales data (if any)
    :param str file_ext_forecast: filename for historical external demand forecast data (if any)
    :param str file_stock: filename for historical stock data (if any)
    :param str file_marketing: filename for historical marketing data (if any)
    :param str file_macro: filename for historical macroeconomics data (if any)
    :param str file_other: filename for historical other relevant data , depending on business activity (if any)
    :param str file_shortlist: filename for list of skus for which to perform forecasting. Filename is the same
                               both for historical and new list.
    :param str forecasting_model: name of forecasting model to be used

    ASSUMPTIONS ON INPUT DATA FILES ARE THE FOLLOWING:

    Demand data files includes the following columns, in order:
    - order_date: date in which order was received from client in the format DD/MM/YYYY
    - sku: "stock keeping unit", the unique code identifying the product ordered on that date. Treated as str
    - client_id: the unique code identifying the client who ordered that product on that date. Treated as str
    - ordered_quantity: quantity of specific product ordered by that client on that date.
    - ordered_units: units of specific product ordered by that client on that date.

    Data files for sets of Exogenous Variables including: Sales, External Forecasts, Stock and Macroeconomics must
    include the columns: order_date, sku and client_id in order to be used.
    Data files for sets of Exogenous Variables including: Marketing must include the columns: order_date and sku.
    Data files for sets of Exogenous Variables including: Other must include AT LEAST the columns: order_date and sku.
    These columns serve as KEYS on which to MERGE each set of exogenous variable data with demand data.

    # TODO : find solution to deal with this for exogenous sets that cannot receive forecasts through external APIs
    Data of each set of Exogenous Variables MUST INCLUDE the forecasts for each single variable in the set.
    The number of forecasts to be included corresponds is equal to the number of future periods to be forecasted for
    the demand.
    """

    # TODO: Create SQL/NoSQL database queries for historic data
    def __init__(self,
                 new_data_folder="dummy_user_uploads",
                 file_new_orders="new_orders_mandatory.csv",
                 file_new_sales="new_ex_sales.csv",
                 file_new_ext_forecast=None,  # from client
                 file_new_stock=None,  # "new_ex_stock.csv",
                 file_new_marketing="new_ex_marketing.csv",
                 file_new_macro="new_ex_macroeconomics.csv",
                 file_new_other=None,  # "new_ex_other.csv",
                 database_folder="dummy_file_db",
                 file_orders="orders_mandatory.csv",  # "orders_mandatory - TS.csv"
                 file_sales="ex_sales.csv",
                 file_ext_forecast=None,  # from client
                 file_stock="ex_stock.csv",
                 file_marketing="ex_marketing.csv",
                 file_macro="ex_macroeconomics.csv",
                 file_other=None,  # "ex_other.csv",
                 file_shortlist="shortlist.csv",
                 forecasting_model="SARIMAX"
                 ):

        # inputs format sanity check
        sources = [new_data_folder, database_folder, file_new_orders, file_orders, file_shortlist]
        for s in sources:
            assert isinstance(s, str), f"'{s}' must be a string"

        vars = [file_new_sales, file_new_ext_forecast, file_new_stock, file_new_marketing, file_new_macro,
                file_new_other, file_sales, file_ext_forecast, file_stock, file_marketing, file_macro,
                file_other]
        for v in vars:
            if v is not None:
                assert isinstance(v, str), f"'{v}' must be a string"
            else:
                continue

        assert forecasting_model.upper() in ["SARIMAX", "PROPHET", "LSTM"], \
            "forecasing_model must be a string"

        # create file paths attributes for new uploads and file database
        self.new_data_folder = f"..\{new_data_folder}"
        self.database_folder = f"..\{database_folder}"

        # create path attributes for new and historic data for signal
        self.path_new_orders = os.path.join(self.new_data_folder, file_new_orders)
        self.path_orders = os.path.join(self.database_folder, file_orders)

        # crete file attributes for new data files for exogenous features
        self.file_new_sales = file_new_sales
        self.file_new_ext_forecast = file_new_ext_forecast
        self.file_new_stock = file_new_stock
        self.file_new_marketing = file_new_marketing
        self.file_new_macro = file_new_macro
        self.file_new_other = file_new_other

        # crete file attributes for historic data files for exogenous features
        self.file_sales = file_sales
        self.file_ext_forecast = file_ext_forecast
        self.file_stock = file_stock
        self.file_marketing = file_marketing
        self.file_macro = file_macro
        self.file_other = file_other

        # create raw input attributes for other user inputs
        self.file_new_sku = os.path.join(self.new_data_folder, file_shortlist)
        self.file_sku = os.path.join(self.database_folder, file_shortlist)
        self.forecasting_model_to_use = forecasting_model

    @staticmethod
    def fix_duplicates(df, drop=True):
        """
        Deletes or signals duplicated lines in a DataFrame. If option to signal is chosen, it raises an assertion error if
        any duplicate is actually encountered
        :param Dataframe df: Data on which to check duplicates
        :param bool drop: Determines whether to automatically drop the duplicates after being detected (Default: True)
        :return: Data with no duplicates; Assertion Error
        :rtype: pd.DataFrame
        """
        if df.duplicated(keep="first").sum() > 0:
            if drop:
                print()
                print("WARNING: The following entries are duplicated and they will be deleted!")
                print(df.iloc[df.duplicated(keep="first").values])
                df.drop_duplicates(keep="first", inplace=True, ignore_index=True)
                return df
            else:
                dup = df.iloc[df.duplicated(keep="first").values]
                assert False, "The following entries appear to be already registered, please CHECK" \
                              "before proceeding!" + "\n" + f"{dup}"
        else:
            print("CHECK OK: No duplicated entries.")
            return df

    ################################## GENERAL METHODS #####################################

    def get_sku_list_updated(self, loaded=True, use_new=False, save=False):
        """
        Returns a list of SKUs identifying the products on which to perform forecasting.
        :param bool loaded: Determines if only list loaded from database has to be used (Default: True)
        :param bool use_new: Determines whether only new list uploaded has to be used (True) or it has to be
        combined with the existing list in the database (Default: False)
        :param bool save: Determines whether updated list used has to be saved in the database or not (Default: False)
        :return: List of selected skus
        :rtype: List
        """
        if loaded:
            sl_updated = pd.read_csv(self.file_sku, sep=";")
            print("SAVED LIST OF SKUs IN USE:")
            print()
        else:
            if use_new:
                sl_updated = pd.read_csv(self.file_new_sku, sep=";")
                print("NEW UPLOADED LIST OF SKUs IN USE:")
                print()
            else:
                print("UPDATING SAVED LIST OF SKUs WITH NEW DATA UPLOADED...")
                print()
                sl = pd.read_csv(self.file_sku, sep=";")
                new_sl = pd.read_csv(self.file_new_sku, sep=";")
                sl_updated = pd.concat([sl, new_sl], axis=0, sort=False, ignore_index=True)

        sl_updated = self.fix_duplicates(sl_updated, drop=True)
        sl_updated.sort_values(by="sku", axis=0, ascending=True, inplace=True, ignore_index=True)
        if save:
            sl_updated.to_csv(os.path.join(self.database_folder, self.file_sku), sep=";", decimal=",", index=False)
            print("SAVED LIST OF SKUs UPDATED IN DATABASE!")
        sl = list(sl_updated["sku"])
        print(f"TOTAL NUMBER OF SKUs IN USE: {len(sl)}")
        print()
        return sl

    # noinspection PyTypeChecker
    def check_path_db(self, var_file: object) -> object:
        """
        Utility method checking whether target set of exogenous variables data are stored in database or not
        :param None var_file: Placeholder for name of file of variable
        :return: Tuple of test result [0] and path to file if result is positive [1]
        :rtype: tup (str, str)
        """
        assert isinstance(var_file, object), f"{var_file} must be an object referring to a string path!"

        db_path = ""
        error = "NO DATA IN FOLDER"
        good = "OK"
        if var_file is not None:
            db_path = os.path.join(self.database_folder, var_file)
            test = good
        else:
            test = error

        return test, db_path

    ################################## DEMAND METHODS #####################################

    def update_demand(self, auto_dup_fix=True):
        """
       Updates history of orders with new data, saves it in the database folder
       :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
       with new data, are deleted by default or not after being flagged (Default: True)
       """
        print("UPDATING ORDERS DATABASE...")

        hist = pd.read_csv(self.path_orders, sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                           dtype={"sku": str, "client_id": str, "order_quantity": float, "order_units": int})
        try:
            new = pd.read_csv(self.path_new_orders, sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                              dtype={"sku": str, "client_id": str, "order_quantity": float, "order_units": int})

            hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
            hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

        except FileNotFoundError:
            hist_updated = hist
            print("NO NEW DATA FOR ORDERS")

        hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
        hist_updated.to_csv(self.path_orders, sep=";", decimal=",", index=False)
        print()
        print("ORDERS DATA FROM DATABASE IN USE!")
        print()

    def get_demand(self, update=True, selected_vars=None):
        """
        Returns history of orders data (signal) from database
        :param bool update: Determines whether to update or not history with new info available (Default: True)
        :param list selected_vars: List including 2 alternative signals for demand, order quantity and units
        :return: DataFrame with history of orders for each sku and client
        :rtype: pd.DataFrame
        """
        print("EXTRACTING ORDERS DATA...")
        print()
        if update:
            self.update_demand()
        else:
            print("Historical data in use for orders, set update_orders = True to update with new info")
            print()

        # define mandatory keys for demand and other exogenous features
        keys = ["order_date", "sku", "client_id"]

        if selected_vars is not None:
            demand = pd.read_csv(self.path_orders, sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                                 usecols=[*keys, *selected_vars], dtype={"sku": str, "client_id": str})
        else:
            demand = pd.read_csv(self.path_orders, sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                                 dtype={"sku": str, "client_id": str})

        return demand

    ################################## SALES METHODS #####################################

    def update_sales(self, db_check=None, auto_dup_fix=True, save_new=False):
        # TODO: insert list of date columns to parse as init attribute
        # TODO: insert dictionary of columns dtypes as init attribute
        """
            Checks whether sales data stored in database, if yes, updates history of sales with new data, saves it in
            the database. If not checks if new data uploaded by user can be used and saves them eventually. If no data
            available returns None
            :param tup db_check: Result of check_path_db method, [0] determines if data available in database o not
            [1] stores path to file in database if available
            :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
            with new data, are deleted by default or not after being flagged (Default: True)
            :param bool save_new: Determines whether saving new data uploaded if database empty (Default: False)
            :return: True if new uploaded data, False if no data available nor uploaded
            :rtype: bool
            """
        print("CHECKING SALES DATA AVAILABILITY IN DATABASE")
        print()
        if db_check[0] == "OK":
            print("UPDATING SALES DATABASE...")

            hist = pd.read_csv(db_check[1], sep=";", decimal=",",
                               parse_dates=["order_date", "exp_delivery_date", "sale_date"], dayfirst=True,
                               dtype={"sku": str, "sale_sku": str, "actual_sold_quantity": float,
                                      "actual_sold_units": int, "sales_channel": str, "sales_agent": str,
                                      "client_id": str, "client_location": str, "unitary_price": float,
                                      "payment_timing": float})
            try:
                # check if new input data uploaded by user in the new_data_folder
                self.path_new_sales = os.path.join(self.new_data_folder, self.file_new_sales)
                new = pd.read_csv(self.path_new_sales, sep=";", decimal=",",
                                  parse_dates=["order_date", "exp_delivery_date", "sale_date"], dayfirst=True,
                                  dtype={"sku": str, "sale_sku": str, "actual_sold_quantity": float,
                                         "actual_sold_units": int, "sales_channel": str, "sales_agent": str,
                                         "client_id": str, "client_location": str,
                                         "unitary_price": float, "payment_timing": float})

                hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
                hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

            except TypeError:
                hist_updated = hist
                print("NO NEW DATA FOR SALES")

            hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
            hist_updated.to_csv(db_check[1], sep=";", decimal=",", index=False)
            print()
            print("SALES DATA FROM DATABASE IN USE!")
            print()

        else:
            print("NO SALES DATA IN DATABASE, CHECKING IF NEW DATA UPLOADED...")
            print()
            try:
                # check if new input data uploaded by user in the new_data_folder and return True in case
                self.path_new_sales = os.path.join(self.new_data_folder, self.file_new_sales)
                new = pd.read_csv(self.path_new_sales, sep=";", decimal=",",
                                  parse_dates=["order_date", "exp_delivery_date", "sale_date"], dayfirst=True,
                                  dtype={"sku": str, "sale_sku": str, "actual_sold_quantity": float,
                                         "actual_sold_units": int, "sales_channel": str, "sales_agent": str,
                                         "client_id": str, "client_location": str,
                                         "unitary_price": float, "payment_timing": float})

                hist_updated = self.fix_duplicates(new, auto_dup_fix)
                hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
                print("NEW SALES DATA UPLOADED IN USE")
                print()
                if save_new:
                    print("SAVING NEW SALES DATA IN DATABASE...")
                    print()
                    hist_updated.to_csv(os.path.join(self.database_folder, self.file_new_sales), sep=";",
                                        decimal=",", index=False)
                return True

            except TypeError:
                print("NO DATA AVAILABLE FOR SALES")
                print()
                return

    def get_sales(self, active=True, update=True, save_new=False, selected_vars=None):
        # TODO: implement check for format of sale date if selected by user
        """
        Returns history data for set of sales variables (exogenous) according to user's selection.
        :param bool active: Determines whether the set of sales features is used (Default: True)
        :param bool update: Determines whether to update data in database before extraction (Default: True)
        :param bool save_new: Determines whether to save uploaded info in database (Default: False)
        :param list selected_vars: List of selected sales variables from the set (columns names)
        :return: DataFrame with updated data for the set of sales exogenous features selected
        :rtype: pd.DataFrame
        """
        # initiate empty DataFrame, to be returned if set of variables is not used
        sales = pd.DataFrame()

        # define variables used as keys to be always included in the user selection
        keys = ["order_date", "sku", "client_id"]

        # define if whole set of variables is going to be used
        if active:
            # check data availability in database (tuple with str result at [0] and str path at [1] if existing)
            check = self.check_path_db(var_file=self.file_sales)
            if check[0] == "OK":
                # select sales variables to be used as exogenous features
                if selected_vars is not None:
                    # define if updating database before extracting selected set of variables
                    if update:
                        self.update_sales(db_check=check)
                    sales = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                        dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                               "client_id": str})
                    # force sale_date and expected delivery date dtype to datetime
                    for col in sales.columns:
                        if col in ["exp_delivery_date", "sale_date"]:
                            sales[col] = sales[col].astype("datetime64")
                        else:
                            continue

                else:
                    if update:
                        self.update_sales(db_check=check)
                    sales = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                        dayfirst=True, dtype={"sku": str, "client_id": str})
                    # force sale_date and expected delivery date dtype to datetime
                    for col in sales.columns:
                        if col in ["exp_delivery_date", "sale_date"]:
                            sales[col] = sales[col].astype("datetime64")
                        else:
                            continue
            else:
                # check if new data uploaded to be used instead of database, if save_new False use them in memory only
                uploaded_data = self.update_sales(db_check=check, save_new=save_new)
                if uploaded_data:
                    if selected_vars is not None:
                        sales = pd.read_csv(self.path_new_sales, sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                   "client_id": str})
                        # force sale_date and expected delivery date dtype to datetime
                        for col in sales.columns:
                            if col in ["exp_delivery_date", "sale_date"]:
                                sales[col] = sales[col].astype("datetime64")
                            else:
                                continue
                    else:
                        # TODO: store custom selection in attribute to be used in memory just for this case
                        sales = pd.read_csv(self.path_new_sales, sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, dtype={"sku": str, "client_id": str})
                        # force sale_date and expected delivery date dtype to datetime
                        for col in sales.columns:
                            if col in ["exp_delivery_date", "sale_date"]:
                                sales[col] = sales[col].astype("datetime64")
                            else:
                                continue
        return sales

    ################################## EXTERNAL FORECAST METHODS #####################################

    def update_ext_forecast(self, db_check=None, auto_dup_fix=True, save_new=False):
        # TODO: insert list of date columns to parse as init attribute
        # TODO: insert dictionary of columns dtypes as init attribute
        """
        Checks whether external forecast data stored in database, if yes, updates history of external forecast with
        new data, saves it in the database. If not checks if new data uploaded by user can be used and saves them
        eventually. If no data available returns None
        :param tup db_check: Result of check_path_db method, [0] determines if data available in database o not
        [1] stores path to file in database if available
        :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
        with new data, are deleted by default or not after being flagged (Default: True)
        :param bool save_new: Determines whether saving new data uploaded if database empty (Default: False)
        :return: True if new uploaded data, False if no data available nor uploaded
        :rtype: bool, None
        """
        print("CHECKING EXT FORECAST DATA AVAILABILITY IN DATABASE")
        print()
        if db_check[0] == "OK":
            print("UPDATING EXT FORECASTS DATABASE...")
            # TODO: defines columns based on business case (test with dummy)
            hist = pd.read_csv(db_check[1], sep=";", decimal=",",
                               parse_dates=["order_date"], dayfirst=True,
                               dtype={"sku": str, "client_id": str, "order_quantity": float, "order_units": int})
            try:
                # check if new input data uploaded by user in the new_data_folder
                self.path_new_ext_forecast = os.path.join(self.new_data_folder, self.file_new_ext_forecast)
                new = pd.read_csv(self.path_new_sales, sep=";", decimal=",",
                                  parse_dates=["order_date"], dayfirst=True,
                                  dtype={"sku": str, "client_id": str, "order_quantity": float, "order_units": int})

                hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
                hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

            except TypeError:
                hist_updated = hist
                print("NO NEW DATA FOR EXT FORECASTS")

            hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
            hist_updated.to_csv(db_check[1], sep=";", decimal=",", index=False)
            print()
            print("EXT FORECAST DATA FROM DATABASE IN USE!")
            print()

        else:
            print("NO EXT FORECAST DATA IN DATABASE, CHECKING IF NEW DATA UPLOADED...")
            print()
            try:
                # check if new input data uploaded by user in the new_data_folder and return True in case
                self.path_new_ext_forecast = os.path.join(self.new_data_folder, self.file_new_ext_forecast)
                new = pd.read_csv(self.path_new_sales, sep=";", decimal=",",
                                  parse_dates=["order_date"], dayfirst=True,
                                  dtype={"sku": str, "client_id": str, "order_quantity": float, "order_units": int})

                hist_updated = self.fix_duplicates(new, auto_dup_fix)
                hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
                print("NEW EXT FORECAST DATA UPLOADED IN USE")
                print()
                if save_new:
                    print("SAVING NEW EXT FORECAST DATA IN DATABASE...")
                    print()
                    hist_updated.to_csv(os.path.join(self.database_folder, self.file_new_ext_forecast), sep=";",
                                        decimal=",", index=False)
                return True

            except TypeError:
                print("NO DATA AVAILABLE FOR EXT FORECAST")
                print()
                return

    def get_ext_forecast(self, active=True, update=True, save_new=False, selected_vars=None):
        # TODO: implement check for types based on business case
        """
        Returns history data for set of external forecasts variables (exogenous) according to user's selection.
        :param bool active: Determines whether the set of sales features is used (Default: True)
        :param bool update: Determines whether to update data in database before extraction (Default: True)
        :param bool save_new: Determines whether to save uploaded data in database (Default: False)
        :param list selected_vars: List of selected sales variables from the set (columns names)
        :return: DataFrame with updated data for the set of sales exogenous features selected
        :rtype: pd.DataFrame
        """
        # initiate empty DataFrame, to be returned if set of variables is not used
        ext_for = pd.DataFrame()

        # define variables used as keys to be always included in the user selection
        keys = ["order_date", "sku", "client_id"]

        # define if whole set of variables is going to be used
        if active:
            # check data availability in database (tuple with str result at [0] and str path at [1] if existing)
            check = self.check_path_db(var_file=self.file_ext_forecast)
            if check[0] == "OK":
                # check list of variables selected by user for this set
                if selected_vars is not None:
                    # define if updating database before extracting selection of variables
                    if update:
                        self.update_ext_forecast(db_check=check)
                    ext_for = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                          dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                 "client_id": str})
                else:
                    if update:
                        self.update_ext_forecast(db_check=check)
                    ext_for = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                                          dtype={"sku": str, "client_id": str})
            else:
                # check if new data uploaded to be used instead of database, if save_new False use them in memory only
                uploaded_data = self.update_ext_forecast(db_check=check, save_new=save_new)
                if uploaded_data:
                    if selected_vars is not None:
                        ext_for = pd.read_csv(self.path_new_ext_forecast, sep=";", decimal=",",
                                              parse_dates=["order_date"],
                                              dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                     "client_id": str})
                    else:
                        # TODO: store custom selection in attribute to be used in memory just for this case
                        ext_for = pd.read_csv(self.path_new_ext_forecast, sep=";", decimal=",",
                                              parse_dates=["order_date"], dayfirst=True, dtype={"sku": str,
                                                                                                "client_id": str})
        return ext_for

    ################################## STOCK METHODS #####################################

    def update_stock(self, db_check=None, auto_dup_fix=True, save_new=False):
        # TODO: insert list of date columns to parse as init attribute
        # TODO: insert dictionary of columns dtypes as init attribute
        """
        Checks whether stock data stored in database, if yes, updates history of external forecast with
        new data, saves it in the database. If not checks if new data uploaded by user can be used and saves them
        eventually. If no data available returns None
        :param tup db_check: Result of check_path_db method, [0] determines if data available in database o not
        [1] stores path to file in database if available
        :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
        with new data, are deleted by default or not after being flagged (Default: True)
        :param bool save_new: Determines whether saving new data uploaded if database empty (Default: False)
        :return: True if new uploaded data, False if no data available nor uploaded
        :rtype: bool, None
        """
        print("CHECKING STOCK DATA AVAILABILITY IN DATABASE")
        print()
        if db_check[0] == "OK":
            print("UPDATING STOCK DATABASE...")
            # TODO: define columns based on business case (test with dummy)
            hist = pd.read_csv(db_check[1], sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                               dtype={"sku": str, "client_id": str, "order_quantity": float,
                                      "stockout_quantity": float})
            try:
                # check if new input data uploaded by user in the new_data_folder
                self.path_new_stock = os.path.join(self.new_data_folder, self.file_new_stock)
                new = pd.read_csv(self.path_new_stock, sep=";", decimal=",",
                                  parse_dates=["order_date"], dayfirst=True,
                                  dtype={"sku": str, "client_id": str, "order_quantity": float,
                                         "stockout_quantity": float})

                hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
                hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

            except TypeError:
                hist_updated = hist
                print("NO NEW DATA FOR STOCK")

            hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
            hist_updated.to_csv(db_check[1], sep=";", decimal=",", index=False)
            print()
            print("STOCK DATA FROM DATABASE IN USE!")
            print()

        else:
            print("NO STOCK DATA IN DATABASE, CHECKING IF NEW DATA UPLOADED...")
            print()
            try:
                # check if new input data uploaded by user in the new_data_folder and return True in case
                self.path_new_stock = os.path.join(self.new_data_folder, self.file_new_stock)
                new = pd.read_csv(self.path_new_stock, sep=";", decimal=",",
                                  parse_dates=["order_date"], dayfirst=True,
                                  dtype={"sku": str, "client_id": str, "order_quantity": float,
                                         "stockout_quantity": float})

                hist_updated = self.fix_duplicates(new, auto_dup_fix)
                hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
                print("NEW STOCK DATA UPLOADED IN USE")
                print()
                if save_new:
                    print("SAVING NEW STOCK DATA IN DATABASE...")
                    print()
                    hist_updated.to_csv(os.path.join(self.database_folder, self.file_new_stock), sep=";",
                                        decimal=",", index=False)
                return True

            except TypeError:
                print("NO DATA AVAILABLE FOR EXT FORECAST")
                print()
                return

    def get_stock(self, active=True, update=True, save_new=False, selected_vars=None):
        # TODO: implement check for types based on business case
        """
        Returns history data for set of stock variables (exogenous) according to user's selection.
        :param bool active: Determines whether the set of stock features is used (Default: True)
        :param bool update: Determines whether to update data in database before extraction (Default: True)
        :param bool save_new: Determines whether to save uploaded data in database (Default: False)
        :param list selected_vars: List of selected sales variables from the set (columns names)
        :return: DataFrame with updated data for the set of sales exogenous features selected
        :rtype: pd.DataFrame
        """
        # initiate empty DataFrame, to be returned if set of variables is not used
        stock = pd.DataFrame()

        # define variables used as keys to be always included in the user selection
        keys = ["order_date", "sku", "client_id"]

        # define if whole set of variables is going to be used
        if active:
            # check data availability in database (tuple with str result at [0] and str path at [1] if existing)
            check = self.check_path_db(var_file=self.file_stock)
            if check[0] == "OK":
                # check list of variables selected by user for this set
                if selected_vars is not None:
                    # define if updating database before extracting selection of variables
                    if update:
                        self.update_stock(db_check=check)
                    stock = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                        dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                               "client_id": str})
                else:
                    if update:
                        self.update_stock(db_check=check)
                    stock = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"], dayfirst=True,
                                        dtype={"sku": str, "client_id": str})
            else:
                # check if new data uploaded to be used instead of database, if save_new False use them in memory only
                uploaded_data = self.update_stock(db_check=check, save_new=save_new)
                if uploaded_data:
                    if selected_vars is not None:
                        stock = pd.read_csv(self.path_new_stock, sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                   "client_id": str})
                    else:
                        # TODO: store custom selection in attribute to be used in memory just for this case
                        stock = pd.read_csv(self.path_new_stock, sep=";", decimal=",",
                                            parse_dates=["order_date"], dayfirst=True, dtype={"sku": str,
                                                                                              "client_id": str})
        return stock

    ################################## MARKETING METHODS #####################################

    def update_marketing(self, db_check=None, auto_dup_fix=True, save_new=False):
        # TODO: insert list of date columns to parse as init attribute
        # TODO: insert dictionary of columns dtypes as init attribute
        """
        Checks whether marketing data stored in database, if yes, updates history of marketing with
        new data, saves it in the database. If not checks if new data uploaded by user can be used and saves them
        eventually.
        :param tup db_check: Result of check_path_db method, [0] determines if data available in database o not
        [1] stores path to file in database if available
        :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
        with new data, are deleted by default or not after being flagged (Default: True)
        :param bool save_new: Determines whether saving new data uploaded if database empty (Default: False)
        :return: True if new uploaded data, False if no data available nor uploaded
        :rtype: bool, None
        """
        print("CHECKING MARKETING DATA AVAILABILITY IN DATABASE")
        print()
        if db_check[0] == "OK":
            print("UPDATING MARKETING DATABASE...")
            # TODO: defines columns based on business case (test with dummy)
            hist = pd.read_csv(db_check[1], sep=";", decimal=",",
                               parse_dates=["order_date", "promotion_date_start", "promotion_date_end"],
                               dayfirst=True,
                               dtype={"sku": str, "market_brand": str, "product_family": str,
                                      "promotion_type": str, "promotion_amount": float,
                                      "promotion_target": str})
            try:
                # check if new input data uploaded by user in the new_data_folder
                self.path_new_marketing = os.path.join(self.new_data_folder, self.file_new_marketing)
                new = pd.read_csv(self.path_new_marketing, sep=";", decimal=",",
                                  parse_dates=["order_date", "promotion_date_start", "promotion_date_end"],
                                  dayfirst=True,
                                  dtype={"sku": str, "market_brand": str, "product_family": str,
                                         "promotion_type": str, "promotion_amount": float,
                                         "promotion_target": str})

                hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
                hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

            except TypeError:
                hist_updated = hist
                print("NO NEW DATA FOR MARKETING")

            hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
            hist_updated.to_csv(db_check[1], sep=";", decimal=",", index=False)
            print()
            print("MARKETING DATA FROM DATABASE IN USE!")
            print()

        else:
            print("NO MARKETING DATA IN DATABASE, CHECKING IF NEW DATA UPLOADED...")
            print()
            try:
                # check if new input data uploaded by user in the new_data_folder and return True in case
                self.path_new_marketing = os.path.join(self.new_data_folder, self.file_new_marketing)
                new = pd.read_csv(self.path_new_marketing, sep=";", decimal=",",
                                  parse_dates=["order_date", "promotion_date_start", "promotion_date_end"],
                                  dayfirst=True,
                                  dtype={"sku": str, "market_brand": str, "product_family": str,
                                         "promotion_type": str, "promotion_amount": float,
                                         "promotion_target": str})

                hist_updated = self.fix_duplicates(new, auto_dup_fix)
                hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
                print("NEW MARKETING DATA UPLOADED IN USE")
                print()
                if save_new:
                    print("SAVING NEW STOCK DATA IN DATABASE...")
                    print()
                    hist_updated.to_csv(os.path.join(self.database_folder, self.file_new_marketing), sep=";",
                                        decimal=",", index=False)
                return True

            except TypeError:
                print("NO DATA AVAILABLE FOR MARKETING")
                print()
                return

    def get_marketing(self, active=True, update=True, save_new=False, selected_vars=None):
        # TODO: implement check for types based on business case
        """
        Returns history data for set of marketing variables (exogenous) according to user's selection.
        :param bool active: Determines whether the set of marketing features is used (Default: True)
        :param bool update: Determines whether to update data in database before extraction (Default: True)
        :param bool save_new: Determines whether to save uploaded data in database (Default: False)
        :param list selected_vars: List of selected sales variables from the set (columns names)
        :return: DataFrame with updated data for the set of sales exogenous features selected
        :rtype: pd.DataFrame
        """
        # initiate empty DataFrame, to be returned if set of variables is not used
        marketing = pd.DataFrame()

        # define variables used as keys to be always included in the user selection
        keys = ["order_date", "sku"]

        # define if whole set of variables is going to be used
        if active:
            # check data availability in database (tuple with str result at [0] and str path at [1] if existing)
            check = self.check_path_db(var_file=self.file_marketing)
            if check[0] == "OK":
                # check list of variables selected by user for this set
                if selected_vars is not None:
                    # define if updating database before extracting selection of variables
                    if update:
                        self.update_marketing(db_check=check)
                    marketing = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                   "client_id": str})
                    # force dtype of promotion dates to datetime if provided
                    for col in marketing.columns:
                        if col in ["promotion_date_start", "promotion_date_end"]:
                            marketing[col] = marketing[col].astype("datetime64")
                        else:
                            continue
                else:
                    if update:
                        self.update_marketing(db_check=check)
                    marketing = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, dtype={"sku": str, "client_id": str})
                    # force dtype of promotion dates to datetime if provided
                    for col in marketing.columns:
                        if col in ["promotion_date_start", "promotion_date_end"]:
                            marketing[col] = marketing[col].astype("datetime64")
                        else:
                            continue
            else:
                # check if new data uploaded to be used instead of database, if save_new False use them in memory only
                uploaded_data = self.update_marketing(db_check=check, save_new=save_new)
                if uploaded_data:
                    if selected_vars is not None:
                        marketing = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                                dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                       "client_id": str})
                        # force dtype of promotion dates to datetime if provided
                        for col in marketing.columns:
                            if col in ["promotion_date_start", "promotion_date_end"]:
                                marketing[col] = marketing[col].astype("datetime64")
                            else:
                                continue
                    else:
                        # TODO: store custom selection in attribute to be used in memory just for this case
                        marketing = pd.read_csv(self.path_new_marketing, sep=";", decimal=",",
                                                parse_dates=["order_date"], dayfirst=True, dtype={"sku": str,
                                                                                                  "client_id": str})
                        # force dtype of promotion dates to datetime if provided
                        for col in marketing.columns:
                            if col in ["promotion_date_start", "promotion_date_end"]:
                                marketing[col] = marketing[col].astype("datetime64")
                            else:
                                continue
        return marketing

    ################################## MACROECONOMICS METHODS #####################################

    def update_macro(self, db_check=None, auto_dup_fix=True, save_new=False):
        # TODO: insert list of date columns to parse as init attribute
        # TODO: insert dictionary of columns dtypes as init attribute
        """
        Checks whether macroeconomic data stored in database, if yes, updates history of macroeconomics data with
        new data, saves it in the database. If not checks if new data uploaded by user can be used and saves them
        eventually.
        :param tup db_check: Result of check_path_db method, [0] determines if data available in database o not
        [1] stores path to file in database if available
        :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
        with new data, are deleted by default or not after being flagged (Default: True)
        :param bool save_new: Determines whether saving new data uploaded if database empty (Default: False)
        :return: True if new uploaded data, False if no data available nor uploaded
        :rtype: bool, None
        """
        print("CHECKING MACROECONOMICS DATA AVAILABILITY IN DATABASE")
        print()
        if db_check[0] == "OK":
            print("UPDATING MACROECONOMICS DATABASE...")
            # TODO: defines columns based on business case (test with dummy)
            hist = pd.read_csv(db_check[1], sep=";", decimal=",", parse_dates=["order_date"],
                               dayfirst=True, dtype={"sku": str, "client_id": str})
            try:
                # check if new input data uploaded by user in the new_data_folder
                self.path_new_macro = os.path.join(self.new_data_folder, self.file_new_macro)
                new = pd.read_csv(self.path_new_macro, sep=";", decimal=",", parse_dates=["order_date"],
                                  dayfirst=True, dtype={"sku": str, "client_id": str})

                hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
                hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

            except TypeError:
                hist_updated = hist
                print("NO NEW DATA FOR MACROECONOMICS")

            hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
            hist_updated.to_csv(db_check[1], sep=";", decimal=",", index=False)
            print()
            print("MACROECONOMICS DATA FROM DATABASE IN USE!")
            print()

        else:
            print("NO MACROECONOMICS DATA IN DATABASE, CHECKING IF NEW DATA UPLOADED...")
            print()
            try:
                # check if new input data uploaded by user in the new_data_folder and return True in case
                self.path_new_macro = os.path.join(self.new_data_folder, self.file_new_macro)
                new = pd.read_csv(self.path_new_macro, sep=";", decimal=",", parse_dates=["order_date"],
                                  dayfirst=True, dtype={"sku": str, "client_id": str})

                hist_updated = self.fix_duplicates(new, auto_dup_fix)
                hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
                print("NEW MACROECONOMICS DATA UPLOADED IN USE")
                print()
                if save_new:
                    print("SAVING NEW MACROECONOMICS DATA IN DATABASE...")
                    print()
                    hist_updated.to_csv(os.path.join(self.database_folder, self.file_new_macro), sep=";",
                                        decimal=",", index=False)
                return True

            except TypeError:
                print("NO DATA AVAILABLE FOR MACROECONOMICS")
                print()
                return

    def get_macro(self, ext=False, active=True, update=True, save_new=False, selected_vars=None):
        # TODO: implement check for types based on business case
        """
        Defines if collecting macroeconomics data from external or internal source.
        Returns history data for set of marketing variables (exogenous) according to user's selection.
        :param: bool ext: Determines whether to use external APIs instead of stored and/or uploded data
        :param bool active: Determines whether the set of marketing features is used (Default: True)
        :param bool update: Determines whether to update data in database before extraction (Default: True)
        :param bool save_new: Determines whether to save uploaded data in database (Default: False)
        :param list selected_vars: List of selected sales variables from the set (columns names)
        :return: DataFrame with updated data for the set of sales exogenous features selected
        :rtype: pd.DataFrame
        """
        # define whether to call external APIs to collect data already updated
        if ext:
            print("SETTING API REQUEST FOR MACROECONOMICS DATA....")
            print()
            # TODO: insert method(s) calling selected APIs here
            pass
        else:
            # initiate empty DataFrame, to be returned if set of variables is not used
            macro = pd.DataFrame()

            # define variables used as keys to be always included in the user selection
            keys = ["order_date", "sku", "client_id"]

            # define if whole set of variables is going to be used
            if active:
                # check data availability in database (tuple with str result at [0] and str path at [1] if existing)
                check = self.check_path_db(var_file=self.file_macro)
                if check[0] == "OK":
                    # check list of variables selected by user for this set
                    if selected_vars is not None:
                        # define if updating database before extracting selection of variables
                        if update:
                            self.update_macro(db_check=check)
                        macro = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                   "client_id": str})
                    else:
                        if update:
                            self.update_macro(db_check=check)
                        macro = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, dtype={"sku": str, "client_id": str})
                else:
                    # check if new data uploaded to be used instead of database, if save_new False use them in memory only
                    uploaded_data = self.update_macro(db_check=check, save_new=save_new)
                    if uploaded_data:
                        if selected_vars is not None:
                            macro = pd.read_csv(self.path_new_macro, sep=";", decimal=",", parse_dates=["order_date"],
                                                dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                       "client_id": str})
                        else:
                            # TODO: store custom selection in attribute to be used in memory just for this case
                            macro = pd.read_csv(self.path_new_macro, sep=";", decimal=",",
                                                parse_dates=["order_date"], dayfirst=True, dtype={"sku": str,
                                                                                                  "client_id": str})
            return macro

    ################################## OTHER EXOGENOUS VARIABLES METHODS #####################################

    def update_other(self, db_check=None, auto_dup_fix=True, save_new=False):
        # TODO: insert list of date columns to parse as init attribute
        # TODO: insert dictionary of columns dtypes as init attribute
        """
        Checks if other exogenous variables data stored in database, if yes, updates history of other exogenous
        variables with new data, saves it in the database. If not checks if new data uploaded by user can be used and
        saves them eventually.
        :param tup db_check: Result of check_path_db method, [0] determines if data available in database o not
        [1] stores path to file in database if available
        :param bool auto_dup_fix: Determines whether eventual duplicated entries resulting from update of history
        with new data, are deleted by default or not after being flagged (Default: True)
        :param bool save_new: Determines whether saving new data uploaded if database empty (Default: False)
        :return: True if new uploaded data, False if no data available nor uploaded
        :rtype: bool, None
        """
        print("CHECKING OTHER EXOGENOUS VARIABLES DATA AVAILABILITY IN DATABASE")
        print()
        if db_check[0] == "OK":
            print("UPDATING OTHER EXOGENOUS VARIABLES DATABASE...")
            # TODO: defines columns based on business case (test with dummy)
            hist = pd.read_csv(db_check[1], sep=";", decimal=",", parse_dates=["order_date"],
                               dayfirst=True, dtype={"sku": str, "client_id": str})
            try:
                # check if new input data uploaded by user in the new_data_folder
                self.path_new_other = os.path.join(self.new_data_folder, self.file_new_other)
                new = pd.read_csv(self.path_new_other, sep=";", decimal=",", parse_dates=["order_date"],
                                  dayfirst=True, dtype={"sku": str, "client_id": str})

                hist_updated = pd.concat([hist, new], axis=0, sort=False, ignore_index=True)
                hist_updated = self.fix_duplicates(hist_updated, auto_dup_fix)

            except TypeError:
                hist_updated = hist
                print("NO NEW DATA FOR OTHER EXOGENOUS VARIABLES")

            hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
            hist_updated.to_csv(db_check[1], sep=";", decimal=",", index=False)
            print()
            print("OTHER EXOGENOUS VARIABLES DATA FROM DATABASE IN USE!")
            print()

        else:
            print("NO OTHER EXOGENOUS VARIABLES DATA IN DATABASE, CHECKING IF NEW DATA UPLOADED...")
            print()
            try:
                # check if new input data uploaded by user in the new_data_folder and return True in case
                self.path_new_other = os.path.join(self.new_data_folder, self.file_new_other)
                new = pd.read_csv(self.path_new_other, sep=";", decimal=",", parse_dates=["order_date"],
                                  dayfirst=True, dtype={"sku": str, "client_id": str})

                hist_updated = self.fix_duplicates(new, auto_dup_fix)
                hist_updated.sort_values(by="order_date", axis=0, ascending=True, inplace=True, ignore_index=True)
                print("NEW OTHER EXOGENOUS VARIABLES DATA UPLOADED IN USE")
                print()
                if save_new:
                    print("SAVING NEW OTHER EXOGENOUS VARIABLES DATA IN DATABASE...")
                    print()
                    hist_updated.to_csv(os.path.join(self.database_folder, self.file_new_other), sep=";",
                                        decimal=",", index=False)
                return True

            except TypeError:
                print("NO DATA AVAILABLE FOR OTHER EXOGENOUS VARIABLES")
                print()
                return

    def get_other(self, ext=False, active=True, update=True, save_new=False, selected_vars=None):
        # TODO: implement check for types based on business case
        """
        Defines if collecting other exogenous variables data from external or internal source.
        Returns history data for set of marketing variables (exogenous) according to user's selection.
        :param: bool ext: Determines whether to use external APIs instead of stored and/or uploded data
        :param bool active: Determines whether the set of marketing features is used (Default: True)
        :param bool update: Determines whether to update data in database before extraction (Default: True)
        :param bool save_new: Determines whether to save uploaded data in database (Default: False)
        :param list selected_vars: List of selected sales variables from the set (columns names)
        :return: DataFrame with updated data for the set of sales exogenous features selected
        :rtype: pd.DataFrame
        """
        # define whether to call external APIs to collect data already updated
        if ext:
            print("SETTING API REQUEST FOR MACROECONOMICS DATA....")
            print()
            # TODO: insert method(s) calling selected APIs here
            pass
        else:
            # initiate empty DataFrame, to be returned if set of variables is not used
            other = pd.DataFrame()

            # define variables used as keys to be always included in the user selection
            keys = ["order_date", "sku", "client_id"]

            # define if whole set of variables is going to be used
            if active:
                # check data availability in database (tuple with str result at [0] and str path at [1] if existing)
                check = self.check_path_db(var_file=self.file_other)
                if check[0] == "OK":
                    # check list of variables selected by user for this set
                    if selected_vars is not None:
                        # define if updating database before extracting selection of variables
                        if update:
                            self.update_other(db_check=check)
                        other = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                   "client_id": str})
                    else:
                        if update:
                            self.update_other(db_check=check)
                        other = pd.read_csv(check[1], sep=";", decimal=",", parse_dates=["order_date"],
                                            dayfirst=True, dtype={"sku": str, "client_id": str})
                else:
                    # check if new data uploaded to be used instead of database, else use them in memory only
                    uploaded_data = self.update_other(db_check=check, save_new=save_new)
                    if uploaded_data:
                        if selected_vars is not None:
                            other = pd.read_csv(self.path_new_other, sep=";", decimal=",", parse_dates=["order_date"],
                                                dayfirst=True, usecols=[*keys, *selected_vars], dtype={"sku": str,
                                                                                                       "client_id": str})
                        else:
                            # TODO: store custom selection in attribute to be used in memory just for this case
                            other = pd.read_csv(self.path_new_other, sep=";", decimal=",",
                                                parse_dates=["order_date"], dayfirst=True, dtype={"sku": str,
                                                                                                  "client_id": str})
            return other
