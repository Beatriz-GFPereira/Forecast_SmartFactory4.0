# General Assumptions:

#### - Data history available for last 7 years

#### - Orders data are registered with WEEKLY FREQUENCY, ON MONDAY.

#### - New orders data are uploaded in blocks of 4 weeks, every end of month.

#### -  Company produces 12 different products (SKU)

#### -  Company has 18 different clients (client_id)

#### - N.B. No seasonlity and/or NO trend means data are purely randomly generated


# DEMAND SIGNAL DATA
##################################################################################

"""
#### In terms of Products, company produces:

- 5 products with high value and low quantity per pallet:
    - HV1: monthly seasonality, + Trend; 25 units/pallet
    - HV2: spring seasonality, - Trend; 25 units/pallet
    - HV3: autumn seasonality AND mid month seasonality, NO Trend; 60 units/pallet
    - HV4: NO seasonality, + Trend; 80 units/pallet
    - HV5, winter seaonality, NO Trend; 50 units/pallet
- 4 products with medium-high value and medium-low quantity per pallet:
    - MV1: summer seasonality AND end of month seasonality, - Trend; 250 units/pallet
    - MV2: mid of month seasonality, + Trend; 300 units/pallet
    - MV3: beginning of week seasonality, NO Trend; 300 units/pallet
    - MV4: winter seasonality, - Trend; 500 units/pallet
- 3 products with low value and high quantity per pallet:
    - LV1: autumn seasonality AND end of week seasonality, + Trend; 1200 units/pallet
    - LV2: summer seasonality, + Trend; 2000 units/pallet
    - LV3: NO seasonality, - Trend; 1550 units/pallet

#### In terms of Clients, company sells to:
C1 up to C18 (client_id)

"""

import pandas as pd
import random as rd
import matplotlib.pyplot as plt

# Start day of orders history
start_date = pd.Timestamp(year=2013, month=7, day=14)

end_date = start_date + pd.offsets.DateOffset(years=7)

# Freq of data
F = "W"

# Products SKU
prods = ["HV1", "HV2", "HV3", "HV4", "HV5", "MV1", "MV2", "MV3", "MV4", "LV1", "LV2", "LV3"]

# Client IDs (18)
c_id = [f"C{n}" for n in range(1, 19)]

# Quantity per unit (pallet)
pallet = [25, 25, 60, 80, 50, 250, 300, 300, 500, 1200, 2000, 1550]
q_ratio = {k: v for k, v in zip(prods, pallet)}

# Number of orders in 1 period (if 0 priod is skipped) MAX == all clients make an order
ords = (0, 12)

# Yearly Seasonality params: N neg seas, beg: starting month, end: ending month in: seas factor, out: non-seas, A: seas active
y_seas_param = [{"N": rd.choice([1, -1]), "beg": 1, "end": 6, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 0},
                {"N": rd.choice([1, -1]), "beg": 4, "end": 6, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 10, "end": 12, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 1, "end": 6, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 0},
                {"N": rd.choice([1, -1]), "beg": 1, "end": 3, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 7, "end": 9, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 1, "end": 6, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 0},
                {"N": rd.choice([1, -1]), "beg": 1, "end": 6, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 0},
                {"N": rd.choice([1, -1]), "beg": 1, "end": 3, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 10, "end": 12, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 7, "end": 9, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 1},
                {"N": rd.choice([1, -1]), "beg": 1, "end": 6, "s": rd.choice([0.7, 0.8, 0.9, 1.1, 1.2]),
                 "out": rd.choice([1.2, 1.3, 1.4, 1.5, 1.6]), "A": 0}]

y_seas = {k: v for k, v in zip(prods, y_seas_param)}

# Noise component of order_units
noise_lim = [(1, 10), (1, 20), (1, 20), (1, 25), (1, 30), (1, 25), (1, 30), (1, 25), (1, 20), (1, 25), (1, 30), (1, 25)]
ns = {k: v for k, v in zip(prods, noise_lim)}

# Trent equation params : N: negative trend, m: slopes to choose, exp: exponents to choose, A trend active
trend_parameters = [{"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": -1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 0},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 0},
                    {"N": -1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 0},
                    {"N": -1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": 1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1},
                    {"N": -1, "m": rd.choice([100, 200, 300]), "exp": rd.choice([0.2, 0.3, 0.4, 0.5]), "A": 1}]
trd = {k: v for k, v in zip(prods, trend_parameters)}

# list of df to merge
P = list()

for sku in prods:

    # Initialize empty df, with required column names
    df = pd.DataFrame(data=None, columns=["order_date", "sku", "client_id", "order_quantity", "order_units",
                                          "sesonality", "noise", "trend", "week", "month", "year", "time"])

    # Create order_dates array with defined frequency
    history = pd.date_range(start=start_date, end=end_date, freq=F)

    # Initialize lists to collect random values for each column
    orders = list()
    clients = list()
    noise_value = list()
    counter = 0
    time = list()

    # Replicate each date a random number of times within range limits and randomly assign to each a product and a client
    for date in history:
        # Replicate date random number of times
        n_orders = rd.randint(ords[0], ords[1])
        counter += 1

        # If date is not skipped assign ranomly a prod sku and a client_id
        if n_orders > 0:
            for i in range(n_orders):
                clients += (rd.sample(c_id, 1))
                noise_value.append(rd.randint(ns[sku][0], ns[sku][1]))
                time.append(counter)

        # Concatenate all replicated dates
        orders += ([date] * n_orders)

    # Set results as columns of inital df
    df["order_date"] = orders
    df["sku"] = sku
    df["client_id"] = clients
    df["week"] = df["order_date"].dt.week
    df["month"] = df["order_date"].dt.month
    df["year"] = df["order_date"].dt.year
    df["time"] = time

    # Yearly seasonalites (winter, spring, summer, autumn)

    # TODO: add other seasonalities
    df.loc[(df["month"] >= y_seas[sku]["beg"]) & (df["month"] <= y_seas[sku]["end"]), "y_seas"] = \
        df["week"].apply(lambda x: y_seas[sku]["N"] * x * y_seas[sku]["s"] if y_seas[sku]["A"] == 1 else x)

    #     df.loc[(df["month"] >= y_seas[sku]["beg"]) & (df["month"] <= y_seas[sku]["end"]), "y_seas"] = \
    #     df["week"].apply(lambda x: (x**1.2 - x*2 + 2)/10*2 if y_seas[sku]["A"] == 1 else x)

    df.loc[(df["month"] < y_seas[sku]["beg"]) | (df["month"] > y_seas[sku]["end"]), "y_seas"] = \
        df["week"].apply(lambda x: y_seas[sku]["N"] * x * y_seas[sku]["out"] if y_seas[sku]["A"] == 1 else x)

    #     df.loc[(df["month"] < y_seas[sku]["beg"]) | (df["month"] > y_seas[sku]["end"]), "y_seas"] = \
    #     df["week"].apply(lambda x: -(x**1.2 - x*2 + 2)/10*2 if y_seas[sku]["A"] == 1 else x)

    # Noise
    df["noise"] = noise_value

    # Trend
    df["trend"] = df["time"].apply(
        lambda x: trd[sku]["A"] * ((trd[sku]["N"] * trd[sku]["m"] * x ** trd[sku]["exp"]) / (10 ** 1)) + 700)

    df["order_units"] = round(df["y_seas"] + df["noise"] + df["trend"])
    df["order_quantity"] = df["order_units"] * q_ratio[sku]

    # Append each df to merge them
    P.append(df[["order_date", "sku", "client_id", "order_quantity", "order_units"]])

    plt.plot(df["order_units"])
    plt.title(sku)
    plt.show()

# Merge all df in one and save it as CSV
final = pd.concat(P, axis=0).sort_values(by="order_date").reset_index(drop=True)

final.iloc[:round((final.shape[0] * 0.9))].to_csv("../dummy_file_db/Orders_mandatory.csv",
                                                  sep=";", decimal=",", index=False)

final.iloc[-round((final.shape[0] * 0.1)):].to_csv("../dummy_user_uploads/New_Orders_mandatory.csv", sep=";",
                                                   decimal=",", index=False)

# TODO: add generators for exog variables