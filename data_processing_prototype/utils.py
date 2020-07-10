import pandas as pd
from typing import Union




# TODO: fix to use eventually
# def get_column_distribution(df: pd.DataFrame, col: str, grouper_col: Union[str, list], normalize: bool = True) -> \
#         pd.DataFrame:
#     """ Some Claims can have several Transactions in the same OperationDate. As such we keep a distribution of values
#     for some columns.
#     :param df: DataFrame containing the col and grouper_col passed
#     :param col: Column to calculate the distributions for
#     :param grouper_col: Column(s) to be used to aggregate when calculating the distribution
#     :return: DataFrame with one column for each category existing on col with corresponding distribution.
#     """
#     # Count each category and get the cumulative over time
#     # After normalize the counts for each snapshot (dividing by the total)
#     temp = df.groupby(grouper_col)[col].value_counts().unstack(-1, fill_value=0).groupby(grouper_col[0]).cumsum()
#     if normalize:
#         temp = temp.div(temp.sum(axis=1), axis=0)
#     temp.columns = str(col) + '_' + temp.columns
#     return temp
