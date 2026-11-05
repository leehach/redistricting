"""Reshape pandas tables into the data that the DEA Pyomo models read."""


def make_dict(df):
    """Turn a DataFrame into a dict that maps (row, col) to its value.

    Both numbers start at 1, to match the model's RangeSets:

        0.40,0.70        {(1, 1): 0.40, (1, 2): 0.70,   # plan 1
        0.50,0.60   ->    (2, 1): 0.50, (2, 2): 0.60,   # plan 2
        0.45,0.80         (3, 1): 0.45, (3, 2): 0.80}   # plan 3

    Pyomo can read some data straight from a DataFrame, so we may not
    need this.
    """
    # a plain NumPy array - reading one cell is much faster than df.iloc
    values = df.to_numpy()
    (M, N) = values.shape  # rows, columns

    return {
        (row + 1, col + 1): values[row, col]
        for row in range(M)
        for col in range(N)
    }
