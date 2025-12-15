import pandas as pd
import map_data as md
import re

def rename_columns(df, col_part, replacement) -> pd.DataFrame:
    '''
    Replacing column name parts

    Input:
    - df: DataFrame
    - col_part: column name part to be replaced
    - replacement: string which replaces column name part

    Returns:
    - updated DataFrame
    '''
    df.rename(columns=lambda x: x.replace(col_part, replacement), inplace=True)
    return df

def map_cols(df, map_dict, columns=None) -> pd.DataFrame:
    '''
    Map column values (for chocsen columns) to standard values based on a mapping dictionary.
    
    Input:
    df: DataFrame with columns to be mapped
    map_dict: dictionary with column values to be replaced and standard replacement values
    columns: list of column(s) which should be mapped
    
    Returns:
    - updated DataFrame with mapped column values
    '''
    if columns is None:
        map_columns = df.columns
    elif isinstance(columns, list) and all(isinstance(col, str) for col in columns):
        map_columns = [c for c in columns if c in df.columns]
    else:
        raise ValueError("Leave chosen columns empty or chose a list of colum names (list of strings).")
    
    for col in map_columns:
        df[col] = df[col].replace(map_dict)
        
    return df

def to_num_col(df, num_columns=None) -> pd.DataFrame:
    '''
    Converts all given columns to numeric ones
    
    df: DataFrame in which columns are converted
    num_columns: list of column names (string list) which are to be numeric
    
    Returns:
    - updated DataFrame with numeric columns
    '''
    if num_columns is None:
        num_columns = df.columns
    elif isinstance(num_columns, list) and all(isinstance(col, str) for col in num_columns):
        num_columns = [c for c in num_columns if c in df.columns]
    else:
        raise ValueError("Leave chosen columns empty or chose a list of colum names (list of strings).")
    
    for col in num_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

# 2. Transform likert-scales to numerical values

# Only select named columns and map correct likert number in case for string reply
def transform_likert(df, ):
    valid_columns = [c for c in md.VALID_COLUMNS if c in df.columns]

    for c in valid_columns:
        s = df[c].replace(md.LIKERT_MAP)
        df[c] = s.astype("string").str.extract(r"(-?\d+\.?\d*)", expand=False)
    to_num_col(df, valid_columns)
    return df

# remove speeding and very slow respondents
# Umbenennen von Duration(inseconds) -> duration (nach dem Spalten-Cleaning)
def rm_speeders(df):
    if "Duration(inseconds)" in df.columns:
        df = rename_columns(df, "Duration(inseconds)", "duration")

        # Sicherstellen, dass duration numerisch ist
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

        p5 = df["duration"].quantile(0.05)
        p95 = df["duration"].quantile(0.95)

        speedy_slowy = (df["duration"] > p5) & (df["duration"] < p95)
        print('min ', min(df['duration']), ' max ', max(df['duration']))
        
        print(f'{len(df)-len(df[speedy_slowy])} respondents were faster than ({p5})s or slower than ({p95})s')

    else:
        print("Duration filter already done")
    return ~speedy_slowy

# remove straightliners
# (respondents who gave the same answer across a set of Likert-scale questions)
def rm_straightliners(df):
    # check 
    likert_costs_cols = [c for c in df.columns if c.startswith("likert_costs_")]
    identity_cols = [c for c in df.columns if c.startswith("identity_group")]
    gal_tan_cols = [c for c in df.columns if c.startswith("gal_tan_")]
    deservingness_cols = [c for c in df.columns if c.startswith("deservingness")]

    straightliners_costs = df[likert_costs_cols].nunique(axis=1, dropna=True).eq(1)
    straightliners_identity = df[identity_cols].nunique(axis=1, dropna=True).eq(1)
    straightliners_gal_tan = df[gal_tan_cols].nunique(axis=1, dropna=True).eq(1)
    straightliners_deservingness = df[deservingness_cols].nunique(axis=1, dropna=True).eq(1)

    straightliners = straightliners_costs | straightliners_identity | straightliners_gal_tan | straightliners_deservingness
    return straightliners


# filter identical IP addresses
# IPs mit mehr als 2 verschiedenen ids
def filter_double_ipa(df):
    shared_ips = (
        df.groupby("IPAddress")["id"]
        .nunique()
        .reset_index(name="user_count")
    )
    shared_ips = shared_ips[shared_ips["user_count"] > 2]["IPAddress"]

    # Nutzer mit diesen IPs
    users_to_remove = df[df["IPAddress"].isin(shared_ips)].copy()

    # anti_join nach id & IPAddress
    df = df.merge(
        users_to_remove[["id", "IPAddress"]].drop_duplicates(),
        on=["id", "IPAddress"],
        how="left",
        indicator=True,
    )
    df = df[df["_merge"] == "left_only"].drop(columns="_merge").copy()
    return df

def string_mapping(df, mapping_dict, column_patterns=None, numeric=False):
    df = df.copy()

    if column_patterns:
        regex = re.compile("|".join(column_patterns))
        cols = [c for c in df.columns if regex.search(c)]
    else:
        cols = df.columns.tolist()

    # df[cols] = df[cols].astype("string").apply(lambda s: s.str.strip())
    df[cols] = df[cols].replace(mapping_dict)
    if numeric:
        df = to_num_col(df, cols)
        
    return df