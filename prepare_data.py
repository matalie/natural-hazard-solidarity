# ------------------------------------------------------------------
# Load Libraries & Data
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import re
import importlib
import f_data_processing as dp
import map_data as md
dp = importlib.reload(dp)
print(os.getcwd())


# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def rm_bad_quality(df):
    # Filter: incompletes, screened out, poor quality, quota met
    df = df[(df["DistributionChannel"] != "preview") & (df["Finished"] != False)].copy()
    df = df[~df["Q_TerminateFlag"].isin(["PoorQuality", "NA", "QuotaMet", "Screened"])].copy()
    return df

# 2. Transform likert-scales to numerical values

# Only select named columns and map correct likert number in case for string reply
def transform_likert(df):
    valid_columns = [c for c in md.VALID_COLUMNS if c in df.columns]

    for c in valid_columns:
        s = df[c].replace(md.LIKERT_MAP)
        df[c] = s.astype("string").str.extract(r"(-?\d+\.?\d*)", expand=False)
    dp.to_num_col(df, valid_columns)
    return df

# remove speeding and very slow respondents
# Umbenennen von Duration(inseconds) -> duration (nach dem Spalten-Cleaning)
def rm_speeders(df):
    if "Duration(inseconds)" in df.columns:
        df = dp.rename_columns(df, "Duration(inseconds)", "duration")

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

def demographics_mapping(df, mapping_dict, column_patterns=None):
    df = df.copy()

    if column_patterns:
        regex = re.compile("|".join(column_patterns))
        cols = [c for c in df.columns if regex.search(c)]
    else:
        cols = df.columns.tolist()

    df[cols] = df[cols].astype("string").apply(lambda s: s.str.strip())
    df[cols] = df[cols].replace(mapping_dict)

    return df


# ------------------------------------------------------------------
# main: prepare data sets
# ------------------------------------------------------------------
# surveys
survey_dict = {"S1": "./data/Survey_Adaptation Natural Hazards_First Wave_raw data.csv", 
               "S2": "./data/Survey_Adaptation Natural Hazards_Second Wave_raw data.csv"}
# survey_dict = {"Survey 2": "../data/Survey_Adaptation Natural Hazards_Second Wave_raw data.csv"}
# survey_dict = {"Survey 1": "../data/Survey_Adaptation Natural Hazards_First Wave_raw data.csv"}

df_dict ={}

for s, file in survey_dict.items():
    df = pd.read_csv(file, dtype={"id": "string"}, skiprows=[1,2])
    # remove lines, replace empty cells, rename columnames
    df = df.replace("", pd.NA)
    df.columns = df.columns.str.replace(".", "", regex=False)
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)
    df = dp.rename_columns(df, 'municipality', 'benefits')
    df = dp.to_num_col(df, ['id', 'm'])

    # filter only valid response rows
    print(len(df))
    df = rm_bad_quality(df)
    print(len(df))
    
    # TODO rm speeders, straightliners, inattentives
    speedy_slowy = rm_speeders(df)
    straightliners = rm_straightliners(df)
    inattentives = df[f'attention_check'] != 'Agree'
    # print(speedy_slowy)
    # print(straightliners)
    # print(inattentives)
    df_filtered = df[~((speedy_slowy | straightliners | inattentives))]
    
    print(len(df_filtered))
    df = df_filtered.copy()
    
    # transform likert scales
    df = transform_likert(df)
    print(len(df))
    
    # remove double ipas
    df = filter_double_ipa(df)
    print(len(df))
    
    df = df.add_prefix(f'{s}_')
    
    df_dict[s] = df
    print(len(df))    

# merge both surveys and check for uniqueness
print(md.KEYS.duplicated().any())
print(df_dict['S1']['S1_id'].duplicated().any())
print(md.KEYS['OLD IDS'].isin(df_dict['S1']['S1_id']).any())
merged_waves_df = (
    md.KEYS
    .merge(df_dict['S1'], how="inner", left_on="NEW IDS", right_on="S1_id")
    .merge(df_dict['S2'], how="inner", left_on="OLD IDS", right_on="S2_m")
)


# reset unique ids
df_cleaned = merged_waves_df.reset_index(drop=True)
df_cleaned["respondent_id"] = df_cleaned.index + 1
print(len(df_cleaned))

# map demographics
df_cleaned = demographics_mapping(df_cleaned, md.DEMOGRAPHICS_DICT, column_patterns=[r"^gender$", r"^age$", r"^education$", r"^income$", r"^language$", r"^language_region$", r"^party_choice$"])

