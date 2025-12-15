# ------------------------------------------------------------------
# Load Libraries & Data
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
print(os.getcwd())

# ------------------------------------------------------------------
# transformation dictionaries
# ------------------------------------------------------------------

valid_columns_s1 = list(set([
    "sensitivity_nh_1", "sensitivity_nh_2", "sensitivity_nh_3", "sensitivity_nh_4",
    "likert_costs_1", "likert_costs_2", "likert_costs_3", "likert_costs_4",
    "likert_costs_5", "likert_costs_6", "likert_costs_7",
    "likert_municipality_1", "likert_municipality_2", "likert_municipality_3",
    "likert_municipality_4", "likert_municipality_5",
    "1_conjoint_acceptance_1", "1_conjoint_acceptance_2",
    "2_conjoint_acceptance_1", "2_conjoint_acceptance_2",
    "3_conjoint_acceptance_1", "3_conjoint_acceptance_2",
    "4_conjoint_acceptance_1", "4_conjoint_acceptance_2",
    "5_conjoint_acceptance_1", "5_conjoint_acceptance_2",
    "6_conjoint_acceptance_1", "6_conjoint_acceptance_2",
    "own_municipality_1", "own_municipality_2", "own_municipality_3", "own_municipality_4",
    "gal_tan_1", "gal_tan_2", "gal_tan_3", "gal_tan_4",
    "lreco_1", "lreco_2", "lreco_3",
    "env_values_1", "env_values_2", "env_values_3",
    "climatechange_nh_1", "climatechange_nh_2", "climatechange_nh_3",
    "costs_cc_policy_1", "finan_vulnerability_1"
]))

likert_map = {
    "1 - Extremely unlikely": 1,
    "1 - Strongly disagree": 1,
    "1 - Totally unacceptable": 1,
    "1 - Very unlikely": 1,
    "1 - Very difficult": 1,
    "6 - Totally acceptable": 6,
    "6 - Strongly agree": 6,
    "6 - Very likely": 6,
    "6 - Very easy": 6,
    "6 - Extremely worried": 6,
    "6 - Extremely likely": 6,
    "I don't know": np.nan,
    "Prefer not to say": np.nan,
    "I don't now": np.nan,
    "7": np.nan
}


# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------

def transform_likert(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.where(~s.isna(), None)
    mapped = s.map(likert_map)
    needs_parse = mapped.isna() & s.notna()
    if needs_parse.any():
        nums = pd.to_numeric(
            s[needs_parse].str.extract(r"(-?\d+\.?\d*)", expand=False),
            errors="coerce",
        )
        mapped = mapped.astype("float")
        mapped.loc[needs_parse] = nums

    return mapped.astype("float")

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
# read first wave data
df = pd.read_csv("./data/Survey_Adaptation Natural Hazards_First Wave_raw data.csv", dtype={"id": "string"})
print(df)
df_second = pd.read_csv("./data/Survey_Adaptation Natural Hazards_Second Wave_raw data.csv", dtype={"id": "string"})

# remove lines and replace empty cells and rename columnames
df = df.iloc[2:].copy()
df = df.replace("", pd.NA)
df.columns = df.columns.str.replace(".", "", regex=False)
df.columns = df.columns.str.replace(r"\s+", "", regex=True)

# Filter: incompletes, screened out, poor quality, quota met
df = df[(df["DistributionChannel"] != "preview") & (df["Finished"] != False)].copy()
df = df[~df["Q_TerminateFlag"].isin(["PoorQuality", "NA", "QuotaMet", "Screened"])].copy()


# ------------------------------------------------------------------
# 2. Transform likert-scales to numerical values

# Only select named columns and map correct likert number in case for string reply
valid_columns_s1 = [c for c in valid_columns_s1 if c in df.columns]
df[valid_columns_s1] = df[valid_columns_s1].apply(transform_likert)

# # psychological distance groups
# psy_distance_s2 = list(set([
#     "identity_group_1", "identity_group_2", "identity_group_3", "identity_group_4",
#     "identity_group_5", "identity_group_6", "identity_group_7", "identity_group_8"
# ]))

# financial vulnerability groups
finan_vulnerability_s1 = list(set([
    "identity_group_1", "identity_group_2", "identity_group_3", "identity_group_4",
    "identity_group_5", "identity_group_6", "identity_group_7", "identity_group_8"
]))

# natural hazard vulnerability groups
nh_vulnerability_s1 = list(set([
    "identity_group_1", "identity_group_2", "identity_group_3", "identity_group_4",
    "identity_group_5", "identity_group_6", "identity_group_7", "identity_group_8"
]))
columns_to_transform2 = [c for c in columns_to_transform2 if c in df.columns]




# ----------------------#
# 3. Remove speeding and very slow respondents.
#----------------------#

# Umbenennen von Duration(inseconds) -> duration (nach dem Spalten-Cleaning)
if "Duration(inseconds)" in df.columns:
    df = df.rename(columns={"Duration(inseconds)": "duration"})

# Sicherstellen, dass duration numerisch ist
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

p5 = df["duration"].quantile(0.05)
p95 = df["duration"].quantile(0.95)

df = df[(df["duration"] > p5) & (df["duration"] < p95)].copy()

min_duration = df["duration"].min(skipna=True)
max_duration = df["duration"].max(skipna=True)

# ----------------------#
# 4. Removing straightliners
# (respondents who gave the same answer across a set of Likert-scale questions)

# check 
likert_costs_cols = [c for c in df.columns if c.startswith("likert_costs_")]
identity_cols = [c for c in df.columns if c.startswith("identity_group")]
gal_tan_cols = [c for c in df.columns if c.startswith("gal_tan_")]
deservingness = [c for c in df.columns if c.startswith("gal_tan_")]


df[likert_costs_cols] = df[likert_costs_cols].apply(pd.to_numeric, errors="coerce")
df[identity_cols] = df[identity_cols].apply(pd.to_numeric, errors="coerce")
df[gal_tan_cols] = df[gal_tan_cols].apply(pd.to_numeric, errors="coerce")

def is_straightliner(sub_df: pd.DataFrame) -> pd.Series:
    if sub_df.shape[1] == 0:
        return pd.Series(False, index=sub_df.index)
    # Anzahl verschiedener Werte (ohne NaN) == 1
    return sub_df.nunique(axis=1, dropna=True).eq(1)

straightliners_costs = is_straightliner(df[likert_costs_cols])
straightliners_identity = is_straightliner(df[identity_cols])
straightliners_gal_tan = is_straightliner(df[gal_tan_cols])

straightliners = straightliners_costs | straightliners_identity | straightliners_gal_tan

df_cleaned = df.loc[~straightliners].copy()

# ----------------------#
# 5. Deleting everybody for which the IP address was used more than 2 times
#----------------------#

# IPs mit mehr als 2 verschiedenen ids
shared_ips = (
    df_cleaned.groupby("IPAddress")["id"]
    .nunique()
    .reset_index(name="user_count")
)
shared_ips = shared_ips[shared_ips["user_count"] > 2]["IPAddress"]

# Nutzer mit diesen IPs
users_to_remove = df_cleaned[df_cleaned["IPAddress"].isin(shared_ips)].copy()

# anti_join nach id & IPAddress
df_cleaned = df_cleaned.merge(
    users_to_remove[["id", "IPAddress"]].drop_duplicates(),
    on=["id", "IPAddress"],
    how="left",
    indicator=True,
)
df_cleaned = df_cleaned[df_cleaned["_merge"] == "left_only"].drop(columns="_merge").copy()

# ----------------------#
# 6. Add unique response IDs
#----------------------#

df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned["respondent_id"] = df_cleaned.index + 1

# ----------------------#
# 7. Recode demographic values
#----------------------#

def apply_mapping(df_in: pd.DataFrame, mapping_dict: dict, column_patterns=None) -> pd.DataFrame:
    df_out = df_in.copy()
    if column_patterns is not None:
        import re

        compiled = [re.compile(p) for p in column_patterns]
        columns_to_map = [
            c for c in df_out.columns
            if any(p.search(c) for p in compiled)
        ]
    else:
        columns_to_map = list(df_out.columns)

    for col in columns_to_map:
        # in String konvertieren und trimmen
        s = df_out[col].astype("string").str.strip()

        # Unbekannte Werte warnen
        uniques = set(s.dropna().unique())
        unmapped = uniques - set(mapping_dict.keys())
        if unmapped:
            warnings.warn(
                f"Unmapped values in column {col}: {', '.join(sorted(map(str, unmapped)))}"
            )

        df_out[col] = s.map(mapping_dict)

    return df_out

demographics_dict = {
    # gender
    "Female": "Female",
    "Male": "Male",
    "Non-binary / Other": np.nan,
    "Prefer not to say ": np.nan,  # beachte Leerzeichen wie im R-Code

    # age
    "18 - 34": "18 - 34",
    "35 - 49": "35 - 49",
    "50 or older": "50+",

    # education
    "No high school diploma": "Below Secondary",
    "Vocational training or apprenticeship": "Vocational training or apprenticeship",
    "High school diploma": "High school diploma",
    "Bachelor's degree": "University degree",
    "Master's degree": "University degree",
    "Doctoral or professional degree (e.g. PhD, MD, JD)": "University degree",
    "Prefer not to say": np.nan,

    # income
    "Less than CHF 50,000": "Low",
    "CHF 50,000 - CHF 70,000": "Low",
    "CHF 70,000 - CHF 100,000": "Mid",
    "CHF 100,001 - CHF 150,000": "Mid",
    "CHF 150,001 - CHF 250,000": "High",
    "More than CHF 250,000": "High",
    "I don't know": np.nan,
    # nochmal "Prefer not to say" (ohne Leerzeichen)
    "Prefer not to say": np.nan,

    # language
    "Deutsch": "German",
    "Français": "French",
    "English": "English",
    "Italiano": "Italian",

    # region
    "German-speaking region": "German-speaking region",
    "Italian-speaking region": "Italian-speaking region",
    "French-speaking region": "French-speaking region",
    "Romansh-speaking region": "Romansh-speaking region",

    # political
    "Social Democratic Party (SP)": "Left",
    "The Greens (GPS)": "Left",
    "Green Liberals (GLP)": "Liberal",
    "The Liberals (FDP)": "Liberal",
    "The Middle Party (merger between the Christian Democratic People's Party (CVP) and the Civic Democratic Party (BDP))": "Liberal",
    "Swiss People's Party (SVP)": "Conservative",
    "Federal Democratic Union (EDU)": "Conservative",
    "Evangelical People's Party of Switzerland (EPP)": "Conservative",
    "Mouvement Citoyens Genevois (MCG)": "Conservative",
    "Lega dei Ticinesi (Lega)": "Conservative",
    "Others, such as": np.nan,
    "I don't feel close to any party": "No party association",
    "Prefer not to say": np.nan,
}

df_cleaned = apply_mapping(
    df_cleaned,
    demographics_dict,
    column_patterns=[
        r"^gender$", r"^age$", r"^education$", r"^income$",
        r"^language$", r"^language_region$", r"^party_choice$"
    ],
)

# ----------------------#
# 8. Save clean data and analyse demographics
#----------------------#

out_path = Path("02_data") / "data_clean_first_wave.csv"
df_cleaned.to_csv(out_path, index=False)

# n_sample für population_margins
n_sample = len(df_cleaned)

population_margins = {
    "age": {
        "18 - 34": 0.2707 * n_sample,
        "35 - 49": 0.2772 * n_sample,
        "50 or older": 0.4522 * n_sample,
    },
    "gender": {
        "Female": 0.5002 * n_sample,
        "Male": 0.4998 * n_sample,
    },
    "language_region": {
        "German-speaking region": 0.7114 * n_sample,
        "French-speaking region": 0.2467 * n_sample,
        "Italian-speaking region": 0.0419 * n_sample,
    },
    "education": {
        "Below Secondary": 0.008 * n_sample,
        "High school diploma": 0.409 * n_sample,
        "University degree": 0.438 * n_sample,
        "Vocational training or apprenticeship": 0.145 * n_sample,
    },
}

# Deskriptive Auswertungen (entspricht group_by + summarise + mutate in R)

def distribution(df_, col):
    tab = (
        df_.groupby(col)
        .size()
        .reset_index(name="anzahl")
    )
    tab["prozent"] = (100 * tab["anzahl"] / tab["anzahl"].sum()).round(1)
    return tab

# Nur ausgeben, wenn die Spalten existieren
if "first_age" in df_cleaned.columns:
    print("first_age:")
    print(distribution(df_cleaned, "first_age"))

if "first_language_region" in df_cleaned.columns:
    print("\nfirst_language_region:")
    print(distribution(df_cleaned, "first_language_region"))

if "first_gender" in df_cleaned.columns:
    print("\nfirst_gender:")
    print(distribution(df_cleaned, "first_gender"))

if "first_education" in df_cleaned.columns:
    print("\nfirst_education:")
    print(distribution(df_cleaned, "first_education"))