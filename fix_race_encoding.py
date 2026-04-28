"""Re-derive race_encoded as 6 distinct categories instead of 5.

The original encoding collapsed Non-Hispanic Unknown Race into the White
category (both became race_encoded=0). This script uses the more granular
Race_and_origin_recode column to produce a clean 6-category encoding:

    0: Non-Hispanic White
    1: Non-Hispanic Black
    2: Hispanic (All Races)
    3: Non-Hispanic Asian or Pacific Islander
    4: Non-Hispanic American Indian/Alaska Native
    5: Non-Hispanic Unknown Race
"""
import pandas as pd

CSV_PATH = 'breast_cancer_4quantum.csv'
RACE_ORIGIN_COL = 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total'

df = pd.read_csv(CSV_PATH)
print(f"Loaded {CSV_PATH}: {df.shape}")

if RACE_ORIGIN_COL not in df.columns:
    raise SystemExit(f"Column not found: {RACE_ORIGIN_COL}. "
                     f"Cannot re-derive race encoding.")

print(f"\nBefore (race_encoded):")
print(df['race_encoded'].value_counts().sort_index())

# 6-category mapping from Race_and_origin_recode_*
race_origin_map = {
    'Non-Hispanic White': 0,
    'Non-Hispanic Black': 1,
    'Hispanic (All Races)': 2,
    'Non-Hispanic Asian or Pacific Islander': 3,
    'Non-Hispanic American Indian/Alaska Native': 4,
    'Non-Hispanic Unknown Race': 5,
}

new_encoded = df[RACE_ORIGIN_COL].map(race_origin_map)
n_unmapped = new_encoded.isna().sum()
if n_unmapped > 0:
    unmapped_values = df.loc[new_encoded.isna(), RACE_ORIGIN_COL].value_counts()
    print(f"\nWARNING: {n_unmapped} rows could not be mapped:")
    print(unmapped_values)
    print("These will be assigned to category 5 (Unknown).")
    new_encoded = new_encoded.fillna(5).astype(int)
else:
    new_encoded = new_encoded.astype(int)

df['race_encoded'] = new_encoded

print(f"\nAfter (race_encoded):")
print(df['race_encoded'].value_counts().sort_index())

print(f"\n6-category mapping:")
for label, code in race_origin_map.items():
    n = (df['race_encoded'] == code).sum()
    print(f"  race_encoded={code} ({label}): {n} patients")

df.to_csv(CSV_PATH, index=False)
print(f"\nSaved updated CSV: {CSV_PATH}")
