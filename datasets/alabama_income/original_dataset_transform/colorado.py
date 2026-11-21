import pandas as pd
from folktables import ACSDataSource, ACSIncome

occps_codes = pd.read_csv("occp_translate_codes.csv")
occps_codes = dict(zip(occps_codes["code"].astype(int), occps_codes["occupation"]))

pobp_codes = pd.read_csv("pobp_translate_codes.csv")
pobp_codes = dict(zip(pobp_codes["code"].astype(int), pobp_codes["region"]))

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
co_data = data_source.get_data(states=["CO"], download=True)

co_features, co_labels, _ = ACSIncome.df_to_pandas(co_data)
co_full = pd.concat([co_features, co_labels], axis=1)
print("co_full", len(co_full))
co_count = co_full["PINCP"].value_counts(normalize=True)
print(co_count)

print(len(co_full))
sample_size = 22268

co_full = co_full.sample(n=sample_size, random_state=42)
print(co_full["PINCP"].value_counts(normalize=True))

co_full["OCCP"] = co_full["OCCP"].astype(int)
co_full["OCCP"] = co_full["OCCP"].map(occps_codes).fillna("Other")

co_full["POBP"] = co_full["POBP"].astype(int)
co_full["POBP"] = co_full["POBP"].map(pobp_codes).fillna("Other")

co_full.to_csv('colorado_income.csv', index=False, float_format="%g")