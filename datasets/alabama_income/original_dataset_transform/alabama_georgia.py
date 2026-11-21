import pandas as pd
from folktables import ACSDataSource, ACSIncome

occps_codes = pd.read_csv("occp_translate_codes.csv")
occps_codes = dict(zip(occps_codes["code"].astype(int), occps_codes["occupation"]))

pobp_codes = pd.read_csv("pobp_translate_codes.csv")
pobp_codes = dict(zip(pobp_codes["code"].astype(int), pobp_codes["region"]))

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ga_data = data_source.get_data(states=["GA"], download=True)
al_data = data_source.get_data(states=["AL"], download=True)

ga_features, ga_labels, _ = ACSIncome.df_to_pandas(ga_data)
ga_full = pd.concat([ga_features, ga_labels], axis=1)
print("ga_full", len(ga_full))
ga_count = ga_full["PINCP"].value_counts(normalize=True)
print(ga_count)

al_features, al_labels, _ = ACSIncome.df_to_pandas(al_data)
al_full = pd.concat([al_features, al_labels], axis=1)
print("al_full", len(al_full))
al_count = al_full["PINCP"].value_counts(normalize=True)
print(al_count)

sample_size = min(len(al_full), len(ga_full))

ga_full = ga_full.sample(n=sample_size, random_state=42)
print(ga_full["PINCP"].value_counts(normalize=True))

al_full = al_full.sample(n=sample_size, random_state=42)
print(al_full["PINCP"].value_counts(normalize=True))

ga_full["OCCP"] = ga_full["OCCP"].astype(int)
ga_full["OCCP"] = ga_full["OCCP"].map(occps_codes).fillna("Other")

al_full["OCCP"] = al_full["OCCP"].astype(int)
al_full["OCCP"] = al_full["OCCP"].map(occps_codes).fillna("Other")

ga_full["POBP"] = ga_full["POBP"].astype(int)
ga_full["POBP"] = ga_full["POBP"].map(pobp_codes).fillna("Other")

al_full["POBP"] = al_full["POBP"].astype(int)
al_full["POBP"] = al_full["POBP"].map(pobp_codes).fillna("Other")

ga_full.to_csv('georgia_income.csv', index=False, float_format="%g")
al_full.to_csv('alabama_income.csv', index=False, float_format="%g")