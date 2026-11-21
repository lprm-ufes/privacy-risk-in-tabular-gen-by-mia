import pandas as pd

translate_dict = {"code": [], "region": []}

for i in range(61, 100):
    translate_dict["code"].append(i)
    translate_dict["region"].append("PI")

translate_dict["code"].append(303)
translate_dict["region"].append("LA")
for i in range(310, 400):
    translate_dict["code"].append(i)
    translate_dict["region"].append("LA")

for i in range(158, 160):
    translate_dict["code"].append(i)
    translate_dict["region"].append("AS")

translate_dict["code"].append(161)
translate_dict["region"].append("AS")

for i in range(200, 300):
    translate_dict["code"].append(i)
    translate_dict["region"].append("AS")

for i in range(100, 158):
    translate_dict["code"].append(i)
    translate_dict["region"].append("EU")

translate_dict["code"].append(160)
translate_dict["region"].append("EU")

for i in range(162, 200):
    translate_dict["code"].append(i)
    translate_dict["region"].append("EU")

for i in range(400, 500):
    translate_dict["code"].append(i)
    translate_dict["region"].append("AF")

for i in range(300, 303):
    translate_dict["code"].append(i)
    translate_dict["region"].append("NA")

for i in range(304, 309):
    translate_dict["code"].append(i)
    translate_dict["region"].append("NA")

translate_dict["code"].append(60)
translate_dict["region"].append("OS")
for i in range(500, 555):
    translate_dict["code"].append(i)
    translate_dict["region"].append("OS")

df_world = pd.DataFrame(translate_dict)

df_states = pd.read_csv('POBP_CODE_US_REGION_TRANSLATE.csv', encoding='utf-8')

df_full = pd.concat([df_states, df_world])

df_full.to_csv('pobp_translate_codes.csv', index=False)