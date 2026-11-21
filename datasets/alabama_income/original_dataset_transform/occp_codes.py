import pandas as pd

translate_dict = {"code": [], "occupation": []}

for i in range(10, 441):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("MGR") # Managers

for i in range(500, 751):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("BUS") # Business Operations

for i in range(800, 961):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("FIN") # Financial

for i in range(1005, 1241):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("CMM") # Mathematical Science

for i in range(1305, 1561):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("ENG") # Enginnering

for i in range(1600, 1981):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("SCI") # Science

for i in range(2001, 2061):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("CMS") # Social assistance

for i in range(2100, 2181):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("LGL") # Legal

for i in range(2205, 2556):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("EDU") # Education

for i in range(2600, 2921):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("ENT") # Entertainment

for i in range(3000, 3551):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("MED") # Healthcare

for i in range(3601, 3656):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("HLS") # Others healthcare

for i in range(3700, 3961):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("PRT") # Protective Service

for i in range(4000, 4160):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("EAT") # Food Service

for i in range(4200, 4256):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("CLN") # Grounds Maintenance

for i in range(4330, 4656):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("PRS") # Personal Care

for i in range(4700, 4966):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("SAL") # Sales and Related Workers

for i in range(5000, 5941):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("OFF") # Office and Administrastive

for i in range(6005, 6131):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("FFF") # Logging Workers

for i in range(6200, 6766):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("CON") # Construction

for i in range(6800, 6951):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("EXT") # Extraction Workers

for i in range(7000, 7641):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("RPR") # Miscellaneous Production Workers

for i in range(9005, 9761):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("TRN") # Material Moving Workers

for i in range(9800, 9831):
    translate_dict["code"].append(i)
    translate_dict["occupation"].append("MIL") # Military

translate_dict["code"].append(9920)
translate_dict["occupation"].append("UNE") # Unemployed And Last Worked 5 Years Ago Or Earlier Or Never Worked

df_full = pd.DataFrame(translate_dict)

df_full.to_csv('occp_translate_codes.csv', index=False)