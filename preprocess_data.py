import pandas as pd

# Read .csv lmm_data (it can accept formats like .xlsx, just change pd.read_XXX)
data = pd.read_csv("Data_Experiment.csv", encoding="utf-8")

print("Reading Data——>>>", end="")
# ---------------------------------
# Step 2/5: Select SUBSET!!!
# ---------------------------------
# Note: If you do not want to select subset,
# delete the line(s) you do not need, or press ctrl+/ annotating the line(s).

# Add consistency col

data['consistency'] = (((data['priming'] == "priming") & (data['exp_type'] == "exp1")) | ((data['priming'] == "primingeq") & (data['exp_type'] == "exp2"))).astype(int)

# data = data[data['ifcorr'] == 1]  # rt lmm_data works on ACC = 1

data['rt'] = data['rt'] * 1000  # if rt is in ms, * 1000 might be better
# lmm_data = lmm_data[lmm_data['exp_type'] == "exp1"]  # pick out one exp
data = data[data['ifanimal'] == True]  # pick out one exp


data["syl"] = data["syl"].map({2: "disyllabic", 3: "trisyllabic"})
data["exp_type"] = data["exp_type"].map({"exp2": "averaged", "exp1": "original"})
data["consistency"] = data["consistency"].map({0: "inconsistent", 1: "consistent"})


data.rename(columns={"ifcorr": "acc", "sub": "subject", "consistency": "priming_effect", "exp_type": "speech_isochrony", "syl": "syllable_number"}, inplace=True)

data = data[["rt", "acc", "subject", "word", "priming_effect", "speech_isochrony", "syllable_number"]]




data.to_csv("data.csv", index=False)

print("Data collected!")

