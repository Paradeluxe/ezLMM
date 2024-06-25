import itertools

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

lmerTest = importr('lmerTest')
emmeans = importr('emmeans')
stats = importr("stats")
Matrix = importr("Matrix")
nlme = importr("nlme")
lme4 = importr("lme4")

pd.set_option("display.max_columns", None)


def r2p(r_obj):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(r_obj)


def extract_contrast(contrast_str, interaction=0):
    raw_contrast = contrast_str.split("\n")
    if interaction == 0:
        contrast_dict = dict(zip(
            raw_contrast[0][1:].split(),
            raw_contrast[1].strip().rsplit(maxsplit=5)
        ))
    elif interaction == 1:
        contrast_dict = dict(zip(
            raw_contrast[1][1:].split(),
            raw_contrast[2].strip().rsplit(maxsplit=5)
        ))
        contrast_dict["under_cond"] = raw_contrast[0].strip(":").replace(" = ", "")
    # print(contrast_dict)
    # {'contrast': '     (Tsyl-0.5) - Tsyl0.5', 'estimate': '-0.161', 'SE': '0.0475', 'df': '21.8', 't.ratio': '-3.394', 'p.value': '0.0026'}
    return contrast_dict


# ---------------------------------
# ----------> For USERS >----------
# ---------------------------------

# ---------------------------------
# Step 1/5: Select SUBSET!!!
# ---------------------------------

# Read .csv data (it can accept formats like .xlsx, just change pd.read_XXX)
data = pd.read_csv("Data_Experiment.csv", encoding="utf-8")

print("Reading Data——>>>", end="")
# ---------------------------------
# Step 2/5: Select SUBSET!!!
# ---------------------------------
# Note: If you do not want to select subset,
# delete the line(s) you do not need, or press ctrl+/ annotating the line(s).


# Preserve data only within 2.5 * SD
new_data = pd.DataFrame()

for sub in list(set(data["sub"])):
    sub_data = data[data["sub"] == sub]

    # 计算'rt'列的均值和标准差
    mean_rt = sub_data['rt'].mean()
    std_rt = sub_data['rt'].std()

    # 根据均值和标准差筛选数据
    filtered_sub_data = sub_data[(sub_data['rt'] > (mean_rt - 2.5 * std_rt)) &
                                 (sub_data['rt'] < (mean_rt + 2.5 * std_rt))]
    # 将筛选后的数据追加到new_data中

    new_data = pd.concat([new_data, filtered_sub_data], ignore_index=True)

data = new_data.copy()

# Add consistency col
for sub in list(set(data["sub"])):
    for word in list(set(data["word"])):
        data['consistency'] = (((data['priming'] == "priming") & (data['exp_type'] == "exp1")) | ((data['priming'] == "primingeq") & (data['exp_type'] == "exp2"))).astype(int)
"""
# Save only both exists
for sub in list(set(data["sub"].tolist())):
    for word in list(set(data["word"].tolist())):

        if not len(data[(data["sub"] == sub) & (data["word"] == word)]) == 2:
            data = data[~((data["sub"] == sub) & (data["word"] == word))]


# Subtract A with B
df1 = data[data['consistency'] == 1]
df1 = df1.sort_values(by=["sub", "word"])
df1 = df1.reset_index(drop=False)

df0 = data[data['consistency'] == 0]
df0 = df0.sort_values(by=["sub", "word"])
df0 = df0.reset_index(drop=False)


df1["rt_diff"] = df1["rt"] - df0["rt"]

data = df1.copy()
"""

data = data[data['ifcorr'] == 1]  # rt data works on ACC = 1

data['rt'] = data['rt'] * 1000  # if rt is in ms, * 1000 might be better
# data = data[data['exp_type'] == "exp1"]  # pick out one exp
data = data[data['ifanimal'] == True]  # pick out one exp

print("Data collected!")
# ---------------------------------
# Step 3/5: Code your variables!!!
# ---------------------------------

# * Optional for categorical variables
# ** Should-do for ordinal variables
# *** must-do for continuous variables

# I adopted Orthogonal Polynomial Coding here:
# 2-level -> -0.5, 0.5
# 3-level -> -1, 0, 1
# 4-level -> -0.671, -0.224, 0.224, 0.671 (not sure why it is like this)

# Ref:
# https://online.stat.psu.edu/stat502/lesson/10/10.2
# https://stats.oarc.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis/

data['Tpriming'] = -0.5 * (data['priming'] == "priming") + 0.5 * (data['priming'] != "priming")
data['Tsyl'] = -0.5 * (data['syl'] == 2) + 0.5 * (data['syl'] != 2)
data['Texp_type'] = -0.5 * (data['exp_type'] == "exp1") + 0.5 * (data['exp_type'] != "exp1")
data["Tifanimal"] = -0.5 * (data['ifanimal'] == True) + 0.5 * (data['ifanimal'] != True)
data["Tconsistency"] = -0.5 * (data['consistency'] == 1) + 0.5 * (data['consistency'] != 1)

# data.to_csv("Data_Exp_rtdiff.csv", index=False)
# ---------------------------------
# Step 4/5: Write your variables and create Formula
# ---------------------------------

dep_var = "rt"
fixed_factor = ["Tsyl", "Tconsistency", "Texp_type"]
random_factor = ["sub", "word"]

# ---------------------------------
# ABOVE IS OK. DO NOT TOUCH BELOW.
# ---------------------------------


fixed_str = " * ".join(fixed_factor)
fixed_combo = []
for i in range(len(fixed_factor), 0, -1):  # 从1开始，因为0会生成空集
    for combo in itertools.combinations(fixed_factor, i):
        combo = list(combo)
        combo.sort()
        fixed_combo.append(":".join(combo))
        # print(":".join(combo))


# ---------------------------------
# Step 5/5 [Optional]: If you want to skip a few formulas
# ---------------------------------

prev_formula = "rt ~ Tsyl * Tconsistency * Texp_type + (1 | sub) + (1 | word)" # "rt ~ Tsyl * Tconsistency * Texp_type + (1 | sub) + (1 | word)"


# ---------------------------------
# ----------< For USERS <----------
# ---------------------------------

if prev_formula:
    random_model = {kw[1]: kw[0].split(" + ") for kw in [full_item.split(")")[0].split(" | ") for full_item in prev_formula.split("(1 + ")[1:]]}
    for key in random_factor:
        try:
            random_model[key]
        except KeyError:
            random_model[key] = []
else:
    random_model = {key: fixed_combo[:] for key in random_factor}

# Change pandas DataFrame into R language's data.frame
with (ro.default_converter + pandas2ri.converter).context():
    r_data = ro.conversion.get_conversion().py2rpy(data)
while True:
    random_str = ""
    for key in random_model:
        if not random_model[key]:  # has no element
            random_str += "(1" + f" | {key}) + "
        else:  # has element
            random_str += "(1 + " + " + ".join(random_model[key]) + f" | {key}) + "
    random_str = random_str.rstrip(" + ")

    formula_str = f"{dep_var} ~ {fixed_str} + {random_str}"
    formula = Formula(formula_str)
    print(f"Running FORMULA: {formula_str}")

    # 使用lmer函数拟合模型
    model1 = lmerTest.lmer(formula, REML=True, data=r_data, control=lme4.lmerControl("bobyqa"))
    summary_model1_r = Matrix.summary(model1)
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        summary_model1 = ro.conversion.get_conversion().rpy2py(summary_model1_r)
    try:
        isWarning = eval(str(summary_model1["optinfo"]["conv"]['lme4']["messages"]).strip("o"))
    except KeyError:
        isWarning = False

    # Transform random table to DataFrame format
    random_table = []
    corrs_supp = []
    lines = str(nlme.VarCorr(model1)).strip().split('\n')
    # print(lines)
    for line in lines[1:]:  # Exclude the 1st
        # Check if the 2nd element is number
        elements = line.strip().split()
        if not elements:
            continue
        if elements[0].split(".")[0].strip("-").isnumeric():
            corrs_supp.extend([float(e) if e.split(".")[0].strip("-").isnumeric() else e for e in elements])
            continue
        # print(elements)
        if elements[1].strip("-").split(".")[0].isnumeric():
            elements = [Groups] + elements
        else:
            Groups = elements[0]
        elements = [float(e) if e.split(".")[0].strip("-").isnumeric() else e for e in elements]
        # print(elements[0], elements[1])
        if elements[1] == "Residual":
            break

        random_table.append(elements)
        # print(elements)

    df = pd.DataFrame(random_table)

    df = df[df[1] != '(Intercept)']  # ignore all the "(intercept)"

    # print(df)
    # Check if there is any corr item that is >= 0.90
    all_corrs = np.array(df.iloc[:, 3:].dropna(how="all")).flatten().tolist()
    all_corrs = [corr for corr in all_corrs if isinstance(corr, (int, float))]
    all_corrs.extend(corrs_supp)

    # print(all_corrs)
    isTooLargeCorr = any(corr >= .9 for corr in all_corrs)

    isGoodModel = not isWarning and not isTooLargeCorr

    if isGoodModel:
        print(f"Formula {formula_str} is a good model.")
        break
    else:
        if not any(random_model.values()):
            print("Every model failed")
            break

        rf2ex = df.loc[df[2].idxmin(0)][0]
        ff2ex = df.loc[df[2].idxmin(0)][1]

        print(f"Exclude random model item: {ff2ex} | {rf2ex}")

        # Processing EXCLUSION
        for ff2ex_item in random_model[rf2ex]:
            if sorted(ff2ex_item.split(":")) == sorted(ff2ex.split(":")):
                random_model[rf2ex].remove(ff2ex_item)
        # print(random_model)
        print("---\n---")

    # ('methTitle', 'objClass', 'devcomp', 'isLmer', 'useScale', 'logLik', 'family', 'link', 'ngrps', 'coefficients', 'sigma', 'vcov', 'varcor', 'AICtab', 'call', 'residuals', 'fitMsgs', 'optinfo', 'corrSet')
    # ('optimizer', 'control', 'derivs', 'conv', 'feval', 'message', 'warnings', 'val')


print(summary_model1_r)
anova_model1 = stats.anova(model1, type=3, ddf="Kenward-Roger")

# anova_model1 change format
anova_model1 = r2p(anova_model1)

print(anova_model1)

# model1 = lmerTest.lmer(Formula("rt ~ Tpriming * Tsyl + (1 | sub) + (1 | word)"), REML=True, data=r_data)
# summary_model1_r = Matrix.summary(model1)
# print(summary_model1_r)


print("--------------- Generating reports here ---------------\n")



final_rpt = f"For RT data, F test of the optimal model was conducted using anova function from stats package. "

for sig_items in anova_model1[anova_model1["Pr(>F)"] <= 0.05].index.tolist():
    if "ntercept" in sig_items:
        continue
    sig_items = sig_items.split(":")
    item_num = len(sig_items)
    df_item = anova_model1[anova_model1["Pr(>F)"] <= 0.05].loc[sig_items[0]]
    """
    Name: Tsyl, dtype: float64
    Sum Sq      1.120534
    Mean Sq     1.120534
    NumDF       1.000000
    DenDF      21.842760
    F value    11.519495
    Pr(>F)      0.002627
    Name: Tsyl, dtype: float64
    """
    if item_num == 1:
        print(f"Main effect {sig_items}")
        emmeans_result = emmeans.contrast(emmeans.emmeans(model1, sig_items[0]), "pairwise", adjust="bonferroni")
        emmeans_result_dict = extract_contrast(str(emmeans_result))


        final_rpt += f"The main effect of {sig_items[0]} was significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "
        final_rpt += f"Post-hoc analysis revealed that "
        if float(emmeans_result_dict['p.value']) <= 0.05:
            final_rpt += f"RT for {emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')} "\
                         f"was significantly {'higher' if float(emmeans_result_dict['estimate']) < 0 else 'lower'}"\
                         f" than that for {emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')} ("\
                         f"β={emmeans_result_dict['estimate']}, "\
                         f"SE={emmeans_result_dict['SE']}, "\
                         f"df={emmeans_result_dict['df']}, "\
                         f"t={emmeans_result_dict['t.ratio']}, "\
                         f"p={float(emmeans_result_dict['p.value']):.3f}). "
        else:
            final_rpt += f"there was no significant difference between {emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')}"\
                         f" and {emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')} in RT ("\
                         f"β={emmeans_result_dict['estimate']}, "\
                         f"SE={emmeans_result_dict['SE']}, "\
                         f"df={emmeans_result_dict['df']}, "\
                         f"t={emmeans_result_dict['t.ratio']}, "\
                         f"p={float(emmeans_result_dict['p.value']):.3f}). "

    elif item_num == 2:
        print(f"2-way Interaction {sig_items}")
        final_rpt += f"The interaction between {' and '.join(sig_items)} was significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "
        final_rpt += f"Simple effect analysis showed that, "

        for i1, i2 in [(0, 1), (-1, -2)]:
            emmeans_result = emmeans.contrast(emmeans.emmeans(model1, specs=sig_items[i1], by=sig_items[i2]), "pairwise", adjust="bonferroni")
            # print(emmeans_result)
            emmeans_result_dict = extract_contrast(str(emmeans_result), 1)
            final_rpt += f"under the condition of {emmeans_result_dict['under_cond']}, "

            if float(emmeans_result_dict['p.value']) <= 0.05:
                final_rpt += f"RT for {emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')} "\
                             f"was significantly {'higher' if float(emmeans_result_dict['estimate']) < 0 else 'lower'}"\
                             f" than that for {emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')} ("\
                             f"β={emmeans_result_dict['estimate']}, "\
                             f"SE={emmeans_result_dict['SE']}, "\
                             f"df={emmeans_result_dict['df']}, "\
                             f"t={emmeans_result_dict['t.ratio']}, "\
                             f"p={float(emmeans_result_dict['p.value']):.3f})"
            else:
                final_rpt += f"no significant difference was found between {emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')}"\
                             f" and {emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')} in RT ("\
                             f"β={emmeans_result_dict['estimate']}, "\
                             f"SE={emmeans_result_dict['SE']}, "\
                             f"df={emmeans_result_dict['df']}, "\
                             f"t={emmeans_result_dict['t.ratio']}, "\
                             f"p={float(emmeans_result_dict['p.value']):.3f})"
            if i1 == 0:
                final_rpt += "; "
            elif i1 == -1:
                final_rpt += ". "



    elif item_num >= 3:
        print(f"3-way Interaction {sig_items} (under construction, use R for 3-way simple effect analysis please)")

for sig_items in anova_model1[anova_model1["Pr(>F)"] > 0.05].index.tolist():
    if "ntercept" in sig_items:
        continue
    sig_items = sig_items.split(":")
    item_num = len(sig_items)
    df_item = anova_model1[anova_model1["Pr(>F)"] > 0.05].loc[sig_items[0]]
    """
    Name: Tsyl, dtype: float64
    Sum Sq      1.120534
    Mean Sq     1.120534
    NumDF       1.000000
    DenDF      21.842760
    F value    11.519495
    Pr(>F)      0.002627
    Name: Tsyl, dtype: float64
    """
    if item_num == 1:
        print(f"Main effect {sig_items}")
        emmeans_result = emmeans.contrast(emmeans.emmeans(model1, sig_items[0]), "pairwise", adjust="bonferroni")
        emmeans_result_dict = extract_contrast(str(emmeans_result))

        final_rpt += f"The main effect of {sig_items[0]} was not significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "

    elif item_num == 2:
        print(f"2-way Interaction {sig_items}")
        final_rpt += f"The interaction between {' and '.join(sig_items)} was not significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "

    elif item_num >= 3:
        print(f"3-way Interaction {sig_items} (under construction, use R for 3-way simple effect analysis please)")


rep_terms = {
    "Tsyl-0.5": "disyllable",
    "Tsyl0.5": "trisyllable",
    "Tsyl": "syllable number",

    "Texp_type-0.5": "stress-timing",
    "Texp_type0.5": "syllable-timing",
    "Texp_type": "isochrony"

}
for rep_term in rep_terms:
    final_rpt = final_rpt.replace(rep_term, rep_terms[rep_term])

print(final_rpt)
print()
print()


print(f"Last formula is {formula_str}\nIt is {isGoodModel}\n")
print("-------------------------------------------------------")
print("SCRIPT End √ | Ignore \"R[write to console]\" down below, as it is an automatic callback")
print("-------------------------------------------------------")

