import itertools
import logging

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula, numpy2ri
# from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# print(__name__)
logging.disable(logging.CRITICAL)
# logging.getLogger("ezlmm.ezlmm").disabled = True
pd.set_option("display.max_columns", None)

lmerTest = importr('lmerTest')
emmeans = importr('emmeans')
stats = importr("stats")
Matrix = importr("Matrix")
lme4 = importr("lme4")
car = importr('car')
nlme = importr("nlme")


def r2p(r_obj):
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(r_obj)

def p2r(p_obj):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(p_obj)


def extract_contrast(contrast_str, factor_num=1):

    raw_contrast = contrast_str.strip().split("\n")
    contrast_dicts = []

    s = 0
    for i in [i for i in range(len(raw_contrast)) if not raw_contrast[i]]:
        contrast_dict = {}
        if factor_num == 1:
            raw_contrast_ = raw_contrast[s:i]

        elif factor_num == 2:
            contrast_dict["under_cond"] = raw_contrast[s].strip(":").replace("=", "").replace(" ", "")
            raw_contrast_ = raw_contrast[s+1:i]

        for j in range(0, len(raw_contrast_), 2):
            contrast_dict |= dict(zip(
                raw_contrast_[j].strip().split(),
                raw_contrast_[j + 1].strip().rsplit(maxsplit=len(raw_contrast_[j].strip().split()) - 1)
            ))

        contrast_dicts.append(contrast_dict)
        s += i+1

    return contrast_dicts




class LinearMixedModel:

    def __init__(self):
        self.path = None
        self.report = None
        self.formula = None

        # Data exclusion
        self.isExcludeSD = True

        # Name variables
        self.dep_var = ""
        self.indep_var = []
        self.random_var = []

        # Generate report
        self.trans_dict = {}
        self.delete_random_item = []

    def read_data(self, path):
        """
        Read data from the data path
        :param path: data file path
        :return: None
        """


        print("Reading Data...", end="")
        # Read .csv self.data (it can accept formats like .xlsx, just change pd.read_XXX)
        self.data = pd.read_csv(path, encoding="utf-8")
        print("Data collected!")

        return None

    def code_variables(self, code):
        """
        Change the values in each column with certain coding methods (customization)
        :param code: coding dict
        :return: None
        """

        print("Coding variables...", end="")
        if not code:
            for var in self.indep_var:
                code[var] = dict(zip(self.data[var].unique(), self.data[var].unique()))

        for var_name in code:
            # print(var_name, code[var_name])
            self.data[var_name] = self.data[var_name].apply(lambda x: code[var_name][x])

        for k in code:
            v = code[k]

            for cond in v:
                self.trans_dict[k + str(v[cond])] = cond
        print("Variables coded!")

        return None

    def exclude_trial_SD(self, target: str, subject: str, SD=2.5):
        """
        Attain only trials within 2.5 * SD for each participant/subject
        :param target: target column name
        :param subject: participant/subject column
        :param SD: default at 2.5
        :return: None
        """

        new_data = pd.DataFrame()
        for sub in list(set(self.data[subject])):
            sub_data = self.data[self.data[subject] == sub]

            # 计算'rt'列的均值和标准差
            mean_rt = sub_data[target].mean()
            std_rt = sub_data[target].std()

            # 根据均值和标准差筛选数据
            filtered_sub_data = sub_data[(sub_data[target] >= (mean_rt - SD * std_rt)) &
                                         (sub_data[target] <= (mean_rt + SD * std_rt))]
            # 将筛选后的数据追加到new_data中

            new_data = pd.concat([new_data, filtered_sub_data], ignore_index=True)

        self.data = new_data.copy()

        return None


    def fit(self, optimizer=[], prev_formula=""):

        dep_var = self.dep_var
        fixed_factor = self.indep_var


        random_factor = self.random_var


        fixed_str = " * ".join(fixed_factor)
        fixed_combo = []
        for i in range(len(fixed_factor), 0, -1):  # 从1开始，因为0会生成空集
            for combo in itertools.combinations(fixed_factor, i):
                combo = list(combo)
                combo.sort()
                fixed_combo.append(":".join(combo))
                # print(":".join(combo))

        if prev_formula:

            prev_formula = prev_formula.replace(" ", "")
            random_model = {kw[1]: kw[0].split("+") for kw in
                            [full_item.split(")")[0].split("|") for full_item in prev_formula.split("(1+")[1:]]}
            for key in random_factor:
                try:
                    random_model[key]
                except KeyError:
                    random_model[key] = []
        else:
            random_model = {key: fixed_combo[:] for key in random_factor}

        # Change pandas DataFrame into R language's self.data.frame
        r_data = p2r(self.data)

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
            print(f"\r[*] Running FORMULA -> {formula_str}", end="")

            if not optimizer:
                model1 = lmerTest.lmer(formula, REML=True, data=r_data)
            else:
                list_optCtrl = ro.ListVector([("maxfun", optimizer[1])])
                model1 = lmerTest.lmer(formula, REML=True, data=r_data, control=lme4.lmerControl(optimizer=optimizer[0], optCtrl=list_optCtrl))


            summary_model1_r = Matrix.summary(model1)

            summary_model1 = r2p(summary_model1_r)

            try:
                isWarning = eval(str(summary_model1["optinfo"]["conv"]['lme4']["messages"]).strip("o"))
            except KeyError:
                isWarning = False
            except SyntaxError:
                isWarning = True

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
                print(f"\r[√] FORMULA -> {formula_str}")
                break
            else:
                print(f"\r[×] FORMULA -> {formula_str}")

                if not any(random_model.values()):
                    break

                rf2ex = df.loc[df[2].idxmin(0)][0]
                ff2ex = df.loc[df[2].idxmin(0)][1]

                self.delete_random_item.append(f"{ff2ex} | {rf2ex}")

                # Processing EXCLUSION
                for ff2ex_item in random_model[rf2ex]:
                    if sorted(ff2ex_item.split(":")) == sorted(ff2ex.split(":")):
                        random_model[rf2ex].remove(ff2ex_item)
                # print(random_model)

            # ('methTitle', 'objClass', 'devcomp', 'isLmer', 'useScale', 'logLik', 'family', 'link', 'ngrps', 'coefficients', 'sigma', 'vcov', 'varcor', 'AICtab', 'call', 'residuals', 'fitMsgs', 'optinfo', 'corrSet')
            # ('optimizer', 'control', 'derivs', 'conv', 'feval', 'message', 'warnings', 'val')
        print("Looking for main effect(s)/interaction(s)...")

        self.summary = summary_model1_r
        anova_model1 = r2p(stats.anova(model1, type=3, ddf="Kenward-Roger"))
        self.anova = anova_model1


        final_rpt = f"For RT data, F test was conducted on the optimal model ({formula_str}). "

        for sig_items in anova_model1[anova_model1["Pr(>F)"] <= 0.05].index.tolist():
            if "ntercept" in sig_items:
                continue
            df_item = anova_model1[anova_model1["Pr(>F)"] <= 0.05].loc[sig_items]

            sig_items = sig_items.split(":")

            if len(sig_items) == 1:
                # print(f"Main effect {sig_items}")
                emmeans_result = emmeans.contrast(emmeans.emmeans(model1, sig_items[0]), "pairwise", adjust="bonferroni")
                emmeans_result_dict = extract_contrast(str(emmeans_result), factor_num=len(sig_items))[0]

                final_rpt += (f"The main effect of {sig_items[0].replace('_', ' ')} was significant "
                              f"(F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, "
                              f"p={df_item['Pr(>F)']:.3f}). Post-hoc analysis revealed that ")

                lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
                lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')

                if float(emmeans_result_dict['p.value']) <= 0.05:
                    final_rpt += f"RT for {self.trans_dict[lvl1]} condition " \
                                 f"was significantly {'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'}" \
                                 f" than that for {self.trans_dict[lvl2]} (" \
                                 f"β={emmeans_result_dict['estimate']}, " \
                                 f"SE={emmeans_result_dict['SE']}, " \
                                 f"df={emmeans_result_dict['df']}, " \
                                 f"t={emmeans_result_dict['t.ratio']}, " \
                                 f"p={float(emmeans_result_dict['p.value']):.3f}). "

                elif float(emmeans_result_dict['p.value']) > 0.05:
                    final_rpt += f"there was no significant difference between {self.trans_dict[lvl1]}" \
                                 f" and {self.trans_dict[lvl2]} in RT (" \
                                 f"β={emmeans_result_dict['estimate']}, " \
                                 f"SE={emmeans_result_dict['SE']}, " \
                                 f"df={emmeans_result_dict['df']}, " \
                                 f"t={emmeans_result_dict['t.ratio']}, " \
                                 f"p={float(emmeans_result_dict['p.value']):.3f}). "


            elif len(sig_items) == 2:
                # print(f"2-way Interaction {sig_items}")
                final_rpt += (f"The interaction between {sig_items[0].replace('_', ' ')} and {sig_items[1].replace('_', ' ')}"
                              f" was significant (F({int(df_item['NumDF'])},"
                              f"{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f})."
                              f" Simple effect analysis showed that, ")

                for i1, i2 in [(0, 1), (-1, -2)]:
                    emmeans_result = emmeans.contrast(emmeans.emmeans(model1, specs=sig_items[i1], by=sig_items[i2]), "pairwise", adjust="bonferroni")
                    for emmeans_result_dict in extract_contrast(str(emmeans_result), len(sig_items)):
                        lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
                        lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')
                        final_rpt += f"under {self.trans_dict[emmeans_result_dict['under_cond']]} condition, "
                        # print(emmeans_result_dict)
                        if float(emmeans_result_dict['p.value']) <= 0.05:
                            final_rpt += f"RT for {self.trans_dict[lvl1]} condition " \
                                         f"was significantly {'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'}" \
                                         f" than that for {self.trans_dict[lvl2]} condition (" \
                                         f"β={emmeans_result_dict['estimate']}, " \
                                         f"SE={emmeans_result_dict['SE']}, " \
                                         f"df={emmeans_result_dict['df']}, " \
                                         f"t={emmeans_result_dict['t.ratio']}, " \
                                         f"p={float(emmeans_result_dict['p.value']):.3f})"
                        else:
                            final_rpt += f"no significant difference was found between {self.trans_dict[lvl1]} condition " \
                                         f"and {self.trans_dict[lvl2]} condition in RT (" \
                                         f"β={emmeans_result_dict['estimate']}, " \
                                         f"SE={emmeans_result_dict['SE']}, " \
                                         f"df={emmeans_result_dict['df']}, " \
                                         f"t={emmeans_result_dict['t.ratio']}, " \
                                         f"p={float(emmeans_result_dict['p.value']):.3f})"
                        final_rpt += "; "

                final_rpt = final_rpt[:-2] + ". "

            elif len(sig_items) >= 3:
                final_rpt += "[Write here for 3-way analysis]"
                # print(f"3-way Interaction {sig_items} (under construction, use R for 3-way simple effect analysis please)")

        for sig_items in anova_model1[anova_model1["Pr(>F)"] > 0.05].index.tolist():
            if "ntercept" in sig_items:
                continue
            df_item = anova_model1[anova_model1["Pr(>F)"] > 0.05].loc[sig_items]
            sig_items = [sig_item.replace("_", " ") for sig_item in sig_items.split(":")]

            if len(sig_items) == 1:
                # print(f"Main effect {sig_items}")
                final_rpt += f"The main effect of {sig_items[0]} was not significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "

            elif len(sig_items) == 2:
                # print(f"2-way Interaction {sig_items}")
                final_rpt += f"The interaction between {' and '.join(sig_items)} was not significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "

            elif len(sig_items) >= 3:
                # print(f"3-way Interaction {sig_items} (under construction, use R for 3-way simple effect analysis please)")
                final_rpt += f"The interaction between {sig_items[0]}, {' and '.join(sig_items[1:])} was not significant (F({int(df_item['NumDF'])},{df_item['DenDF']:.3f})={df_item['F value']:.3f}, p={df_item['Pr(>F)']:.3f}). "

        print("Generating Reports...", end="")

        final_rpt = final_rpt.replace("=0.000", "<0.001")

        if isGoodModel:
            self.formula = formula_str

        self.report = final_rpt

        print("Completed")

class GeneralizedLinearMixedModel:

    def __init__(self):
        self.path = None
        self.report = None
        self.formula = None

        # Data exclusion
        self.isExcludeSD = True

        # Name variables
        self.dep_var = ""
        self.indep_var = []
        self.random_var = []

        # Generate report
        self.trans_dict = {}
        self.delete_random_item = []

    def read_data(self, path):
        """
        Read data from the data path
        :param path: data file path
        :return: None
        """


        print("Reading Data...", end="")
        # Read .csv self.data (it can accept formats like .xlsx, just change to pd.read_XXX)
        self.data = pd.read_csv(path, encoding="utf-8")
        print("Data collected!")

        return None

    def code_variables(self, code=None):
        """
        Change the values in each column with certain coding methods (customization)
        :param code: your coding dict
        :return: None
        """

        if code is None:
            code = {}
        print("Coding variables...", end="")

        if not code:
            for var in self.indep_var:
                code[var] = dict(zip(self.data[var].unique(), self.data[var].unique()))

        for var_name in code:
            # print(var_name, code[var_name])
            self.data[var_name] = self.data[var_name].apply(lambda x: code[var_name][x])

        for k in code:
            v = code[k]

            for cond in v:
                self.trans_dict[k + str(v[cond])] = cond
        print("Variables coded!")

        return None

    def exclude_trial_SD(self, target: str, subject: str, SD=2.5):
        """
        Attain only trials within 2.5 * SD for each participant/subject
        :param target: target column name
        :param subject: participant/subject column
        :param SD: default at 2.5
        :return: None
        """

        new_data = pd.DataFrame()
        for sub in list(set(self.data[subject])):
            sub_data = self.data[self.data[subject] == sub]

            # 计算'rt'列的均值和标准差
            mean_rt = sub_data[target].mean()
            std_rt = sub_data[target].std()

            # 根据均值和标准差筛选数据
            filtered_sub_data = sub_data[(sub_data[target] >= (mean_rt - SD * std_rt)) &
                                         (sub_data[target] <= (mean_rt + SD * std_rt))]
            # 将筛选后的数据追加到new_data中

            new_data = pd.concat([new_data, filtered_sub_data], ignore_index=True)

        self.data = new_data.copy()

        return None


    def fit(self, optimizer=[], prev_formula=""):

        dep_var = self.dep_var
        fixed_factor = self.indep_var
        random_factor = self.random_var


        fixed_str = " * ".join(fixed_factor)
        fixed_combo = []
        for i in range(len(fixed_factor), 0, -1):  # 从1开始，因为0会生成空集
            for combo in itertools.combinations(fixed_factor, i):
                combo = list(combo)
                combo.sort()
                fixed_combo.append(":".join(combo))
                # print(":".join(combo))

        if prev_formula:

            prev_formula = prev_formula.replace(" ", "")
            random_model = {kw[1]: kw[0].split("+") for kw in
                            [full_item.split(")")[0].split("|") for full_item in prev_formula.split("(1+")[1:]]}
            for key in random_factor:
                try:
                    random_model[key]
                except KeyError:
                    random_model[key] = []
        else:
            random_model = {key: fixed_combo[:] for key in random_factor}

        # Change pandas DataFrame into R language's self.data.frame
        r_data = p2r(self.data)

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
            print(f"\r[*] Runing FORMULA -> {formula_str}", end="")

            if not optimizer:
                model1 = lme4.glmer(formula, family="binomial", data=r_data)

            else:
                list_optCtrl = ro.ListVector([("maxfun", optimizer[1])])

                model1 = lmerTest.lmer(formula, REML=True, data=r_data, control=lme4.lmerControl(optimizer=optimizer[0], optCtrl=list_optCtrl))

            summary_model1_r = Matrix.summary(model1)

            summary_model1 = r2p(summary_model1_r)

            try:
                isWarning = eval(str(summary_model1["optinfo"]["conv"]['lme4']["messages"]).strip("o"))
            except KeyError:
                isWarning = False
            except SyntaxError:
                isWarning = True

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
                print(f"\r[√] FORMULA -> {formula_str}")
                break
            else:
                print(f"\r[×] FORMULA -> {formula_str}")

                if not any(random_model.values()):
                    break

                rf2ex = df.loc[df[2].idxmin(0)][0]
                ff2ex = df.loc[df[2].idxmin(0)][1]

                self.delete_random_item.append(f"{ff2ex} | {rf2ex}")

                # Processing EXCLUSION
                for ff2ex_item in random_model[rf2ex]:
                    if sorted(ff2ex_item.split(":")) == sorted(ff2ex.split(":")):
                        random_model[rf2ex].remove(ff2ex_item)

        self.summary = summary_model1_r
        anova_model1 = r2p(car.Anova(model1, type=3, test="Chisq"))
        self.anova = anova_model1

        print("Looking for main effect(s)/interaction(s)...")

        final_rpt = f"For ACC data, Wald chi-square test was conducted on the optimal model ({formula_str}). "
        # print(anova_model1)
        for sig_items in anova_model1[anova_model1["Pr(>Chisq)"] <= 0.05].index.tolist():
            if "ntercept" in sig_items:
                continue

            df_item = anova_model1[anova_model1["Pr(>Chisq)"] <= 0.05].loc[sig_items]
            sig_items = sig_items.split(":")

            if len(sig_items) == 1:
                # print(f"Main Effect {sig_items}")
                emmeans_result = emmeans.contrast(emmeans.emmeans(model1, sig_items[0]), "pairwise", adjust="bonferroni")
                emmeans_result_dict = extract_contrast(str(emmeans_result))
                # print(emmeans_result)
                lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
                lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')

                final_rpt += (f"The main effect of {sig_items[0].replace('_', ' ')} was"
                              f" significant (\chi^2({int(df_item['Df'])})={df_item['Chisq']:.3f},"
                              f" p={df_item['Pr(>Chisq)']:.3f}). Post-hoc analysis revealed that ")

                if float(emmeans_result_dict['p.value']) <= 0.05:
                    final_rpt += f"ACC for {self.trans_dict[lvl1]} " \
                                 f"was significantly {'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'}" \
                                 f" than that for {self.trans_dict[lvl2]} (" \
                                 f"β={emmeans_result_dict['estimate']}, " \
                                 f"SE={emmeans_result_dict['SE']}, " \
                                 f"z={emmeans_result_dict['z.ratio']}, " \
                                 f"p={float(emmeans_result_dict['p.value']):.3f}). "
                else:
                    final_rpt += f"there was no significant difference between {self.trans_dict[lvl1]}" \
                                 f" and {self.trans_dict[lvl2]} in ACC (" \
                                 f"β={emmeans_result_dict['estimate']}, " \
                                 f"SE={emmeans_result_dict['SE']}, " \
                                 f"z={emmeans_result_dict['z.ratio']}, " \
                                 f"p={float(emmeans_result_dict['p.value']):.3f}). "

            elif len(sig_items) == 2:
                # print(f"2-way Interaction {sig_items}")
                lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
                lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')

                final_rpt += (f"The interaction between {sig_items[0].replace('_', ' ')} and"
                              f" {sig_items[1].replace('_', ' ')} was significant"
                              f" (\chi^2({int(df_item['Df'])})={df_item['Chisq']:.3f}, "
                              f"p={df_item['Pr(>Chisq)']:.3f}). Simple effect analysis showed that, ")

                for i1, i2 in [(0, 1), (-1, -2)]:
                    emmeans_result = emmeans.contrast(emmeans.emmeans(model1, specs=sig_items[i1], by=sig_items[i2]),
                                                      "pairwise", adjust="bonferroni")

                    for emmeans_result_dict in extract_contrast(str(emmeans_result), 1):

                        final_rpt += f"under {emmeans_result_dict['under_cond']} condition, "

                        if float(emmeans_result_dict['p.value']) <= 0.05:
                            final_rpt += f"ACC for {self.trans_dict[lvl1]} condition " \
                                         f"was significantly {'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'}" \
                                         f" than that for {self.trans_dict[lvl2]} condition (" \
                                         f"β={emmeans_result_dict['estimate']}, " \
                                         f"SE={emmeans_result_dict['SE']}, " \
                                         f"z={emmeans_result_dict['z.ratio']}, " \
                                         f"p={float(emmeans_result_dict['p.value']):.3f})"
                        else:
                            final_rpt += f"no significant difference was found between {self.trans_dict[lvl1]}" \
                                         f" condition and {self.trans_dict[lvl2]} condition in ACC (" \
                                         f"β={emmeans_result_dict['estimate']}, " \
                                         f"SE={emmeans_result_dict['SE']}, " \
                                         f"z={emmeans_result_dict['z.ratio']}, " \
                                         f"p={float(emmeans_result_dict['p.value']):.3f})"

                        final_rpt += "; "

                final_rpt = final_rpt[:-2] + ". "

            elif len(sig_items) >= 3:
                final_rpt += "[Write here for 3-way analysis]"


        for sig_items in anova_model1[anova_model1["Pr(>Chisq)"] > 0.05].index.tolist():
            if "ntercept" in sig_items:
                continue
            df_item = anova_model1[anova_model1["Pr(>Chisq)"] > 0.05].loc[sig_items]
            sig_items = [sig_item.replace("_", " ") for sig_item in sig_items.split(":")]

            if len(sig_items) == 1:
                # print(f"Main effect {sig_items}")
                final_rpt += (f"The main effect of {sig_items[0]} was not"
                              f" significant (\chi^2({int(df_item['Df'])})={df_item['Chisq']:.3f},"
                              f" p={df_item['Pr(>Chisq)']:.3f}). ")

            elif len(sig_items) == 2:
                # print(f"2-way Interaction {sig_items}")
                final_rpt += (f"The interaction between {sig_items[0]}"
                              f"and {sig_items[1].replace('_', ' ')} was not significant"
                              f" (\chi^2({int(df_item['Df'])})={df_item['Chisq']:.3f},"
                              f" p={df_item['Pr(>Chisq)']:.3f}). ")

            elif len(sig_items) >= 3:
                # print(f"3-way Interaction {sig_items} (under construction, use R for 3-way simple effect analysis please)")
                final_rpt += (f"The interaction between {sig_items[0]}, "
                              f"{sig_items[1].replace('_', ' ')} and {sig_items[2].replace('_', ' ')}"
                              f" was not significant (\chi^2({int(df_item['Df'])})={df_item['Chisq']:.3f},"
                              f" p={df_item['Pr(>Chisq)']:.3f}). ")



        print("Generating Reports...", end="")

        final_rpt = final_rpt.replace("=0.000", "<0.001")

        if isGoodModel:
            self.formula = formula_str

        self.report = final_rpt

        print("Completed")

