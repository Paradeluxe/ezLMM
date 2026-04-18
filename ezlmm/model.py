"""Linear and generalized linear mixed models."""

import itertools
import os

import numpy as np
import pandas as pd

os.environ['RPY2_CFFI_MODE'] = 'ABI'

import warnings
warnings.filterwarnings("ignore")

from ezlmm.utils import (
    lmerTest, lme4, stats, Matrix, car, nlme,
    Formula, r_object, r2p, p2r
)
from ezlmm.data import DataLoader
from ezlmm.report import write_simple_effect_lmm, write_simple_effect_glmm


pd.set_option("display.max_columns", None)


class LinearMixedModel(DataLoader):
    """
    Fit a linear mixed model with automatic random-structure search.

    Inherits data methods from ``DataLoader``.
    """

    def __init__(self):
        super().__init__()
        self.report = None
        self.formula = None
        self.model_r = None
        self.summary_r = None
        self.summary = None
        self.anova = None

    def fit(self, optimizer=None, prev_formula: str = "", report: bool = True):
        """
        Fit the model, searching for the optimal random structure.

        Parameters
        ----------
        optimizer : list or None
            [optimizer_name, maxfun], e.g. ["bobyqa", 1e5].
        prev_formula : str
            R formula string from a previous fit to continue from.
        report : bool
            If True, generate the APA report after fitting.

        Side-effects
        ------------
        Sets ``self.model_r``, ``self.summary_r``, ``self.summary``,
        ``self.anova``, ``self.formula``, ``self.report``.
        """
        if optimizer is None:
            optimizer = []

        dep_var = self.dep_var
        fixed_factor = self.indep_var
        random_factor = self.random_var

        fixed_str = " * ".join(fixed_factor)
        fixed_combo = []
        for i in range(len(fixed_factor), 0, -1):
            for combo in itertools.combinations(fixed_factor, i):
                combo = list(combo)
                combo.sort()
                fixed_combo.append(":".join(combo))

        if prev_formula:
            prev_formula = prev_formula.replace(" ", "")
            random_model = {
                kw[1]: kw[0].split("+")
                for kw in [
                    full_item.split(")")[0].split("|")
                    for full_item in prev_formula.split("(1+")[1:]
                ]
            }
            for key in random_factor:
                random_model.setdefault(key, [])
        else:
            random_model = {key: fixed_combo[:] for key in random_factor}

        r_data = p2r(self.data)

        while True:
            random_str = ""
            for key in random_model:
                if not random_model[key]:
                    random_str += f"(1 | {key}) + "
                else:
                    random_str += "(1 + " + " + ".join(random_model[key]) + f" | {key}) + "
            random_str = random_str.rstrip(" + ")

            formula_str = f"{dep_var} ~ {fixed_str} + {random_str}"
            print(f"\r[*] Running FORMULA -> {formula_str}", end="")

            if not optimizer:
                model1 = lmerTest.lmer(Formula(formula_str), REML=True, data=r_data)
            else:
                list_optCtrl = ro.ListVector([("maxfun", optimizer[1])])
                model1 = lmerTest.lmer(
                    Formula(formula_str), REML=True, data=r_data,
                    control=lme4.lmerControl(optimizer=optimizer[0], optCtrl=list_optCtrl)
                )

            summary_model1_r = Matrix.summary(model1)
            summary_model1 = r_object(summary_model1_r)
            print()
            print("-")
            try:
                print(summary_model1["optinfo"]["conv"]["lme4"]["messages"])
            except Exception:
                pass
            print("-")

            # Convergence check
            try:
                isWarning = summary_model1["optinfo"]["conv"]["lme4"]["messages"]
            except (KeyError, TypeError):
                isWarning = False

            # Parse VarCorr table
            random_table = []
            corrs_supp = []
            lines = str(nlme.VarCorr(model1)).strip().split('\n')
            for line in lines[1:]:
                elements = line.strip().split()
                if not elements:
                    continue
                if elements[0].split(".")[0].strip("-").isnumeric():
                    corrs_supp.extend([
                        float(e) if e.split(".")[0].strip("-").isnumeric() else e
                        for e in elements
                    ])
                    continue
                if elements[1].strip("-").split(".")[0].isnumeric():
                    elements = [Groups] + elements
                else:
                    Groups = elements[0]
                elements = [
                    float(e) if e.split(".")[0].strip("-").isnumeric() else e
                    for e in elements
                ]
                if elements[1] == "Residual":
                    break
                random_table.append(elements)

            df = pd.DataFrame(random_table)
            df = df[df[1] != '(Intercept)']

            all_corrs = np.array(df.iloc[:, 3:].dropna(how="all")).flatten().tolist()
            all_corrs = [c for c in all_corrs if isinstance(c, (int, float))]
            all_corrs.extend(corrs_supp)
            isTooLargeCorr = any(corr >= .9 for corr in all_corrs)

            isGoodModel = not isWarning and not isTooLargeCorr

            if isGoodModel:
                print(f"\r[√] FORMULA -> {formula_str}")
                break

            print(f"\r[×] FORMULA -> {formula_str}")
            if not any(random_model.values()):
                break

            # Drop the item with the smallest variance
            rf2ex = df.loc[df[2].idxmin(0)][0]
            ff2ex = df.loc[df[2].idxmin(0)][1]
            self.delete_random_item.append(f"{ff2ex} | {rf2ex}")

            for ff2ex_item in random_model[rf2ex]:
                target_items = sorted(ff2ex.split(":"))
                this_items = sorted(ff2ex_item.split(":"))
                if len(target_items) != len(this_items):
                    continue
                target_items = [
                    target_items[ti][:len(this_items[ti])]
                    for ti in range(len(target_items))
                ]
                if this_items == target_items:
                    random_model[rf2ex].remove(ff2ex_item)

        print("Looking for main effect(s)/interaction(s)...", end="")

        self.model_r = model1
        self.summary_r = summary_model1_r
        self.summary = summary_model1
        self.anova = r2p(stats.anova(model1, type=3, ddf="Kenward-Roger"))
        print("Found!")

        if not report:
            return

        print("Generating Reports...", end="")
        if isGoodModel:
            self.formula = formula_str
        self.report = write_simple_effect_lmm(
            self.dep_var, self.trans_dict, formula_str, self.anova, self.model_r
        )
        print("Completed")


class GeneralizedLinearMixedModel(DataLoader):
    """
    Fit a generalized linear mixed model with automatic random-structure search.

    Inherits data methods from ``DataLoader``.
    """

    def __init__(self):
        super().__init__()
        self.report = None
        self.formula = None
        self.model_r = None
        self.summary_r = None
        self.summary = None
        self.anova = None
        self.family = None

    def fit(self, optimizer=None, prev_formula: str = "", family=None, report: bool = True):
        """
        Fit a GLMM, searching for the optimal random structure.

        Parameters
        ----------
        optimizer : list or None
            [optimizer_name, maxfun].
        prev_formula : str
            R formula string from a previous fit.
        family : str
            GLMM family, e.g. "binomial". Default: "binomial".
        report : bool
            If True, generate the APA report (note: not yet wired to write_simple_effect).

        Side-effects
        ------------
        Sets ``self.model_r``, ``self.summary_r``, ``self.summary``,
        ``self.anova``, ``self.formula``.
        """
        if optimizer is None:
            optimizer = []
        if family is None:
            family = "binomial"

        self.family = family

        dep_var = self.dep_var
        fixed_factor = self.indep_var
        random_factor = self.random_var

        fixed_str = " * ".join(fixed_factor)
        fixed_combo = []
        for i in range(len(fixed_factor), 0, -1):
            for combo in itertools.combinations(fixed_factor, i):
                combo = list(combo)
                combo.sort()
                fixed_combo.append(":".join(combo))

        if prev_formula:
            prev_formula = prev_formula.replace(" ", "")
            random_model = {
                kw[1]: kw[0].split("+")
                for kw in [
                    full_item.split(")")[0].split("|")
                    for full_item in prev_formula.split("(1+")[1:]
                ]
            }
            for key in random_factor:
                random_model.setdefault(key, [])
        else:
            random_model = {key: fixed_combo[:] for key in random_factor}

        r_data = p2r(self.data)

        while True:
            random_str = ""
            for key in random_model:
                if not random_model[key]:
                    random_str += f"(1 | {key}) + "
                else:
                    random_str += "(1 + " + " + ".join(random_model[key]) + f" | {key}) + "
            random_str = random_str.rstrip(" + ")

            formula_str = f"{dep_var} ~ {fixed_str} + {random_str}"
            print(f"\r[*] Running FORMULA -> {formula_str}", end="")

            if not optimizer:
                model1 = lme4.glmer(Formula(formula_str), family=self.family, data=r_data)
            else:
                list_optCtrl = ro.ListVector([("maxfun", optimizer[1])])
                model1 = lme4.glmer(
                    Formula(formula_str), data=r_data,
                    control=lme4.lmerControl(optimizer=optimizer[0], optCtrl=list_optCtrl)
                )

            summary_model1_r = Matrix.summary(model1)
            summary_model1 = r_object(summary_model1_r)

            try:
                isWarning = summary_model1["optinfo"]["conv"]["lme4"]["messages"]
            except (KeyError, TypeError):
                isWarning = False

            # Parse VarCorr table
            random_table = []
            corrs_supp = []
            lines = str(nlme.VarCorr(model1)).strip().split('\n')
            for line in lines[1:]:
                elements = line.strip().split()
                if not elements:
                    continue
                if elements[0].split(".")[0].strip("-").isnumeric():
                    corrs_supp.extend([
                        float(e) if e.split(".")[0].strip("-").isnumeric() else e
                        for e in elements
                    ])
                    continue
                if elements[1].strip("-").split(".")[0].isnumeric():
                    elements = [Groups] + elements
                else:
                    Groups = elements[0]
                elements = [
                    float(e) if e.split(".")[0].strip("-").isnumeric() else e
                    for e in elements
                ]
                if elements[1] == "Residual":
                    break
                random_table.append(elements)

            df = pd.DataFrame(random_table)
            df = df[df[1] != '(Intercept)']

            all_corrs = np.array(df.iloc[:, 3:].dropna(how="all")).flatten().tolist()
            all_corrs = [c for c in all_corrs if isinstance(c, (int, float))]
            all_corrs.extend(corrs_supp)
            isTooLargeCorr = any(corr >= .9 for corr in all_corrs)

            isGoodModel = not isWarning and not isTooLargeCorr

            if isGoodModel:
                print(f"\r[√] FORMULA -> {formula_str}")
                break

            print(f"\r[×] FORMULA -> {formula_str}")
            if not any(random_model.values()):
                break

            rf2ex = df.loc[df[2].idxmin(0)][0]
            ff2ex = df.loc[df[2].idxmin(0)][1]
            self.delete_random_item.append(f"{ff2ex} | {rf2ex}")

            for ff2ex_item in random_model[rf2ex]:
                target_items = sorted(ff2ex.split(":"))
                this_items = sorted(ff2ex_item.split(":"))
                if len(target_items) != len(this_items):
                    continue
                target_items = [
                    target_items[ti][:len(this_items[ti])]
                    for ti in range(len(target_items))
                ]
                if this_items == target_items:
                    random_model[rf2ex].remove(ff2ex_item)

        print("Looking for main effect(s)/interaction(s)...", end="")

        self.model_r = model1
        self.summary_r = summary_model1_r
        self.summary = summary_model1
        self.anova = r2p(car.Anova(model1, type=3, test="Chisq"))
        print("Found!")

        if not report:
            return

        print("Generating Reports...", end="")
        if isGoodModel:
            self.formula = formula_str
        # NOTE: GLMM report generation is not yet wired — see report.py
        print("Completed")
