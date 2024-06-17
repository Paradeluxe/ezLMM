import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula, numpy2ri
from rpy2.robjects.packages import importr
import itertools
from rpy2.robjects.conversion import localconverter
import numpy as np

# 导入R的库
emmeans = importr('emmeans')
carData = importr('carData')
car = importr('car')
stats = importr("stats")
Matrix = importr("Matrix")
lme4 = importr("lme4")
nlme = importr("nlme")


if __name__ == "__main__":

    # 读取CSV文件
    data = pd.read_csv("Data_Experiment.csv", encoding="utf-8")

    # 在Python中进行数据子集选择
    # data = data[(data['exp_type'] == "exp2") & (data['ifanimal'] == True)]
    # data = data[(data['ifanimal'] == True)]

    # if dep_var == "rt":
    # data = data[data['ifcorr'] == 1]

    # 定义因子变量
    data['Tpriming'] = -0.5 * (data['priming'] == "priming") + 0.5 * (data['priming'] != "priming")
    data['Tsyl'] = -0.5 * (data['syl'] == 2) + 0.5 * (data['syl'] != 2)
    data['Texp_type'] = -0.5 * (data['exp_type'] == "exp1") + 0.5 * (data['exp_type'] != "exp1")

    # 将pandas DataFrame转换为R的data.frame
    with (ro.default_converter + pandas2ri.converter).context():
        r_data = ro.conversion.get_conversion().py2rpy(data)

    # Create Formula and do the For loop

    # Construct
    # 定义R的公式
    dep_var = "ifcorr"
    fixed_factor = ["Tpriming", "Tsyl", "Texp_type"]
    random_factor = ["sub", "word"]
    fixed_str = " * ".join(fixed_factor)

    fixed_combo = []

    # 使用itertools.combinations生成所有非空组合
    # 从1开始，因为0会生成空集
    for i in range(len(fixed_factor), 0, -1):
        for combo in itertools.combinations(fixed_factor, i):
            fixed_combo.append(":".join(combo))

    random_model = {key: fixed_combo[:] for key in random_factor}


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
        model1 = lme4.glmer(formula, family="binomial", data=r_data)
        summary_model1_r = Matrix.summary(model1)
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            summary_model1 = ro.conversion.get_conversion().rpy2py(summary_model1_r)
        try:
            isWarning = list(summary_model1["optinfo"]["conv"]['lme4']["messages"])
        except KeyError:
            isWarning = False

        # Transform random table to DataFrame format
        random_table = []
        lines = str(nlme.VarCorr(model1)).strip().split('\n')
        corrs_supp = []
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

        isTooLargeCorr = any(corr >= .9 for corr in all_corrs)

        isGoodModel = not isWarning and not isTooLargeCorr

        if isGoodModel:
            break
        else:
            rf2ex = df.loc[df[2].idxmin(0)][0]
            ff2ex = df.loc[df[2].idxmin(0)][1]

            print(f"Exclude random model item: {ff2ex} | {rf2ex}")

            # Processing EXCLUSION
            random_model[rf2ex].remove(ff2ex)
            # print(random_model)
            print("---\n---")

        if not any(random_model.values()):
            print("Loop end, nothing found.")
            break


        # ('methTitle', 'objClass', 'devcomp', 'isLmer', 'useScale', 'logLik', 'family', 'link', 'ngrps', 'coefficients', 'sigma', 'vcov', 'varcor', 'AICtab', 'call', 'residuals', 'fitMsgs', 'optinfo', 'corrSet')
        # ('optimizer', 'control', 'derivs', 'conv', 'feval', 'message', 'warnings', 'val')


    print(summary_model1_r)

    anova_model1 = car.Anova(model1, type=3, test="Chisq")
    print(anova_model1)
    with (ro.default_converter + pandas2ri.converter).context():
        anova_model1 = ro.conversion.get_conversion().rpy2py(anova_model1)

    print("-------------------------------------------------------")
    print("SCRIPT End √ | Ignore \"R[write to console]\" down below, as it is an automatic callback")
    print("-------------------------------------------------------")
    # print(summary_model1)

    # 进行emmeans分析
    # emmeans_result = emmeans.emmeans(model1, pairwise ~ Tsyl)

    # 打印结果
    # print(emmeans_result)
