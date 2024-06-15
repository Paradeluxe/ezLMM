import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula, numpy2ri
from rpy2.robjects.packages import importr
import itertools
from rpy2.robjects.conversion import localconverter
import numpy as np

# 导入R的库
lmerTest = importr('lmerTest')
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
    data = data[(data['exp_type'] == "exp2") & (data['ifanimal'] == True)]

    # if dep_var == "rt":
    data = data[data['ifcorr'] == 1]

    data['rt'] = data['rt'] * 1000

    # 定义因子变量
    data['Tpriming'] = -0.5 * (data['priming'] == "priming") + 0.5 * (data['priming'] != "priming")
    data['Tsyl'] = -0.5 * (data['syl'] == 2) + 0.5 * (data['syl'] != 2)
    # data['Texp_type'] = -0.5 * (data['exp_type'] == "exp1") + 0.5 * (data['exp_type'] != "exp1")

    # 将pandas DataFrame转换为R的data.frame
    with (ro.default_converter + pandas2ri.converter).context():
        r_data = ro.conversion.get_conversion().py2rpy(data)

    # Create Formula and do the For loop

    # Construct
    # 定义R的公式
    fixed_factor = ["Tpriming", "Tsyl"]
    random_factor = ["sub", "word"]
    fixed_str = " * ".join(fixed_factor)

    fixed_combo = []

    # 使用itertools.combinations生成所有非空组合
    # 从1开始，因为0会生成空集
    for i in range(len(fixed_factor), 0, -1):
        for combo in itertools.combinations(fixed_factor, i):
            fixed_combo.append(":".join(combo))

    random_model = {key: fixed_combo for key in random_factor}


    while True:
        random_str = ""
        for key in random_model:
            random_str += "(1 + " + " + ".join(random_model[key]) + f" | {key}) + "
        random_str = random_str.rstrip(" + ")

        formula = f"rt ~ {fixed_str} + {random_str}"
        Formula = Formula(formula)
        print(f"Running FORMULA: {formula}")

        # 使用lmer函数拟合模型
        model1 = lmerTest.lmer(Formula, REML=True, data=r_data)

        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            summary_model1 = ro.conversion.get_conversion().rpy2py(Matrix.summary(model1))

        isSingular = summary_model1["optinfo"]["conv"]['lme4']["messages"][0] == "boundary (singular) fit: see help('isSingular')"

        # print(type(nlme.VarCorr(model1)))
        # with (ro.default_converter + pandas2ri.converter).context():
        #     varcorr_model1 = ro.conversion.get_conversion().rpy2py(nlme.VarCorr(model1))


        # Check if corr < 0.90 and exclude the least std
        random_table = []

        # 将字符串按行分割
        lines = str(nlme.VarCorr(model1)).strip().split('\n')
        for line in lines[1:-1]:  # Exclude the 1st and last line
            # Check if the 2nd element is number
            elements = line.split()
            if elements[1].split(".")[0].isnumeric():
                elements = [Groups] + elements
            else:
                Groups = elements[0]
            elements = [float(e) if e.split(".")[0].strip("-").isnumeric() else e for e in elements]

            random_table.append(elements)
            print(elements)

        df = pd.DataFrame(random_table)

        # Check if there is any corr item that is >= 0.90
        all_corrs = np.array(df.iloc[:, 3:].dropna()).flatten().tolist()
        isTooLargeCorr = any(corr >= .9 for corr in all_corrs)

        # if isTooLargeCorr:
        #     print("Find corr >= 0.90")

        isGoodModel = not isSingular and not isTooLargeCorr

        if isGoodModel:
            break
        else:
            rf2ex = df.loc[df[2].idxmin(0)][0]
            ff2ex = df.loc[df[2].idxmin(0)][1]

            print(f"Exclude Random Factor \"{rf2ex}\", and Fixed item \"{ff2ex}\"")

            # Processing EXCLUSION
            random_model[rf2ex].remove(ff2ex)
            print(random_model)

        # ('methTitle', 'objClass', 'devcomp', 'isLmer', 'useScale', 'logLik', 'family', 'link', 'ngrps', 'coefficients', 'sigma', 'vcov', 'varcor', 'AICtab', 'call', 'residuals', 'fitMsgs', 'optinfo', 'corrSet')
        # ('optimizer', 'control', 'derivs', 'conv', 'feval', 'message', 'warnings', 'val')


    anova_model1 = stats.anova(model1, type=3, ddf="Kenward-Roger")
    # print(anova_model1.colnames)
    # print(anova_model1.rownames)

    # anova_model1 change format
    with (ro.default_converter + pandas2ri.converter).context():
        anova_model1 = ro.conversion.get_conversion().rpy2py(anova_model1)

    # print(anova_model1["Pr(>F)"])
    print(anova_model1)
    # print(summary_model1)

    # 进行emmeans分析
    # emmeans_result = emmeans.emmeans(model1, pairwise ~ Tsyl)

    # 打印结果
    # print(emmeans_result)
