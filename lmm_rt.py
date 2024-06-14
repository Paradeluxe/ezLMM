import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
import itertools

# 导入R的库
lmerTest = importr('lmerTest')
emmeans = importr('emmeans')
carData = importr('carData')
car = importr('car')
stats = importr("stats")
Matrix = importr("Matrix")



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

    fixed_combo = []

    # 使用itertools.combinations生成所有非空组合
    # 从1开始，因为0会生成空集
    for i in range(len(fixed_factor), 0, -1):
        for combo in itertools.combinations(fixed_factor, i):
            fixed_combo.append(":".join(combo))

    fixed_model = " * ".join(fixed_factor)
    random_model = " + ".join([f"(1+|{rf})" for rf in random_factor])

    formulas = [Formula(f"rt ~ {fixed_model} + {random_model}") for fc in fixed_combo]



    for formula in formulas:
        print(f"Running FORMULA: {formula}")

        # 使用lmer函数拟合模型
        model1 = lmerTest.lmer(formula, REML=True, data=r_data)

        summary_model1 = Matrix.summary(model1)
        print(summary_model1)
        with (ro.default_converter + pandas2ri.converter).context():
            summary_model1 = ro.conversion.get_conversion().rpy2py(summary_model1)



        isSingular = summary_model1["optinfo"]["conv"]['lme4']["messages"][0] == "boundary (singular) fit: see help('isSingular')"

        isGoodModel = not isSingular

        if isGoodModel:
            break
        else:
            continue


    # ('methTitle', 'objClass', 'devcomp', 'isLmer', 'useScale', 'logLik', 'family', 'link', 'ngrps', 'coefficients', 'sigma', 'vcov', 'varcor', 'AICtab', 'call', 'residuals', 'fitMsgs', 'optinfo', 'corrSet')
    # ('optimizer', 'control', 'derivs', 'conv', 'feval', 'message', 'warnings', 'val')

    anova_model1 = stats.anova(model1, type=3, ddf="Kenward-Roger")
    # print(anova_model1.colnames)
    # print(anova_model1.rownames)

    # anova_model1 change format
    with (ro.default_converter + pandas2ri.converter).context():
        anova_model1 = ro.conversion.rpy2py(anova_model1)

    print(anova_model1["Pr(>F)"])


    # print(summary_model1)

    # 进行emmeans分析
    # emmeans_result = emmeans.emmeans(model1, pairwise ~ Tsyl)

    # 打印结果
    # print(emmeans_result)
