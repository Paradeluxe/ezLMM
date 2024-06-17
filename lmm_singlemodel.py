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
    # data = data[(data['exp_type'] == "exp2") & (data['ifanimal'] == True)]
    # data = data[(data['ifanimal'] == True)]

    # if dep_var == "rt":
    # data = data[data['ifcorr'] == 1]
    data['rt'] = data['rt'] * 1000
    # 定义因子变量
    data['Tpriming'] = -0.5 * (data['priming'] == "priming") + 0.5 * (data['priming'] != "priming")
    data['Tsyl'] = -0.5 * (data['syl'] == 2) + 0.5 * (data['syl'] != 2)
    data['Texp_type'] = -0.5 * (data['exp_type'] == "exp1") + 0.5 * (data['exp_type'] != "exp1")

    # 将pandas DataFrame转换为R的data.frame
    with (ro.default_converter + pandas2ri.converter).context():
        r_data = ro.conversion.get_conversion().py2rpy(data)

    # Create Formula and do the For loop


    formula_str = f"rt ~ Tpriming * Tsyl * Texp_type + (1 + Tsyl:Texp_type + Texp_type | sub) + (1 + Tsyl | word)"
    formula = Formula(formula_str)
    print(f"Running FORMULA: {formula_str}")

    # 使用lmer函数拟合模型
    model1 = lmerTest.lmer(formula, REML=True, data=r_data)
    summary_model1_r = Matrix.summary(model1)
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        summary_model1 = ro.conversion.get_conversion().rpy2py(summary_model1_r)
    exit()
    print(summary_model1_r)
    anova_model1 = car.Anova(model1, type=3, test="Chisq")
    # print(anova_model1.colnames)
    # print(anova_model1.rownames)
    # anova_model1 change format
    with (ro.default_converter + pandas2ri.converter).context():
        anova_model1 = ro.conversion.get_conversion().rpy2py(anova_model1)

    # print(anova_model1["Pr(>F)"])
    print(anova_model1)

    print("-------------------------------------------------------")
    print("SCRIPT End √ | Ignore \"R[write to console]\" down below, as it is an automatic callback")
    print("-------------------------------------------------------")
    # print(summary_model1)

    # 进行emmeans分析
    # emmeans_result = emmeans.emmeans(model1, pairwise ~ Tsyl)

    # 打印结果
    # print(emmeans_result)
