import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr

# 导入R的库
lmerTest = importr('lmerTest')
emmeans = importr('emmeans')
carData = importr('carData')
car = importr('car')
stats = importr("stats")
Matrix = importr("Matrix")



if __name__ == "__main__":


    exptype = "exp2"
    dep_var = "rt"  # ifcorr rt

    # 读取CSV文件
    data = pd.read_csv(r"C:\Users\Lenovo-PC\Desktop\linguistic rhythm\Data_Experiment.csv", encoding="utf-8")

    # 在Python中进行数据子集选择
    data = data[(data['exp_type'] == exptype) & (data['ifanimal'] == True)]

    if dep_var == "rt":
        data = data[data['ifcorr'] == 1]

    data['rt'] = data['rt'] * 1000

    # 定义因子变量
    data['Tpriming'] = -0.5 * (data['priming'] == "priming") + 0.5
    data['Tsyl'] = -0.5 * (data['syl'] == 2) + 0.5
    data['Texp_type'] = -0.5 * (data['exp_type'] == "exp1") + 0.5

    # 将pandas DataFrame转换为R的data.frame
    # r_data = pandas2ri.py2ri(data)
    with (ro.default_converter + pandas2ri.converter).context():
        r_data = ro.conversion.get_conversion().py2rpy(data)

    # Create Formula and do the For loop


    # 定义R的公式
    formula = Formula("rt ~ Tpriming * Tsyl + (1 | sub) + (1 | word)")

    # 使用lmer函数拟合模型
    if dep_var == "rt":
        model1 = lmerTest.lmer(formula, REML=True, data=r_data)
        # print(model1)
        anova_model1 = stats.anova(model1, type=3, ddf="Kenward-Roger")
        print(anova_model1)
        print(type(anova_model1))
        summary_model1 = Matrix.summary(model1)
        print(summary_model1)
        # ro.r('anova(model1, type=3, ddf="Kenward-Roger")')


    else:
        # 这里需要根据你的实际模型进行调整
        pass

    # 进行emmeans分析
    # emmeans_result = emmeans.emmeans(model1, pairwise ~ Tsyl)

    # 打印结果
    # print(emmeans_result)