from ezlmm import LinearMixedModel
import pandas as pd

# 创建一个简单的测试数据
test_data = pd.DataFrame({
    'subject': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
    'word': ['W1', 'W2', 'W1', 'W2', 'W1', 'W2'],
    'proficiency': ['high', 'high', 'low', 'low', 'high', 'low'],
    'word_type': ['noun', 'verb', 'noun', 'verb', 'noun', 'verb'],
    'rt': [450, 520, 480, 550, 460, 530],
    'acc': [1, 1, 1, 1, 1, 1]
})

# 保存测试数据
test_data.to_csv('test_data.csv', index=False)

# 创建模型实例
lmm = LinearMixedModel()

# 读取测试数据
lmm.read_data('test_data.csv')

# 设置变量
lmm.dep_var = "rt"
lmm.indep_var = ["proficiency", "word_type"]
lmm.random_var = ["subject", "word"]

print("开始拟合模型...")
try:
    lmm.fit()
    print("模型拟合成功！")
except Exception as e:
    print(f"模型拟合失败，错误信息: {e}")
    print(f"错误类型: {type(e)}")
    import traceback
    print("详细错误追踪:")
    print(traceback.format_exc())