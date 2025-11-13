from ezlmm import LinearMixedModel, GeneralizedLinearMixedModel


lmm = GeneralizedLinearMixedModel()  # LinearMixedModel()
# glmm = GeneralizedLinearMixedModel()

lmm.read_data(r".\subject_all.csv")  # Change with your own data path


# Dependant variable
lmm.dep_var = "acc"
# Independent variable(s)
lmm.indep_var = ["proficiency", "word_type"]
# Random factor(s)
lmm.random_var = ["subject", "word"]


# [Optional]
# --------------------
lmm.exclude_trial_SD(target="rt", subject="subject", SD=2.5)
# lmm.data = lmm.data[lmm.data['acc'] == 1]

# Descriptive statistics
descriptive_stats = lmm.descriptive_stats()
print(descriptive_stats)

# Code your variables right before fitting
# lmm.code_variables({
#     "syllable_number": {"disyllabic": -0.5, "trisyllabic": 0.5},
#     "speech_isochrony": {"averaged": -0.5, "original": 0.5},
#     "priming_effect": {"inconsistent": -0.5, "consistent": 0.5}
# })
# --------------------

# Fitting the model until it is converged
lmm.fit()

# Print output
# print(lmm.report)  # plain text for your paper
print(lmm.summary)  # model fitting
print(lmm.anova)  # anova on fixed model