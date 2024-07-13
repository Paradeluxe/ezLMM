# ezLMM: A Python Script for Linear Mixed Model using rpy2

## Install package
```bash
pip install ezlmm
```

## How to use this package
### Need to know
- *ezlmm* executes from ***full model*** to ***null model*** for random model.
- *ezlmm* won't change your fixed model. (i.e., It stays the same.)
- *ezlmm* currently considers ***random intercept*** as default for each random effect
### Import ezlmm
```python
# In my research area:
# continuous variables (e.g., reaction time) use lmm
# categorical variables (e.g., accuracy) use glmm

from ezlmm import LinearMixedModel, GeneralizedLinearMixedModel
```

### Create instance
For linear mixed model,

```python
lmm = LinearMixedModel()
```

For generalized linear mixed model,
```python
glmm = GeneralizedLinearMixedModel()
```

### Read data (.csv only)
```python
lmm.read_data("data_path.csv")  # Change with your own data path
```

### Define your variables (i.e., colname in data file)
I recommend you use the exact var name you want to use in your paper.

More specifically, use "_" (underscore) to replace " " (space) in column name and *ezlmm* will help you replace it back in the final report.

```python
# Dependant variable
lmm.dep_var = "rt"

# Independent variable(s)
lmm.indep_var = ["syllable_number", "priming_effect", "speech_isochrony"]

# Random factor(s)
lmm.random_var = ["subject", "word"]
```

### [optional] Select only the correctly responded answers
I use pandas here, so you can edit it in the pandas way
```python
lmm.data = lmm.data[lmm.data['acc'] == 1]
```

### [optional] Code your variable(s)
I use pandas here so you can edit it in the pandas way
```python
lmm.code_variables({
    "syllable_number": {"disyllabic": -0.5, "trisyllabic": 0.5},
    "speech_isochrony": {"averaged": -0.5, "original": 0.5},
    "priming_effect": {"inconsistent": -0.5, "consistent": 0.5}
})
```

In case you do not want to code variables, use this instead:
```python
lmm.code_variables()  # or lmm.code_variables({})
```

### [optional] Exclude trials with RT exceeding 2.5 SD
```python
lmm.exclude_trial_SD(target="rt", subject="subject", SD=2.5)
```


### Finally, try fitting the model
```python
# Fitting the model until it is converged
lmm.fit(optimizer=["bobyqa", 20000], prev_formula="rt ~ syllable_number * priming_effect * speech_isochrony + (1 + speech_isochrony | subject) + (1 | word)")
```