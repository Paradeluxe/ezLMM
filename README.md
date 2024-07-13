# ezLMM: Python interface of Linear Mixed Model analysis in R language

## Before installation

Make sure you have downloaded [R interpreter](https://www.r-project.org/) in your system and add it to PATH.

## Install package
```bash
pip install ezlmm
```

# How to use this package
## Example: LMM on Reaction Time
```python
from ezlmm import LinearMixedModel, GeneralizedLinearMixedModel


lmm = LinearMixedModel()
# glmm = GeneralizedLinearMixedModel()

lmm.read_data("data_path.csv")  # Change with your own data path


# Dependant variable
lmm.dep_var = "rt"
# Independent variable(s)
lmm.indep_var = ["syllable_number", "priming_effect", "speech_isochrony"]
# Random factor(s)
lmm.random_var = ["subject", "word"]


# [Optional]
lmm.exclude_trial_SD(target="rt", subject="subject", SD=2.5)
lmm.data = lmm.data[lmm.data['acc'] == 1]
lmm.code_variables({
    "syllable_number": {"disyllabic": -0.5, "trisyllabic": 0.5},
    "speech_isochrony": {"averaged": -0.5, "original": 0.5},
    "priming_effect": {"inconsistent": -0.5, "consistent": 0.5}
})


# Fitting the model until it is converged
lmm.fit()

# Print output
print(lmm.report)  # plain text for your paper
print(lmm.summary)  # model fitting 
print(lmm.anova)  # anova on fixed model
```

## Step by Step
### Need to know
- *ezlmm* executes from **full model** to **null model** for random model.
- *ezlmm* operates on **full fixed model** and won't change your fixed model. (i.e., It stays the same.)
- *ezlmm* currently considers **random intercept** as default for each random effect.

In other words, 
- *ezlmm* goes from
`dv ~ iv1 * iv2 + (1+iv1:iv2+iv1+iv2|rf)` to `dv ~ iv1 * iv2 + (1|rf)`.
- It only stops and generates output until (1) it meets a model that fits, or (2) no a single model fits.

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

### [optional] Exclude trials with RT exceeding 2.5 SD
```python
lmm.exclude_trial_SD(target="rt", subject="subject", SD=2.5)
```

### [optional] Select only the correctly responded answers
I use pandas here, so you can edit it in the pandas way
```python
lmm.data = lmm.data[lmm.data['acc'] == 1]
```

### [optional] Code your variable(s)
The structure should be: `var_name: {lvl_name: lvl_name_coded}`
```python
lmm.code_variables({
    "syllable_number": {"disyllabic": -0.5, "trisyllabic": 0.5},
    "speech_isochrony": {"averaged": -0.5, "original": 0.5},
    "priming_effect": {"inconsistent": -0.5, "consistent": 0.5}
})
```

In case you do not want to code these variables, use this instead:
```python
lmm.code_variables()  # or lmm.code_variables({})
```


### Finally, try fitting the model

To start from the ***full model***, use this
```python
# Fitting the model until it is converged
lmm.fit()
```

In case you want to use optimizer, add *optimizer* parameter (I have only tested "bobyqa" yet XD)
```python
# give it a list, i.e., [optimizer name, iteration times]
lmm.fit(optimizer=["bobyqa", 20000])
```

Running the first few formulas can take quite long time. To start from the midpoint,
```python
# This is just an example, type in any formula you want it to start from
lmm.fit(prev_formula="rt ~ syllable_number * priming_effect * speech_isochrony + (1 + speech_isochrony | subject) + (1 | word)")
```
*ezlmm* currently does not support you add an extra variable that does not appear in lmm.indep_var to random model.

### Print out the result
I add a function that can generate report, which you can simply copy-paste to your paper.
```python
print(lmm.report)
```

To see the model fitting results, and F-test like anova results on fixed model,
```python
# model fitting
print(lmm.summary)

# anova on fixed model
print(lmm.anova)
```



## For the record

### Why I wrote *ezLMM*

You see, my procedure was to start from full random model to null random model, and discard random slope with 
the least variance. It is not a big deal if you have only 1 or 2 independent variables, but not for 3 and above. 
You can easily misjudge which item has the least variance. It is annoying and tiring. That is the main reason.

One more thing is that, R language was simply not my thing. But still, I think the best way you can do LMM analysis is to 
use R packages like `lme4`, `lmerTest`, etc. Therefore, I created *ezlmm*, so that you can enjoy the convenience of Python, and
at the same time cite these R packages. *ezlmm* is just a Python interface, it's not a big deal; you should credit them for their great work.



### My understanding of LMM

There was one day I was having a talk with my gf about independent T test and paired T test. I came to understand that, although
paired T test can take between-subject difference into consideration, it needs to average a lot of trials, which might not be a good thing;
for independent T test, it preserves the integrity of your data, it fails to take between-subject difference into account. 

And, same thing for repeated-measures ANOVA and regular ANOVA: rm ANOVA needs averaging on each condition; regular ANOVA fails to consider the effect of repeated measures.

Therefore, for my understanding, I believe LMM is the combination of these two advantages: 
- not averaging your data
- taking the consideration of between-subject/item/XXXX differences (depending on what's your random factors).

# Contact me
I am a Psychology/NeuroScience student currently doing my Master program. My area is Psycholinguistics.
That is to say, I am not a professional in terms of neither statistics nor machine learning.
I wrote a lot of scripts for my own research as well as my colleagues'. 
So this is kind of my first Python package (lol I don't even know how to use *pull request*)

Feel free to reach out to me at `paradeluxe3726@gmail.com` or `zhengyuan.liu@connect.um.edu.mo`.








