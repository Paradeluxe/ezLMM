"""
ezlmm Integration Test — uses Data_Experiment.csv (6022 rows, 64 subjects)

Tests the full ezlmm pipeline end-to-end:
  1. Preprocess: load Data_Experiment.csv, add contrast-coded columns
  2. Fit LinearMixedModel (rt ~ priming * exp_type, random = subject)
  3. Fit GeneralizedLinearMixedModel (acc ~ priming * exp_type, random = subject)
  4. Verify outputs are sane DataFrames
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import ezlmm
from ezlmm import LinearMixedModel, GeneralizedLinearMixedModel

# ── resolve repo root ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
src = _REPO_ROOT / "data.csv"           # 2986 rows, 64 subjects, 2×2×2 factorial
out = _REPO_ROOT / "test_integration_data.csv"

data = pd.read_csv(src)
print(f"[1] Loaded {len(data)} rows: {list(data.columns)}")
print(f"    Subjects: {data['subject'].nunique()}, Words: {data['word'].nunique()}")

# ── 2. Add orthogonal contrast columns ───────────────────────────────────────
# priming_effect: inconsistent (-0.5) vs consistent (+0.5)
data["Tpriming"] = data["priming_effect"].map({"inconsistent": -0.5, "consistent": 0.5})
# speech_isochrony: original (-0.5) vs averaged (+0.5)
data["Tspeech"] = data["speech_isochrony"].map({"original": -0.5, "averaged": 0.5})
# syllable_number: disyllabic (-0.5) vs trisyllabic (+0.5)
data["Tsyllable"] = data["syllable_number"].map({"disyllabic": -0.5, "trisyllabic": 0.5})

data.to_csv(out, index=False)
print(f"[2] Saved contrast-coded data to {out}")

# ── 2. Fit LMM (reaction time as continuous outcome) ───────────────────────
print("\n[3] ── Fitting LinearMixedModel (RT) ──")
lmm = LinearMixedModel()
lmm.read_data(out)

lmm.dep_var = "rt"
lmm.indep_var = ["Tpriming", "Tspeech", "Tsyllable"]
lmm.random_var = ["subject"]

lmm.fit(report=False)
print(f"    LMM anova shape: {lmm.anova.shape}")
print(f"    LMM anova columns: {list(lmm.anova.columns)}")

# Sanity checks
assert isinstance(lmm.anova, pd.DataFrame), "LMM anova is not a DataFrame!"
assert len(lmm.anova) > 0, "LMM anova is empty!"
print(f"    LMM anova:\n{lmm.anova}")

# ── 3. Fit GLMM (accuracy as binary outcome) ───────────────────────────────
print("\n[4] ── Fitting GeneralizedLinearMixedModel (ACC) ──")
glmm = GeneralizedLinearMixedModel()
glmm.read_data(out)

glmm.dep_var = "acc"
glmm.indep_var = ["Tpriming", "Tspeech", "Tsyllable"]
glmm.random_var = ["subject"]
glmm.family = "binomial"

glmm.fit(report=False)
print(f"    GLMM anova shape: {glmm.anova.shape}")
print(f"    GLMM anova columns: {list(glmm.anova.columns)}")

# Sanity checks
assert isinstance(glmm.anova, pd.DataFrame), "GLMM anova is not a DataFrame!"
assert len(glmm.anova) > 0, "GLMM anova is empty!"
print(f"    GLMM anova:\n{glmm.anova}")

# ── 4. Clean up temp file ────────────────────────────────────────────────────
os.remove(out)
print(f"\n[5] Temp file cleaned up.")

print("\n" + "=" * 50)
print("✅ Integration test PASSED — all outputs valid")
print("=" * 50)
