"""Report generation: contrast extraction and APA-style prose."""

import pandas as pd


def extract_contrast(contrast_str: str, factor_num: int = 1) -> list[dict]:
    """
    Parse emmeans character output into a list of contrast dicts.

    Parameters
    ----------
    contrast_str : str
        String representation of emmeans contrast results.
    factor_num : int
        1 for a main effect, 2 for a two-way interaction.

    Returns
    -------
    list[dict]
        Each dict contains contrast results for one comparison.
    """
    raw_contrast = contrast_str.strip().split("\n")
    contrast_dicts = []

    s = 0
    for i in [i for i in range(len(raw_contrast)) if not raw_contrast[i]]:
        contrast_dict = {}
        if factor_num == 1:
            raw_contrast_ = raw_contrast[s:i]
        elif factor_num == 2:
            contrast_dict["under_cond"] = (
                raw_contrast[s].strip(":").replace("=", "").replace(" ", "")
            )
            raw_contrast_ = raw_contrast[s+1:i]

        for j in range(0, len(raw_contrast_), 2):
            contrast_dict |= dict(zip(
                raw_contrast_[j].strip().split(),
                raw_contrast_[j + 1].strip().rsplit(
                    maxsplit=len(raw_contrast_[j].strip().split()) - 1
                )
            ))

        if contrast_dict.get("p.value") == "<.0001":
            contrast_dict["p.value"] = 0.000

        contrast_dicts.append(contrast_dict)
        s = i + 1

    return contrast_dicts


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_detail_lmm(result_dict: dict, used_method: str) -> str:
    """Write detail for LMM — handles ANOVA (F value) and contrast (t.ratio) results."""
    pval = result_dict['p.value']
    pval_str = str(pval) if '<' in str(pval) else f"{float(pval):.3f}"

    # ANOVA main effects/interactions: show F + p
    if 'F value' in result_dict:
        return (
            f"(F={float(result_dict['F value']):.3f}, "
            f"p={pval_str})"
        )
    # Pairwise contrasts: t.ratio IS the statistic — don't duplicate it
    elif 't.ratio' in result_dict:
        method_label = used_method.replace('.ratio', '')
        return (
            f"({method_label}={float(result_dict[used_method]):.3f}, "
            f"p={pval_str})"
        )
    elif 'Chisq' in result_dict:
        return (
            f"(χ²={float(result_dict['Chisq']):.3f}, "
            f"p={pval_str})"
        )
    else:
        return (
            f"({used_method}={result_dict.get(used_method, 'N/A')}, "
            f"p={pval_str})"
        )


def _write_detail_glmm(result_dict: dict, used_method: str) -> str:
    """Write detail for GLMM — handles contrast results with Chisq."""
    pval = result_dict['p.value']
    df_val = result_dict['df']
    df_str = f"df={df_val}, " if df_val != 'Inf' else ""
    pval_str = str(pval) if '<' in str(pval) else f"{float(pval):.3f}"
    return (
        f"(β={result_dict['estimate']}, SE={float(result_dict['SE']):.3f}, "
        f"{df_str}"
        f"{used_method.replace('.ratio', '')}={result_dict[used_method]}, "
        f"p={pval_str})"
    )


def _write_main_lmm(result_df) -> str:
    pval = result_df['Pr(>F)']
    pval_str = str(pval) if '<' in str(pval) else f"{float(pval):.3f}"
    return (
        f"(F({int(result_df['NumDF'])},{result_df['DenDF']:.3f})"
        f"={result_df['F value']:.3f}, p={pval_str})"
    )


def _write_main_glmm(result_df) -> str:
    pval = result_df['Pr(>Chisq)']
    pval_str = str(pval) if '<' in str(pval) else f"{float(pval):.3f}"
    return (
        fr"(\u03c2\u00b2({int(result_df['Df'])})={result_df['Chisq']:.3f}, "
        f"p={pval_str})"
    )


def _readable_level(ref: str, trans_dict: dict, robj_model, data) -> str:
    """Look up a readable name for a contrast level reference.

    Examples:
        'Tsyl-0.5'  → 'disyllabic'
        'Tsyl0.5'   → 'trisyllabic'

    Strategy:
    1. Check trans_dict directly (works when code_variables() was used).
    2. Parse the factor name and numeric suffix from the ref string, then
       search ``data`` for the original factor levels by matching the
       coded column name (strip leading 'T') to the original column.
    3. Fall back to the raw ref string.
    """
    import re as _re

    # Fast path: trans_dict already has the mapping
    if ref in trans_dict:
        return trans_dict[ref]

    # Parse factor name and coded value from strings like 'Tsyl-0.5' or 'Tiso0.5'
    # Sign is optional — presence means negative, absence means positive
    m = _re.match(r'^([A-Za-z_]\w*?)([-\+])?0?\.?5$', ref)
    if not m:
        return ref  # unexpected format, use as-is
    factor, sign = m.group(1), m.group(2)
    coded = -0.5 if sign == '-' else 0.5

    # Try to find original factor levels from the pandas DataFrame
    if data is not None:
        try:
            # Build a mapping from coded column → original column using
            # (a) direct name match after stripping 'T', and
            # (b) longest-common-substring to handle abbreviations
            #     (e.g. 'Tsyl' ↔ 'syllable_number' via 'syl')
            import re as _re

            # Identify coded columns (2 unique values in {-0.5, 0.5})
            coded_cols = {}
            for col in data.columns:
                uniq = data[col].dropna().unique()
                if len(uniq) == 2 and set(uniq).issubset({-0.5, 0.5}):
                    coded_cols[col] = uniq

            # Identify original factor columns (2 unique string values each)
            orig_cols = {}
            for col in data.columns:
                uniq = data[col].dropna().unique()
                if len(uniq) == 2 and all(isinstance(v, str) for v in uniq):
                    orig_cols[col] = list(uniq)  # preserve insertion order

            def _lcs_len(a: str, b: str) -> int:
                """Longest common substring length (order matters)."""
                m, n = len(a), len(b)
                # Quick check: one contains the other
                if a in b:
                    return len(a)
                if b in a:
                    return len(b)
                # DP table for LCS
                dp = [[0] * (n + 1) for _ in range(2)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if a[i - 1] == b[j - 1]:
                            dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
                        else:
                            dp[i % 2][j] = 0
                return max(max(row) for row in dp)

            # Minimum LCS score to accept a match (avoids spurious single-char hits)
            _MIN_LCS = 3

            # Match each coded col to the best original col
            coded_to_orig = {}
            for coded_col in coded_cols:
                base = coded_col[1:] if coded_col.startswith(('T', 't')) else coded_col
                base_lower = base.lower()
                best_orig, best_score = None, 0
                for orig_col in orig_cols:
                    # Score by longest common substring (case-insensitive)
                    score = _lcs_len(base_lower, orig_col.lower())
                    if score > best_score:
                        best_score = score
                        best_orig = orig_col
                if best_orig and best_score >= _MIN_LCS:
                    coded_to_orig[coded_col] = best_orig

            # Now look up the ref
            # Strip T from factor
            factor_stripped = factor[1:] if factor.startswith(('T', 't')) else factor
            # Find the best matching coded column
            best_coded_col, best_match_score = None, 0
            for coded_col in coded_cols:
                base = coded_col[1:] if coded_col.startswith(('T', 't')) else coded_col
                score = _lcs_len(factor_stripped.lower(), base.lower())
                if score > best_match_score:
                    best_match_score = score
                    best_coded_col = coded_col

            if best_coded_col and best_match_score >= _MIN_LCS and best_coded_col in coded_to_orig:
                orig_col = coded_to_orig[best_coded_col]
                # Build empirical mapping from coded → original using actual rows
                rev_map = {}
                for _, row in data[[best_coded_col, orig_col]].dropna().iterrows():
                    cv, ov = row[best_coded_col], row[orig_col]
                    if cv in {-0.5, 0.5} and ov not in rev_map:
                        rev_map[cv] = ov
                if coded in rev_map:
                    return rev_map[coded]
        except Exception:
            pass

    return ref  # give up, use raw ref


def _process_simple_effect(
    dep_var: str,
    sig_items: list[str],
    df_item,
    robj_model: object,
    trans_dict: dict,
    write_detail,
    write_main,
    factor_num: int,
    data: "pd.DataFrame | None" = None,
) -> str:
    """
    Shared simple-effect prose builder for main effects and two-way interactions.

    Parameters
    ----------
    dep_var : str
        Dependent variable name.
    sig_items : list[str]
        Split effect name (e.g. ["Condition"] or ["Condition", "Group"]).
    df_item : pd.Series
        ANOVA row for this effect.
    robj_model : robjects
        Fitted R model.
    trans_dict : dict
        Level name translator.
    write_detail : callable
        write_detail_lmm or write_detail_glmm.
    write_main : callable
        write_main_lmm or write_main_glmm.
    factor_num : int
        1 for main effect, 2 for two-way interaction.

    Returns
    -------
    str
        Prose fragment for this effect.
    """
    from ezlmm.utils import emmeans

    # Intercept is never a meaningful experimental effect — skip it
    if sig_items == ["(Intercept)"] or sig_items == ["Intercept"]:
        return ""

    def _contrast_prose(emmeans_result_dict: dict, prefix: str = "") -> str:
        lvl1_raw = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
        lvl2_raw = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')
        lvl1 = _readable_level(lvl1_raw, trans_dict, robj_model, data)
        lvl2 = _readable_level(lvl2_raw, trans_dict, robj_model, data)
        target_method = [m for m in emmeans_result_dict if ".ratio" in m][0]
        direction = 'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'
        pval_str = str(emmeans_result_dict["p.value"])
        if "<" in pval_str:
            is_sig = True  # e.g. "<0.0001" means p < threshold → significant
        else:
            is_sig = float(pval_str) <= 0.05

        if is_sig:
            body = (
                f"{dep_var.upper()} for {lvl1} condition "
                f"was significantly {direction} than that for "
                f"{lvl2} condition"
            )
        else:
            body = (
                f"no significant difference was found between "
                f"{lvl1} condition and {lvl2} "
                f"condition in {dep_var.upper()}"
            )
        return prefix + body + f" {write_detail(emmeans_result_dict, target_method)}"

    if len(sig_items) == 1:
        effect_name = sig_items[0].replace("_", " ")
        plural = "was"
    elif len(sig_items) == 2:
        effect_name = " × ".join(s.replace("_", " ") for s in sig_items)
        plural = "was"
    elif len(sig_items) >= 3:
        effect_name = " × ".join(s.replace("_", " ") for s in sig_items)
        plural = "was"
    final_rpt = f"The {effect_name} {plural} significant {write_main(df_item)}." + " "

    if len(sig_items) == 1:
        try:
            emmeans_result = emmeans.contrast(
                emmeans.emmeans(robj_model, sig_items[0]),
                "pairwise", adjust="bonferroni"
            )
            emmeans_result_dict = extract_contrast(
                str(emmeans_result), factor_num=factor_num
            )[0]
            final_rpt += "Post-hoc analysis revealed that "
            final_rpt += _contrast_prose(emmeans_result_dict) + ". "
        except Exception as e:
            final_rpt += (
                f"[Post-hoc analysis could not be conducted "
                f"(model reference grid issue — the random structure may be too complex for emmeans).] "
            )

    elif len(sig_items) == 2:
        final_rpt += "Simple effect analysis showed that, "
        contrast_parts = []
        try:
            for i1, i2 in [(0, 1), (-1, -2)]:
                emmeans_result = emmeans.contrast(
                    emmeans.emmeans(robj_model, specs=sig_items[i1], by=sig_items[i2]),
                    "pairwise", adjust="bonferroni"
                )
                for emmeans_result_dict in extract_contrast(
                    str(emmeans_result), factor_num=factor_num
                ):
                    contrast_parts.append(_contrast_prose(
                        emmeans_result_dict,
                        prefix=f"under {_readable_level(emmeans_result_dict['under_cond'], trans_dict, robj_model, data)} condition, ",
                    ))
            final_rpt += "; ".join(p.rstrip("; ") for p in contrast_parts) + ". "
        except Exception as e:
            final_rpt += (
                f"[Simple effect analysis could not be conducted "
                f"(model reference grid issue — the random structure may be too complex for emmeans).] "
            )

    elif len(sig_items) >= 3:
        final_rpt += (
            f"[Not yet ready for simple simple effect analysis "
            f"({', '.join(sig_items)}). "
            f"Please construct individual models by subsetting your data.]"
        )

    return final_rpt


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def write_simple_effect_lmm(
    dep_var: str,
    trans_dict: dict,
    optimal_model: str,
    df_anova,
    robj_model,
    data: pd.DataFrame | None = None,
) -> str:
    """
    Generate APA-style prose for a linear mixed model.

    Handles main effects and two-way interactions; three-way and above
    return a "not yet ready" placeholder.
    """
    write_detail = _write_detail_lmm
    write_main = _write_main_lmm
    return _write_simple_effect(
        dep_var, trans_dict, optimal_model, df_anova, robj_model,
        write_detail, write_main,
        test_name="F test",
        main_test="Pr(>F)",
        data=data,
    )


def write_simple_effect_glmm(
    dep_var: str,
    trans_dict: dict,
    optimal_model: str,
    df_anova,
    robj_model,
    data: pd.DataFrame | None = None,
) -> str:
    """
    Generate APA-style prose for a generalized linear mixed model.

    Handles main effects and two-way interactions; three-way and above
    return a "not yet ready" placeholder.
    """
    write_detail = _write_detail_glmm
    write_main = _write_main_glmm
    return _write_simple_effect(
        dep_var, trans_dict, optimal_model, df_anova, robj_model,
        write_detail, write_main,
        test_name="Wald Chi-square test",
        main_test="Pr(>Chisq)",
        data=data,
    )


def _write_simple_effect(
    dep_var: str,
    trans_dict: dict,
    optimal_model: str,
    df_anova,
    robj_model: object,
    write_detail,
    write_main,
    test_name: str,
    main_test: str,
    data: "pd.DataFrame | None" = None,
) -> str:
    """Shared implementation for LMM and GLMM report writers."""
    final_rpt = (
        f"For {dep_var.upper()} data, {test_name} was conducted "
        f"on the optimal model ({optimal_model}). "
    )

    # Significant effects
    sig_mask = df_anova[main_test] <= 0.05
    for effect_name in df_anova[sig_mask].index.tolist():
        if effect_name == "Intercept" or ":" in effect_name and "Intercept" in effect_name:
            continue
        df_item = df_anova[sig_mask].loc[effect_name]
        sig_items = effect_name.split(":")

        final_rpt += _process_simple_effect(
            dep_var, sig_items, df_item, robj_model,
            trans_dict, write_detail, write_main,
            factor_num=len(sig_items),
            data=data,
        )

    # Non-significant effects
    for effect_name in df_anova[~sig_mask].index.tolist():
        if effect_name == "Intercept" or ":" in effect_name and "Intercept" in effect_name:
            continue
        df_item = df_anova[~sig_mask].loc[effect_name]
        items = [si.replace("_", " ") for si in effect_name.split(":")]

        if len(items) == 1:
            final_rpt += (
                f"The main effect of {items[0]} was not significant "
                f"{write_main(df_item)}. "
            )
        else:
            conj = "×".join(items)  # e.g. "Tpriming × Tiso"
            final_rpt += (
                f"The interaction {conj} was not significant "
                f"{write_main(df_item)}. "
            )

    final_rpt = final_rpt.replace("=0.000", "<0.001")
    return final_rpt
