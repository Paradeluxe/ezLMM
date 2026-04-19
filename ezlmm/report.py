"""Report generation: contrast extraction and APA-style prose."""


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
    return (
        f"(β={result_dict['estimate']}, SE={result_dict['SE']}, "
        f"{'df=' + result_dict['df'] + ', ' if result_dict['df'] != 'Inf' else ''}"
        f"{used_method.replace('.ratio', '')}={result_dict[used_method]}, "
        f"p={float(result_dict['p.value']):.3f})"
    )


def _write_detail_glmm(result_dict: dict, used_method: str) -> str:
    return (
        f"(β={result_dict['estimate']}, SE={float(result_dict['SE']):.3f}, "
        f"{'df=' + result_dict['df'] + ', ' if result_dict['df'] != 'Inf' else ''}"
        f"{used_method.replace('.ratio', '')}={result_dict[used_method]}, "
        f"p={float(result_dict['p.value']):.3f})"
    )


def _write_main_lmm(result_df) -> str:
    return (
        f"(F({int(result_df['NumDF'])},{result_df['DenDF']:.3f})"
        f"={result_df['F value']:.3f}, p={result_df['Pr(>F)']:.3f})"
    )


def _write_main_glmm(result_df) -> str:
    return (
        fr"(\u03c7\u00b2({int(result_df['Df'])})={result_df['Chisq']:.3f}, "
        f"p={result_df['Pr(>Chisq)']:.3f})"
    )


def _process_simple_effect(
    dep_var: str,
    sig_items: list[str],
    df_item,
    robj_model: object,
    trans_dict: dict,
    write_detail,
    write_main,
    factor_num: int,
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

    def _contrast_prose(emmeans_result_dict: dict, prefix: str = "") -> str:
        lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
        lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')
        target_method = [m for m in emmeans_result_dict if ".ratio" in m][0]
        direction = 'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'
        is_sig = float(emmeans_result_dict["p.value"]) <= 0.05

        if is_sig:
            body = (
                f"{dep_var.upper()} for {trans_dict[lvl1]} condition "
                f"was significantly {direction} than that for "
                f"{trans_dict[lvl2]} condition"
            )
        else:
            body = (
                f"no significant difference was found between "
                f"{trans_dict[lvl1]} condition and {trans_dict[lvl2]} "
                f"condition in {dep_var.upper()}"
            )
        return prefix + body + f" {write_detail(emmeans_result_dict, target_method)}."

    effect_name = " and ".join(s.replace("_", " ") for s in sig_items)
    plural = "was" if len(sig_items) == 1 else "were"
    final_rpt = f"The {effect_name} {plural} significant {write_main(df_item)}. "

    if len(sig_items) == 1:
        emmeans_result = emmeans.contrast(
            emmeans.emmeans(robj_model, sig_items[0]),
            "pairwise", adjust="bonferroni"
        )
        emmeans_result_dict = extract_contrast(
            str(emmeans_result), factor_num=factor_num
        )[0]
        final_rpt += "Post-hoc analysis revealed that "
        final_rpt += _contrast_prose(emmeans_result_dict).replace("..", ".")

    elif len(sig_items) == 2:
        final_rpt += "Simple effect analysis showed that, "
        for i1, i2 in [(0, 1), (-1, -2)]:
            emmeans_result = emmeans.contrast(
                emmeans.emmeans(robj_model, specs=sig_items[i1], by=sig_items[i2]),
                "pairwise", adjust="bonferroni"
            )
            for emmeans_result_dict in extract_contrast(
                str(emmeans_result), factor_num=factor_num
            ):
                final_rpt += _contrast_prose(
                    emmeans_result_dict,
                    prefix=f"under {trans_dict[emmeans_result_dict['under_cond']]} condition, "
                )
                final_rpt += "; "
        final_rpt = final_rpt[:-2] + ". "

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
    )


def write_simple_effect_glmm(
    dep_var: str,
    trans_dict: dict,
    optimal_model: str,
    df_anova,
    robj_model,
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
            conj = "among" if len(items) > 2 else "between"
            final_rpt += (
                f"The interaction {conj} "
                f"{', '.join(items[:-1])} and {items[-1]} was not significant "
                f"{write_main(df_item)}. "
            )

    final_rpt = final_rpt.replace("=0.000", "<0.001")
    return final_rpt
