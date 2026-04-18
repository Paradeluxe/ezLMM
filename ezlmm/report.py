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
    def write_detail(result_dict: dict, used_method: str) -> str:
        return (
            f"(β={result_dict['estimate']}, SE={result_dict['SE']}, "
            f"{'df=' + result_dict['df'] + ', ' if result_dict['df'] != 'Inf' else ''}"
            f"{used_method.replace('.ratio', '')}={result_dict[used_method]}, "
            f"p={float(result_dict['p.value']):.3f})"
        )

    def write_main(result_df) -> str:
        return (
            f"(F({int(result_df['NumDF'])},{result_df['DenDF']:.3f})"
            f"={result_df['F value']:.3f}, p={result_df['Pr(>F)']:.3f})"
        )

    main_test = [t for t in df_anova.head() if "Pr(>" in t][0]
    target_test_str = "F test" if main_test == "Pr(>F)" else "[Unknown test]"

    final_rpt = (
        f"For {dep_var.upper()} data, {target_test_str} was conducted "
        f"on the optimal model ({optimal_model}). "
    )

    # Significant effects
    for sig_items in df_anova[df_anova[main_test] <= 0.05].index.tolist():
        if "ntercept" in sig_items:
            continue
        df_item = df_anova[df_anova[main_test] <= 0.05].loc[sig_items]
        sig_items = sig_items.split(":")

        from ezlmm.utils import emmeans
        if len(sig_items) == 1:
            emmeans_result = emmeans.contrast(
                emmeans.emmeans(robj_model, sig_items[0]),
                "pairwise", adjust="bonferroni"
            )
            emmeans_result_dict = extract_contrast(
                str(emmeans_result), factor_num=len(sig_items)
            )[0]
            target_method = [m for m in emmeans_result_dict if ".ratio" in m][0]
            lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
            lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')

            final_rpt += (
                f"The main effect of {sig_items[0].replace('_', ' ')} was"
                f" significant {write_main(df_item)}. "
                f"Post-hoc analysis revealed that "
            )
            direction = 'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'
            if float(emmeans_result_dict["p.value"]) <= 0.05:
                final_rpt += (
                    f"{dep_var.upper()} for {trans_dict[lvl1]} condition "
                    f"was significantly {direction} than that for "
                    f"{trans_dict[lvl2]} condition"
                )
            else:
                final_rpt += (
                    f"there was no significant difference between "
                    f"{trans_dict[lvl1]} condition and {trans_dict[lvl2]} condition "
                    f"in {dep_var.upper()}"
                )
            final_rpt += f" {write_detail(emmeans_result_dict, target_method)}. "

        elif len(sig_items) == 2:
            final_rpt += (
                f"The interaction between {sig_items[0].replace('_', ' ')} and "
                f"{sig_items[1].replace('_', ' ')} was significant {write_main(df_item)}. "
                f"Simple effect analysis showed that, "
            )
            for i1, i2 in [(0, 1), (-1, -2)]:
                from ezlmm.utils import emmeans
                emmeans_result = emmeans.contrast(
                    emmeans.emmeans(robj_model, specs=sig_items[i1], by=sig_items[i2]),
                    "pairwise", adjust="bonferroni"
                )
                for emmeans_result_dict in extract_contrast(
                    str(emmeans_result), factor_num=len(sig_items)
                ):
                    lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
                    lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')
                    target_method = [m for m in emmeans_result_dict if ".ratio" in m][0]
                    direction = 'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'
                    final_rpt += (
                        f"under {trans_dict[emmeans_result_dict['under_cond']]} condition, "
                    )
                    if float(emmeans_result_dict["p.value"]) <= 0.05:
                        final_rpt += (
                            f"{dep_var.upper()} for {trans_dict[lvl1]} condition "
                            f"was significantly {direction} than that for "
                            f"{trans_dict[lvl2]} condition "
                            f"{write_detail(emmeans_result_dict, target_method)}"
                        )
                    else:
                        final_rpt += (
                            f"no significant difference was found between "
                            f"{trans_dict[lvl1]} condition and {trans_dict[lvl2]} "
                            f"condition in {dep_var.upper()} "
                            f"{write_detail(emmeans_result_dict, target_method)}"
                        )
                    final_rpt += "; "
            final_rpt = final_rpt[:-2] + ". "

        elif len(sig_items) >= 3:
            final_rpt += (
                f"The interaction among {', '.join(sig_items[:-1])} and "
                f"{sig_items[-1]} was significant {write_main(df_item)}. "
                f"[Not yet ready for simple simple effect analysis "
                f"({', '.join(sig_items)}). "
                f"Please construct individual models by subsetting your data.]"
            )

    # Non-significant effects
    for sig_items in df_anova[df_anova[main_test] > 0.05].index.tolist():
        if "ntercept" in sig_items:
            continue
        df_item = df_anova[df_anova[main_test] > 0.05].loc[sig_items]
        sig_items = [si.replace("_", " ") for si in sig_items.split(":")]

        if len(sig_items) == 1:
            final_rpt += (
                f"The main effect of {sig_items[0]} was not significant "
                f"{write_main(df_item)}. "
            )
        elif len(sig_items) >= 2:
            final_rpt += (
                f"The interaction {'among' if len(sig_items) > 2 else 'between'} "
                f"{', '.join(sig_items[:-1])} and {sig_items[-1]} was not significant "
                f"{write_main(df_item)}. "
            )

    final_rpt = final_rpt.replace("=0.000", "<0.001")
    return final_rpt


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
    def write_detail(result_dict: dict, used_method: str) -> str:
        return (
            f"(β={result_dict['estimate']}, SE={float(result_dict['SE']):.3f}, "
            f"{'df=' + result_dict['df'] + ', ' if result_dict['df'] != 'Inf' else ''}"
            f"{used_method.replace('.ratio', '')}={result_dict[used_method]}, "
            f"p={float(result_dict['p.value']):.3f})"
        )

    def write_main(result_df) -> str:
        return (
            fr"(\chi^2({int(result_df['Df'])})={result_df['Chisq']:.3f}, "
            f"p={result_df['Pr(>Chisq)']:.3f})"
        )

    main_test = [t for t in df_anova.head() if "Pr(>" in t][0]
    target_test_str = "Wald Chi-square test" if main_test == "Pr(>Chisq)" else "[Unknown test]"

    final_rpt = (
        f"For {dep_var.upper()} data, {target_test_str} was conducted "
        f"on the optimal model ({optimal_model}). "
    )

    # Significant effects
    for sig_items in df_anova[df_anova[main_test] <= 0.05].index.tolist():
        if "ntercept" in sig_items:
            continue
        df_item = df_anova[df_anova[main_test] <= 0.05].loc[sig_items]
        sig_items = sig_items.split(":")

        from ezlmm.utils import emmeans
        if len(sig_items) == 1:
            emmeans_result = emmeans.contrast(
                emmeans.emmeans(robj_model, sig_items[0]),
                "pairwise", adjust="bonferroni"
            )
            emmeans_result_dict = extract_contrast(
                str(emmeans_result), factor_num=len(sig_items)
            )[0]
            target_method = [m for m in emmeans_result_dict if ".ratio" in m][0]
            lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
            lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')

            final_rpt += (
                f"The main effect of {sig_items[0].replace('_', ' ')} was significant "
                f"{write_main(df_item)}. Post-hoc analysis revealed that "
            )
            direction = 'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'
            if float(emmeans_result_dict["p.value"]) <= 0.05:
                final_rpt += (
                    f"{dep_var.upper()} for {trans_dict[lvl1]} condition "
                    f"was significantly {direction} than that for "
                    f"{trans_dict[lvl2]} condition"
                )
            else:
                final_rpt += (
                    f"there was no significant difference between "
                    f"{trans_dict[lvl1]} condition and {trans_dict[lvl2]} "
                    f"condition in {dep_var.upper()}"
                )
            final_rpt += f" {write_detail(emmeans_result_dict, target_method)}. "

        elif len(sig_items) == 2:
            final_rpt += (
                f"The interaction between {sig_items[0].replace('_', ' ')} and "
                f"{sig_items[1].replace('_', ' ')} was significant {write_main(df_item)}. "
                f"Simple effect analysis showed that, "
            )
            for i1, i2 in [(0, 1), (-1, -2)]:
                from ezlmm.utils import emmeans
                emmeans_result = emmeans.contrast(
                    emmeans.emmeans(robj_model, specs=sig_items[i1], by=sig_items[i2]),
                    "pairwise", adjust="bonferroni"
                )
                for emmeans_result_dict in extract_contrast(
                    str(emmeans_result), factor_num=len(sig_items)
                ):
                    lvl1 = emmeans_result_dict['contrast'].split(' - ')[0].strip().strip('()')
                    lvl2 = emmeans_result_dict['contrast'].split(' - ')[1].strip().strip('()')
                    target_method = [m for m in emmeans_result_dict if ".ratio" in m][0]
                    direction = 'higher' if float(emmeans_result_dict['estimate']) > 0 else 'lower'
                    final_rpt += (
                        f"under {trans_dict[emmeans_result_dict['under_cond']]} condition, "
                    )
                    if float(emmeans_result_dict["p.value"]) <= 0.05:
                        final_rpt += (
                            f"{dep_var.upper()} for {trans_dict[lvl1]} condition "
                            f"was significantly {direction} than that for "
                            f"{trans_dict[lvl2]} condition "
                            f"{write_detail(emmeans_result_dict, target_method)}"
                        )
                    else:
                        final_rpt += (
                            f"no significant difference was found between "
                            f"{trans_dict[lvl1]} condition and {trans_dict[lvl2]} "
                            f"condition in {dep_var.upper()} "
                            f"{write_detail(emmeans_result_dict, target_method)}"
                        )
                    final_rpt += "; "
            final_rpt = final_rpt[:-2] + ". "

        elif len(sig_items) >= 3:
            final_rpt += (
                f"The interaction among {', '.join(sig_items[:-1])} and "
                f"{sig_items[-1]} was significant {write_main(df_item)}. "
                f"[Not yet ready for simple simple effect analysis "
                f"({', '.join(sig_items)}). "
                f"Please construct individual models by subsetting your data.]"
            )

    # Non-significant effects
    for sig_items in df_anova[df_anova[main_test] > 0.05].index.tolist():
        if "ntercept" in sig_items:
            continue
        df_item = df_anova[df_anova[main_test] > 0.05].loc[sig_items]
        sig_items = [si.replace("_", " ") for si in sig_items.split(":")]

        if len(sig_items) == 1:
            final_rpt += (
                f"The main effect of {sig_items[0]} was not significant "
                f"{write_main(df_item)}. "
            )
        elif len(sig_items) >= 2:
            final_rpt += (
                f"The interaction {'among' if len(sig_items) > 2 else 'between'} "
                f"{', '.join(sig_items[:-1])} and {sig_items[-1]} was not significant "
                f"{write_main(df_item)}. "
            )

    final_rpt = final_rpt.replace("=0.000", "<0.001")
    return final_rpt
