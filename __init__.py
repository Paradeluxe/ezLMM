from core.ezlmm import LinearMixedModel


__all__ = ["LinearMixedModel"]


if __name__ == "__main__":

    # Create instance
    lmm = LinearMixedModel()  # or you can import it as LMM(), lmm(), or anything you like

    # Write in path
    lmm.read_data(r"C:\Users\18357\Desktop\linguistic_rhythm\data.csv")

    # [optional] If you want to do some extra editing to your data
    lmm.data = lmm.data[lmm.data['acc'] == 1]  # rt self.data works on ACC = 1



    # Define your variables
    lmm.dep_var = "rt"
    lmm.indep_var = ["syllable_number", "priming_effect", "speech_isochrony"]
    lmm.random_var = ["subject", "word"]

    # [optional] Code your variables (coded var will start with 't', e.g., "name" -> "Tname")
    lmm.code_variables({
        "syllable_number": {"disyllabic": -0.5, "trisyllabic": 0.5},
        "speech_isochrony": {"averaged": -0.5, "original": 0.5},
        "priming_effect": {"inconsistent": -0.5, "consistent": 0.5}
    })


    # Process your data
    lmm.exclude_trial_SD(target="rt", subject="subject", SD=2.5)

    # Fitting the model until it is converged
    lmm.fit(optimizer=["bobyqa", 1000], prev_formula="rt ~ syllable_number * priming_effect * speech_isochrony + (1 | subject) + (1 | word)")

    # Print report
    print(lmm.report)
