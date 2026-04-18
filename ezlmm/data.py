"""Data handling: loading, coding, exclusion, descriptive stats."""

import itertools

import pandas as pd


def read_data(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read a data file into a DataFrame.

    Supported formats: csv, tsv, xlsx, json, hdf, xml.
    """
    extension = path.split('.')[-1].lower()
    if extension in ['csv', 'tsv']:
        return pd.read_csv(path, encoding=encoding)
    elif extension == 'xlsx':
        return pd.read_excel(path)
    elif extension == 'json':
        return pd.read_json(path, encoding=encoding)
    elif extension == 'hdf':
        return pd.read_hdf(path, encoding=encoding)
    elif extension == 'xml':
        return pd.read_xml(path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


class DataLoader:
    """
    Shared data-handling methods for LMM and GLMM.

    Subclasses must define ``dep_var``, ``indep_var``, ``random_var``.
    """

    def __init__(self):
        self.path = None
        self.data = None
        self.isExcludeSD = True
        self.dep_var = ""
        self.indep_var = []
        self.random_var = []
        self.trans_dict = {}
        self.delete_random_item = []
        self.is_output = True

    def read_data(self, path: str, encoding: str = "utf-8") -> None:
        """Load data from a file path."""
        print("Reading Data...", end="")
        self.data = read_data(path, encoding)
        print("Data collected!")

    def code_variables(self, code=None) -> None:
        """
        Recode column values using a mapping dict.

        If ``code`` is None, each independent variable is coded
        numerically in the order its unique values appear.
        """
        if code is None:
            code = {}

        print("Coding variables...", end="")
        if not code:
            for var in self.indep_var:
                code[var] = dict(zip(
                    self.data[var].unique(),
                    self.data[var].unique()
                ))

        for var_name, mapping in code.items():
            self.data[var_name] = self.data[var_name].apply(lambda x: mapping[x])

        for k, v in code.items():
            for cond in v:
                self.trans_dict[k + str(v[cond])] = cond
                self.trans_dict[cond] = cond

        print("Variables coded!")

    def exclude_trial_SD(self, target: str, subject: str, SD: float = 2.5) -> None:
        """
        Exclude trials beyond ``SD`` standard deviations from each subject's mean.

        Operates per-subject on ``target`` column.
        """
        new_data = pd.DataFrame()
        for sub in list(set(self.data[subject])):
            sub_data = self.data[self.data[subject] == sub]
            mean_val = sub_data[target].mean()
            std_val = sub_data[target].std()
            filtered = sub_data[
                (sub_data[target] >= (mean_val - SD * std_val)) &
                (sub_data[target] <= (mean_val + SD * std_val))
            ]
            new_data = pd.concat([new_data, filtered], ignore_index=True)
        self.data = new_data.copy()

    def descriptive_stats(self) -> pd.DataFrame:
        """Return descriptive statistics (mean ± SD) for each cell of the design."""
        if not self.dep_var and not self.indep_var:
            raise ValueError("Dependent/independent variables not yet specified.")

        df_des = pd.DataFrame()
        conds = [self.data[vid].unique().tolist() for vid in self.indep_var]
        for cond in itertools.product(*conds):
            sub_df = self.data
            for col, val in dict(zip(self.indep_var, cond)).items():
                sub_df = sub_df[sub_df[col] == val]
            sub_df_des = sub_df.describe()[self.dep_var].T
            for col, val in dict(zip(self.indep_var, cond)).items():
                sub_df_des[col] = val
            sub_df_des = sub_df_des.reset_index().T
            sub_df_des.columns = sub_df_des.iloc[0]
            sub_df_des = sub_df_des[1:]
            df_des = pd.concat([df_des, sub_df_des], ignore_index=True)

        df_des[self.dep_var] = df_des.apply(
            lambda row: f"{row['mean']:.3f} ({row['std']:.3f})", axis=1
        )
        return df_des
