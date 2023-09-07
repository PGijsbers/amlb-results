"""Defines operations which are commonly performed on the amlb result data"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

def is_old(framework: str, constraint: str, metric: str) -> bool:
    """Encodes the table in `raw_to_clean.ipynb`"""
    if framework == "TunedRandomForest":
        return True
    if constraint == "1h8c_gp3":
        return False
    if framework in ["autosklearn2", "GAMA(B)", "TPOT"]:
        return True
    return framework == "MLJAR(B)" and metric != "neg_rmse"
    

def get_print_friendly_name(name: str, extras: dict[str, str] = None) -> str:
    if extras is None:
        extras = {}
        
    frameworks = {
        "AutoGluon_benchmark": "AutoGluon(B)",
        "AutoGluon_hq": "AutoGluon(HQ)",
        "AutoGluon_hq_il001": "AutoGluon(HQIL)",
        "GAMA_benchmark": "GAMA(B)",
        "mljarsupervised_benchmark": "MLJAR(B)",
        "mljarsupervised_perform": "MLJAR(P)",
    }
    budgets = {
        "1h8c_gp3": "1 hour",
        "4h8c_gp3": "4 hours",
    }
    print_friendly_names = (frameworks | budgets | extras)
    return print_friendly_names.get(name, name)


def impute_results(results: pd.DataFrame, where: pd.Series, with_: str = "constantpredictor", indicator_column: str = "imputed") -> pd.DataFrame:
    """Impute result column of `results`, `where_` a condition holds true, `with_` the result of another framework.

    results: pd.DataFrame
      Regular AMLB results dataframe, must have columns "framework", "task", "fold", "constraint", and "result".
    where: pd.Series
      A logical index into `results` that defines the row where "result" should be imputed.
    with_: str
      The name of the "framework" which should be used to determine the value to impute with.
    indicator_column: str, optional
      The name of the column where a boolean will mark whether or not the "result" value of the row was imputed.
    
    Returns a copy of the original dataframe with imputed results.
    """
    if with_ not in results["framework"].unique():
        raise ValueError(f"{with_=} is not in `results`")
    results = results.copy()
    
    if indicator_column and indicator_column not in results.columns:
        results[indicator_column] = False
        
    lookup_table = results.set_index(["framework", "task", "fold", "constraint"])
    for index, row in results[where].iterrows():
        task, fold, constraint = row[["task", "fold", "constraint"]]
        results.loc[index, "result"] = lookup_table.loc[(with_, task, fold, constraint)].result 
        if indicator_column:
            results.loc[index, indicator_column] = True
    return results

def calculate_ranks(results: pd.DataFrame) -> dict[str, float]:
    """Produce a mapping framework->rank based on ranking mean performance per task"""
    mean_performance = results[["framework", "task", "result"]].groupby(["framework", "task"], as_index=False).mean()
    mean_performance["rank"] = mean_performance.groupby("task").result.rank(ascending=False, method="average", na_option="bottom")
    ranks_by_framework = {
        framework: mean_performance[mean_performance["framework"] == framework]["rank"]
        for framework in mean_performance["framework"].unique()
    }
    
    _, p = friedmanchisquare(*ranks_by_framework.values())
    if p >= 0.05:
        # Given the number of results we don't really expect this to happen.
        raise RuntimeError("Ranks are not statistically significantly different.")
    
    return {framework: ranks.mean() for framework, ranks in ranks_by_framework.items()}

def add_rescale(data: pd.DataFrame, lower: str) -> pd.DataFrame:
    """Adds a `scaled` column to data scaling between -1 (lower) and 0 (best observed)."""
    if "constraint" not in data:
        data["constraint"] = "unknown"
    
    lookup = data.set_index(["framework", "task", "constraint"]).sort_index()
    oracle = data.groupby(["task", "constraint"]).max().sort_index()
    
    for index, row in data.sort_values(["task"]).iterrows():
        task, constraint = row["task"], row["constraint"]
        lb = lookup.loc[(lower, task, constraint)].result
        ub = oracle.loc[(task, constraint)].result
        if lb == ub:
            data.loc[index, "rescaled"] = float("nan")
        else:
            v = -((row["result"] - lb) / (ub - lb)) + 1
            data.loc[index, "scaled"] = v

    return data
            
    
    
# def scaled_result(results: pd.DataFrame, low: str = "RandomForest") -> pd.DataFrame:
#     """Adds `scaled` column which has result scaled from -1 (low) and 0 (best known result) for the (task, fold, constraint)-combination."""
#     lookup = data.set_index(["framework", "task", "constraint"]).sort_index()
#     oracle = data.groupby(["task", "constraint"]).max().sort_index()
    
#     for index, row in data.sort_values(["task"]).iterrows():
#         task, constraint = row["task"], row["constraint"]
#         lb = lookup.loc[(lower, task, constraint)].result
#         ub = oracle.loc[(task, constraint)].result
#         if lb == ub:
#             data.loc[index, "rescaled"] = float("nan")
#         else:
#             v = -((row["result"] - lb) / (ub - lb)) + 1
#             data.loc[index, "rescaled"] = v
#     return data
            