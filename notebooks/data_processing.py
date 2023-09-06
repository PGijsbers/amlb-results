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


def impute_missing_results(results: pd.DataFrame, with_results_from: str = "constantpredictor", with_indicator: bool = True) -> pd.DataFrame:
    """Imputes missing values in `results` with the corresponding score from `constantpredictor`"""
    if with_results_from not in results["framework"].unique():
        raise ValueError(f"{with_results_from=} is not in `results`")
    results = results.copy()
    if with_indicator:
        results["imputed"] = False
        
    lookup_table = results.set_index(["framework", "task", "fold", "constraint"])
    rows_with_missing_result = ((index, row) for index, row in results.iterrows() if np.isnan(row["result"]))
    for index, row in rows_with_missing_result:
        task, fold, constraint = row[["task", "fold", "constraint"]]
        value = lookup_table.loc[(with_results_from, task, fold, constraint)].result
        results.loc[index, "result"] = value
        if with_indicator:
            results.loc[index, "imputed"] = True
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
            