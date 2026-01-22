import math
from multiprocessing import Pool, RLock, freeze_support, cpu_count
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CDrift Approaches
from cdrift.approaches import earthmover, bose, martjushev, lcdd
#Maaradji
from cdrift.approaches import maaradji as runs
# Zheng
from cdrift.approaches.zheng import applyMultipleEps
#Process Graph CPD
from cdrift.approaches import process_graph_metrics as pm
# Complexity Drift Detection
from cdrift.approaches import complexity_drift_detection

# Helper functions and evaluation functions
from cdrift import evaluation
from cdrift.utils import helpers

#Misc
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from itertools import product
import yaml

#################################
############ HELPERS ############
#################################

def calcDurationString(startTime, endTime):
    """
        Formats start and endtime to duration in hh:mm:ss format
    """
    elapsed_time = math.floor(endTime - startTime)
    return datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')

def calcDurFromSeconds(seconds):
    """
        Formats ellapsed seconds into hh:mm:ss format
    """
    seconds = math.floor(seconds)
    return datetime.strftime(datetime.utcfromtimestamp(seconds), '%H:%M:%S')

def plotPvals(pvals, changepoints, actual_changepoints, path, xlabel="", ylabel="", autoScale:bool=False):
    """
        Plots a series of p-values with detected and known change points and saves the figure
        args:
            - pvals
                List or array of p-values to be plotted
            - changepoints
                List of indices where change points were detected
            - actual_changepoints
                List of indices of actual change points
            - path
                The savepath of the generated image
            - xlabel
                Label of x axis
            - ylabel
                Label of y axis
            - autoScale
                Boolean whether y axis should autoscale by matplotlib (True) or be limited (0,max(pvals)+0.1) (False)
    """
    # Plotting Configuration
    fig = plt.figure(figsize=(10,4))
    plt.plot(pvals)
    # Not hardcoded 0-1 because of earthmovers distance (and +.1 so 1 is also drawn)
    if not autoScale:
        plt.ylim(0,max(pvals)+.1)
    for cp in changepoints:
        plt.axvline(x=cp, color='red', alpha=0.5)
    for actual_cp in actual_changepoints:
        plt.axvline(x=actual_cp, color='gray', alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{path}")
    plt.close()

#################################
##### Evaluation Functions ######
#################################

def testBose(filepath, window_size, step_size, F1_LAG, cp_locations, do_j:bool=True, do_wc:bool=True, position=None, show_progress_bar=True):
    j_dur = 0
    wc_dur = 0

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]
    entries = []

    if do_j:
        j_start = default_timer()
        pvals_j = bose.detectChange_JMeasure_KS_Step(log, window_size, step_size=step_size, show_progress_bar=show_progress_bar, progressBarPos=position)
        cp_j = bose.visualInspection_Step(pvals_j, window_size, step_size)
        j_dur = default_timer() - j_start

        durStr_J = calcDurFromSeconds(j_dur)
        new_entry_j = {
            'Algorithm':"Bose J",
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'Window Size': window_size,
            'SW Step Size': step_size,
            'Detected Changepoints': cp_j,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=cp_j, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_j, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr_J,
            'Duration (Seconds)': j_dur,
            'Seconds per Case': j_dur / len(log)
        }
        entries.append(new_entry_j)

    if do_wc:
        wc_start = default_timer()
        pvals_wc = bose.detectChange_WC_KS_Step(log, window_size, step_size=step_size, show_progress_bar=show_progress_bar, progressBarPos=position)
        cp_wc = bose.visualInspection_Step(pvals_wc, window_size, step_size)
        wc_dur = default_timer() - wc_start

        durStr_WC = calcDurFromSeconds(wc_dur)
        new_entry_wc = {
            'Algorithm':"Bose WC", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'Window Size': window_size,
            'SW Step Size': step_size,
            'Detected Changepoints': cp_wc,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=cp_wc, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_wc, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr_WC,
            'Duration (Seconds)': wc_dur,
            'Seconds per Case': wc_dur / len(log)
        }
        entries.append(new_entry_wc)

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame(entries).to_csv(Path("Reproducibility_Intermediate_Results", "Bose",f"{logname}_WIN{window_size}.csv"), index=False)

    return entries

def testMartjushev(filepath, window_size, F1_LAG, cp_locations, do_j:bool=True, do_wc:bool=True, position=None, show_progress_bar=True):
    PVAL = 0.55
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    entries = []

    if do_j:
        j_start = default_timer()
        rb_j_cp = martjushev.detectChange_JMeasure_KS(log, window_size, PVAL, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
        j_dur = default_timer() - j_start

        durStr_J = calcDurFromSeconds(j_dur)
        new_entry_j = {
            'Algorithm':"Martjushev J", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'P-Value': PVAL,
            'Window Size': window_size,
            'Detected Changepoints': rb_j_cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=rb_j_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=rb_j_cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr_J,
            'Duration (Seconds)': j_dur,
            'Seconds per Case': j_dur / len(log)
        }
        entries.append(new_entry_j)

    if do_wc:
        wc_start = default_timer()
        rb_wc_cp = martjushev.detectChange_WindowCount_KS(log, window_size, PVAL, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
        wc_dur = default_timer() - wc_start

        durStr_WC = calcDurFromSeconds(wc_dur)
        new_entry_wc = {
            'Algorithm':"Martjushev WC", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'P-Value': PVAL,
            'Window Size': window_size,
            'Detected Changepoints': rb_wc_cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=rb_wc_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=rb_wc_cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr_WC,
            'Duration (Seconds)': wc_dur,
            'Seconds per Case': wc_dur / len(log)
        }
        entries.append(new_entry_wc)

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame(entries).to_csv(Path("Reproducibility_Intermediate_Results", "Martjushev", f"{logname}_WIN{window_size}.csv"), index=False)
    return entries

def testMartjushev_ADWIN(filepath, min_max_window_pair, pvalue, step_size, F1_LAG, cp_locations, do_j:bool=True, do_wc:bool=True, position=None, show_progress_bar=True):
    log = helpers.importLog(filepath, verbose=False)

    min_window, max_window = min_max_window_pair

    if len(log) <= min_window:
        # If the log is too short, we can't use the ADWIN algorithm because even the initial windows do not fit
        return np.NaN

    logname = filepath.split('/')[-1].split('.')[0]

    entries = []

    if do_j:
        j_start = default_timer()
        adwin_j_cp = martjushev.detectChange_ADWIN_JMeasure_KS(log, min_window, max_window, pvalue, step_size, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
        j_dur = default_timer() - j_start

        durStr_J = calcDurFromSeconds(j_dur)
        new_entry_j = {
            'Algorithm':"Martjushev ADWIN J", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'P-Value': pvalue,
            'Min Adaptive Window': min_window,
            'Max Adaptive Window': max_window,
            'ADWIN Step Size': step_size,
            'Detected Changepoints': adwin_j_cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=adwin_j_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=adwin_j_cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr_J,
            'Duration (Seconds)': j_dur,
            'Seconds per Case': j_dur / len(log)
        }
        entries.append(new_entry_j)

    if do_wc:
        wc_start = default_timer()
        adwin_wc_cp = martjushev.detectChange_ADWIN_WindowCount_KS(log, min_window, max_window, pvalue, step_size, return_pvalues=False, show_progress_bar=show_progress_bar, progressBarPos=position)
        wc_dur = default_timer() - wc_start

        durStr_WC = calcDurFromSeconds(wc_dur)
        new_entry_wc = {
            'Algorithm':"Martjushev ADWIN WC", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'P-Value': pvalue,
            'Min Adaptive Window': min_window,
            'Max Adaptive Window': max_window,
            'ADWIN Step Size': step_size,
            'Detected Changepoints': adwin_wc_cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=adwin_wc_cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=adwin_wc_cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr_WC,
            'Duration (Seconds)': wc_dur,
            'Seconds per Case': wc_dur / len(log)
        }
        entries.append(new_entry_wc)

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame(entries).to_csv(Path("Reproducibility_Intermediate_Results", "Martjushev ADWIN", f"{logname}_MINW{min_window}_MAXW{max_window}.csv"), index=False)

    return entries

def testEarthMover(filepath, window_size, step_size, F1_LAG, cp_locations, position, show_progress_bar=True):
    LINE_NR = position

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    # Earth Mover's Distance
    traces = earthmover.extractTraces(log)
    # em_dists = earthmover.calculateDistSeries(traces, window_size, show_progressBar=show_progress_bar, progressBar_pos=LINE_NR)

    # cp_em = earthmover.visualInspection(em_dists, window_size)

    cp_em = earthmover.detect_change(log, window_size, step_size, show_progress_bar=show_progress_bar, progress_bar_pos=LINE_NR)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    new_entry = {
        'Algorithm':"Earth Mover's Distance", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'SW Step Size': step_size,
        'Detected Changepoints': cp_em,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_em, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_em, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr,
        'Duration (Seconds)': (endTime-startTime),
        'Seconds per Case': (endTime-startTime) / len(log)
    }

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", "Earthmover", f"{logname}_WIN{window_size}.csv"), index=False)

    return [new_entry]

def testMaaradji(filepath, window_size, step_size, F1_LAG, cp_locations, position, show_progress_bar=True):

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    # cp_runs = runs.detectChangepoints(log,window_size, pvalue=0.05, return_pvalues=False, show_progress_bar=show_progress_bar,progressBar_pos=position)
    cp_runs = runs.detectChangepoints_Stride(log, window_size, step_size, pvalue=0.05, return_pvalues=False, show_progress_bar=show_progress_bar, progressBar_pos=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    new_entry = {
        'Algorithm':"Maaradji Runs",
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'SW Step Size': step_size,
        'Detected Changepoints': cp_runs,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_runs, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_runs, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr,
        'Duration (Seconds)': (endTime-startTime),
        'Seconds per Case': (endTime-startTime) / len(log)
    }
    
    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", "Maaradji", f"{logname}_WIN{window_size}.csv"), index=False)
    return [new_entry]

def testGraphMetrics(filepath, min_max_window_pair, pvalue, F1_LAG, cp_locations, position=None, show_progress_bar=True):
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    min_window, max_window = min_max_window_pair

    startTime = default_timer()

    cp = pm.detectChange(log, min_window, max_window, pvalue=pvalue, show_progress_bar=show_progress_bar,progressBarPosition=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    new_entry = {
        'Algorithm':"Process Graph Metrics", 
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'P-Value': pvalue,
        'Min Adaptive Window': min_window,
        'Max Adaptive Window': max_window,
        'Detected Changepoints': cp,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr,
        'Duration (Seconds)': (endTime-startTime),
        'Seconds per Case': (endTime-startTime) / len(log)
    }

    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", "ProcessGraph", f"{logname}_P{pvalue}_MINW{min_window}_MAXW{max_window}.csv"), index=False)
    return [new_entry]

def testZhengDBSCAN(filepath, mrid, eps_modifiers, F1_LAG, cp_locations, position, show_progress_bar=True):
    # candidateCPDetection is independent of eps, so we can calculate the candidates once and use them for multiple eps!
    epsList = [mrid*meps for meps in eps_modifiers]


    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()
    
    # CPD #
    cps = applyMultipleEps(log, mrid=mrid, epsList=epsList, show_progress_bar=show_progress_bar, progressPos=position)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    ret = []
    for eps in epsList:
        cp = cps[eps]

        new_entry = {
            'Algorithm':"Zheng DBSCAN", 
            'Log Source': Path(filepath).parent.name,
            'Log': logname,
            'MRID': mrid,
            'Epsilon': eps,
            'Detected Changepoints': cp,
            'Actual Changepoints for Log': cp_locations,
            'F1-Score': evaluation.F1_Score(detected=cp, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
            'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp, actual_changepoints=cp_locations, lag=F1_LAG),
            'Duration': durStr,
            'Duration (Seconds)': (endTime-startTime),
            'Seconds per Case': (endTime-startTime) / len(log)
        }
        ret.append(new_entry)
    if os.path.exists("Reproducibility_Intermediate_Results"):
        for entry in ret:
            pd.DataFrame([entry]).to_csv(Path("Reproducibility_Intermediate_Results", "Zheng", f"{logname}_MRID{mrid}_EPS{str(entry['Epsilon']).replace('.','_')}.csv"), index=False)
    return ret

def testLCDD(filepath, window_pairs, stable_period, F1_LAG, cp_locations, position, show_progress_bar=True):

    complete_window_size, detection_window_size = window_pairs

    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    cp_lcdd = lcdd.calculate(log, complete_window_size, detection_window_size, stable_period)

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #

    new_entry = {
        'Algorithm':"LCDD",
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Complete-Window Size': complete_window_size,
        'Detection-Window Size': detection_window_size,
        'Stable Period': stable_period,
        'Detected Changepoints': cp_lcdd,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_lcdd, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_lcdd, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr,
        'Duration (Seconds)': (endTime-startTime),
        'Seconds per Case': (endTime-startTime) / len(log)
    }
    
    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", "LCDD", f"{logname}_CW{complete_window_size}_DW{detection_window_size}_SP{stable_period}.csv"), index=False)
    return [new_entry]

def testComplexityDriftDetection(filepath, window_size, penalty, complexity_metric, min_consecutive_windows, F1_LAG, cp_locations, position=None, show_progress_bar=True):
    """Test ComplexityDriftDetection approach.

    Parameters
    ----------
    filepath : str
        Path to the event log file.
    window_size : int
        Size of non-overlapping windows.
    penalty : float
        Penalty parameter for PELT algorithm.
    complexity_metric : str
        Name of the complexity metric to use.
    min_consecutive_windows : int
        Minimum number of consecutive windows for PELT.
    F1_LAG : int
        Lag window for F1 score calculation.
    cp_locations : List[int]
        List of actual change point locations.
    position : int, optional
        Position for progress bar. Default is None.
    show_progress_bar : bool, optional
        Whether to show progress bar. Default is True.

    Returns
    -------
    List[dict]
        List of result entry dictionaries.
    """
    log = helpers.importLog(filepath, verbose=False)
    logname = filepath.split('/')[-1].split('.')[0]

    startTime = default_timer()

    # Detect change points using complexity-based approach
    # Use logname as log_id for better caching
    cp_detected = complexity_drift_detection.detect_change(
        log=log,
        window_size=window_size,
        penalty=penalty,
        complexity_metric=complexity_metric,
        min_consecutive_windows=min_consecutive_windows,
        log_id=logname,  # Use log name for caching
    )

    endTime = default_timer()
    durStr = calcDurationString(startTime, endTime)

    # Save Results #
    new_entry = {
        'Algorithm': "ComplexityDriftDetection",
        'Log Source': Path(filepath).parent.name,
        'Log': logname,
        'Window Size': window_size,
        'Penalty': penalty,
        'Complexity Metric': complexity_metric,
        'Min Consecutive Windows': min_consecutive_windows,
        'Detected Changepoints': cp_detected,
        'Actual Changepoints for Log': cp_locations,
        'F1-Score': evaluation.F1_Score(detected=cp_detected, known=cp_locations, lag=F1_LAG, zero_division=np.NaN),
        'Average Lag': evaluation.get_avg_lag(detected_changepoints=cp_detected, actual_changepoints=cp_locations, lag=F1_LAG),
        'Duration': durStr,
        'Duration (Seconds)': (endTime-startTime),
        'Seconds per Case': (endTime-startTime) / len(log)
    }
    
    if os.path.exists("Reproducibility_Intermediate_Results"):
        pd.DataFrame([new_entry]).to_csv(Path("Reproducibility_Intermediate_Results", "ComplexityDriftDetection", f"{logname}_WIN{window_size}_PEN{penalty}_MET{complexity_metric.replace(' ', '_')}_MIN{min_consecutive_windows}.csv"), index=False)
    return [new_entry]

def callFunction(arg):
    """Wrapper for testing functions, as for the multiprocessing pool, one can only use one function, not multiple

    Args:
        idx (int): Position-Index for the progress bar of the evaluation
        vals (Tuple[str,List]): Tuple of name of the approach, and its parameter values
    """
    funcname, args = arg
    return globals()[funcname](**args)

def get_logpaths_with_changepoints():
    # Setup all Paths to logs alongside their change point locations
    logPaths_Changepoints = [
        (Path("EvaluationLogs","Bose", "bose_log.xes.gz").as_posix(), [1199, 2399, 3599, 4799]), # A change every 1200 cases, 6000 cases in total (Skipping 5999 because a change on the last case doesnt make sense)
    ]

    ceravolo_root = Path("EvaluationLogs","Ceravolo")
    for item in ceravolo_root.iterdir():
        _, _, _,num_cases, _ = item.stem.split("_")
        if int(num_cases) != 1000: # Only use logs of length 1000
            continue
        drift_indices = [(int(num_cases)//2) - 1] # "The first half of the stream is composed of the baseline model, and the second half is composed of the drifted model"
        logPaths_Changepoints.append((item.as_posix(), drift_indices))
    
    logPaths_Changepoints += [
        (item.as_posix(), [999,1999])
        for item in Path("EvaluationLogs","Ostovar").iterdir()
    ]

    # Add Kraus logs if available
    kraus_root = Path("EvaluationLogs","Kraus")
    gold_standard_path = kraus_root / "gold_standard.csv"
    if kraus_root.exists() and gold_standard_path.exists():
        try:
            gold_standard = pd.read_csv(gold_standard_path)
            # Parse change points from the CSV
            for _, row in gold_standard.iterrows():
                log_name = row['log_name']
                # Find the corresponding log file
                log_file = kraus_root / log_name
                if not log_file.exists():
                    # Try with .xes.gz extension if not already present
                    if not log_name.endswith('.xes.gz'):
                        log_file = kraus_root / f"{log_name}.xes.gz"
                
                if log_file.exists():
                    # Parse change_point column (it's a string representation of a list)
                    cp_value = row['change_point']
                    if isinstance(cp_value, str):
                        try:
                            change_points = eval(cp_value)  # Safe for list literals from CSV
                        except:
                            # Try parsing as comma-separated values
                            change_points = [int(x.strip()) for x in cp_value.strip('[]').split(',') if x.strip()]
                    elif isinstance(cp_value, (list, tuple)):
                        change_points = list(cp_value)
                    else:
                        change_points = []
                    
                    logPaths_Changepoints.append((log_file.as_posix(), change_points))
        except Exception as e:
            print(f"Warning: Could not load Kraus logs from gold_standard.csv: {e}")
            # Fallback: add all .xes.gz files in Kraus directory without change points
            for item in kraus_root.glob("*.xes.gz"):
                if "gold_standard" not in item.name:
                    logPaths_Changepoints.append((item.as_posix(), []))
    elif kraus_root.exists():
        # Kraus directory exists but no gold_standard.csv - add all logs without change points
        for item in kraus_root.glob("*.xes.gz"):
            if "gold_standard" not in item.name:
                logPaths_Changepoints.append((item.as_posix(), []))

    return logPaths_Changepoints


def build_arguments_list(config, logPaths_Changepoints, is_test_run=False):
    _args = { approach["function"]: (approach.get("meta-params", dict()), approach["params"]) for approach in config["approaches"].values() if approach.get("enabled", True) == True }

    arguments = []
    for funcname, (meta_args, args) in _args.items():
        keys, values = zip(*args.items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]

        if is_test_run:
            arguments += [(funcname, permutation | meta_args) for permutation in permutations_dicts[:1]]
        else:
            arguments += [(funcname, permutation | meta_args) for permutation in permutations_dicts]

    if is_test_run:
        logPaths_Changepoints = logPaths_Changepoints[:1]

    meta_args = [ # Arguments that all functions take.
        {
            "F1_LAG": config["meta-parameters"]["F1_LAG"], # For the per-instance-F1-Score. Not relevant for evaluation anymore.
            "filepath": logpath, # Path to the event log
            "cp_locations": cp_locations, # List of indices of the changepoints in this event log
            "show_progress_bar": not config["meta-parameters"]["DO_SINGLE_BAR"]
        }
        for logpath, cp_locations in logPaths_Changepoints
    ]

    arguments = [
        (funcname, arg_dict | meta_arg)
        for funcname, arg_dict in arguments
        for meta_arg in meta_args
    ]
    # Shuffle the Tasks
    np.random.shuffle(arguments)
    # Give each task an index for progress bar (only used if DO_SINGLE_BAR is False)
    arguments = [
        (funcname, d | {"position": idx})
        for idx, (funcname, d) in enumerate(arguments)
    ]
    return arguments


def main(test_run:bool = False, num_cores:int = None):
    if num_cores is None:
        num_cores = cpu_count() - 2

    logPaths_Changepoints = get_logpaths_with_changepoints()

    ## Load the Arguments from testAll_config.yml ##
    config = None
    with open("testAll_config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    arguments = build_arguments_list(config, logPaths_Changepoints, is_test_run=test_run)

    ## Set up File Structure
    for approach, approach_config in config["approaches"].items():
        if approach_config["enabled"] == True:
            Path("Reproducibility_Intermediate_Results", approach).mkdir(parents=True, exist_ok=True)

    ## Run all experiments using multiprocessing ##
    time_start = default_timer()
    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    results = []
    with Pool(num_cores,initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        if config["meta-parameters"]["DO_SINGLE_BAR"]:
            for result in tqdm(p.imap(callFunction, arguments), desc="Calculating.. Completed PCD Instances", total=len(arguments)):
                results.append(result)
        else:
            results = p.map(callFunction, arguments)

    # Remove NaN return values from the results, source is Martjushev_ADWIN if the log is too short for the chosen windows
    results = [result for result in results if not result == np.NaN]

    elapsed_time = math.floor(default_timer() - time_start)
    # Write instead of print because of progress bars (although it shouldnt be a problem because they are all done)
    elapsed_formatted = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
    tqdm.write(f"The execution took {elapsed_formatted}")


    flattened_results = [res for function_return in results for res in function_return]
    df = pd.DataFrame(flattened_results)
    df.to_csv("algorithm_results.csv", index=False)

def main_kraus_only(test_run:bool = False, num_cores:int = None):
    """Run evaluation only for ComplexityDriftDetection on Kraus logs.
    
    Parameters
    ----------
    test_run : bool, optional
        If True, use test run mode (fewer parameter combinations). Default is False.
    num_cores : int, optional
        Number of CPU cores to use. Default is min(4, cpu_count() - 2) to avoid memory issues on Windows.
    """
    if num_cores is None:
        # Use fewer cores on Windows to avoid paging file issues
        num_cores = min(4, max(1, cpu_count() - 2))
    
    print(f"Using {num_cores} CPU core(s) for multiprocessing.")
    if num_cores > 4:
        print("Warning: Using many cores may cause memory issues on Windows. Consider using --cores 2 or --cores 1 if you encounter paging file errors.")

    # Get all log paths, then filter to only Kraus logs
    all_logPaths_Changepoints = get_logpaths_with_changepoints()
    logPaths_Changepoints = [
        (logpath, cp_locations)
        for logpath, cp_locations in all_logPaths_Changepoints
        if "Kraus" in logpath
    ]
    
    if len(logPaths_Changepoints) == 0:
        print("Warning: No Kraus logs found. Make sure EvaluationLogs/Kraus/ directory exists with log files.")
        return

    print(f"Found {len(logPaths_Changepoints)} Kraus log(s) to process.")

    ## Load the Arguments from testAll_config.yml ##
    config = None
    with open("testAll_config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    # Filter config to only ComplexityDriftDetection
    filtered_config = {
        "approaches": {
            "ComplexityDriftDetection": config["approaches"]["ComplexityDriftDetection"]
        },
        "meta-parameters": config["meta-parameters"]
    }
    
    arguments = build_arguments_list(filtered_config, logPaths_Changepoints, is_test_run=test_run)

    ## Set up File Structure
    Path("Reproducibility_Intermediate_Results", "ComplexityDriftDetection").mkdir(parents=True, exist_ok=True)

    ## Run all experiments using multiprocessing ##
    time_start = default_timer()
    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention
    results = []
    with Pool(num_cores,initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        if filtered_config["meta-parameters"]["DO_SINGLE_BAR"]:
            for result in tqdm(p.imap(callFunction, arguments), desc="Calculating.. Completed PCD Instances", total=len(arguments)):
                results.append(result)
        else:
            results = p.map(callFunction, arguments)

    # Remove NaN return values from the results
    results = [result for result in results if not result == np.NaN]

    elapsed_time = math.floor(default_timer() - time_start)
    # Write instead of print because of progress bars (although it shouldnt be a problem because they are all done)
    elapsed_formatted = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
    tqdm.write(f"The execution took {elapsed_formatted}")

    flattened_results = [res for function_return in results for res in function_return]
    df = pd.DataFrame(flattened_results)
    output_file = "algorithm_results_kraus_complexity_drift.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Total results: {len(flattened_results)}")
    print(f"Columns: {list(df.columns)}")

if __name__ == '__main__':
    import sys
    # Check if --kraus-only flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == '--kraus-only':
        # Check for --cores argument
        num_cores = None
        if len(sys.argv) > 2 and sys.argv[2] == '--cores' and len(sys.argv) > 3:
            try:
                num_cores = int(sys.argv[3])
            except ValueError:
                print(f"Warning: Invalid number of cores '{sys.argv[3]}', using default")
        main_kraus_only(num_cores=num_cores)
    else:
        main()