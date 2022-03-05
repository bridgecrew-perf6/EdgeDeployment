#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 18:51:52 2022

@author: msweber
"""

import pandas as pd
import numpy as np
import pickle


results_dict = {
    "ModelName": [],
    "Precision": [],
    "Device": [],
    "Avg_Inf_Time": [],
    "Accuracy": []
    }

with open("Tensorbook_results.p", "rb") as f:
    tensorbook_results = pickle.load(f)
    
for key, value in tensorbook_results.items():
    q, w = key.split("/")
    p, q2 = q.split("_")
    name, suffix = w.split(".")
    if p == "pb":
        precision = "NOT_OPTIMIZED"
    elif p == 'trt':
        precision = "FP32"
    else:
        precision = p
    results_dict["ModelName"].append(name)
    results_dict["Precision"].append(precision)
    results_dict["Device"].append("Tensorbook")
    results_dict["Avg_Inf_Time"].append(value[0])
    results_dict["Accuracy"].append(value[1])
    
with open("nano_results.p", "rb") as f:
    nano_results = pickle.load(f)
    
for key, value in nano_results.items():
    q, w = key.split("/")
    p, q2 = q.split("_")
    name, suffix = w.split(".")
    if p == "pb":
        precision = "NOT_OPTIMIZED"
    elif p == 'trt':
        precision = "FP32"
    else:
        precision = p
    results_dict["ModelName"].append(name)
    results_dict["Precision"].append(precision)
    results_dict["Device"].append("JetsonNano")
    results_dict["Avg_Inf_Time"].append(value[0])
    results_dict["Accuracy"].append(value[1])
    
    
df = pd.DataFrame.from_dict(results_dict)

df = df.sort_values("ModelName")
df.to_csv("results.csv")
