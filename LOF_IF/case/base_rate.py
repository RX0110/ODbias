#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:17:50 2023

@author: hahaha
"""
import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(__file__)
od_bias_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(od_bias_dir)
print(sys.path)

import utils.cluster_data as cluster_data
import utils.scatter_data as scatter_data
import utils.matrix as matrix
import utils.matrix_plot as matrix_plot
import utils.save as save
import pandas as pd
from sklearn.metrics import roc_auc_score

# group a and group b have data respective 1000
# change base rate ratio, while holding total number of criminals same

# setting
seed = list(range(123, 133))
method = "LOF" # LOF / Isolation Forest
data_type = "cluster" # cluster / scatter

num_a = 1000
num_b = 1000
total = 100
inflate_list = list(range(10,10+5*3,3)) 

# proprotion of criminal in each group: group_a:always 1. group_b:1,2,3,4
prop_b = list(range(1,5))

# compute base_rate
base_rate_a = []
base_rate_b = []
for p in prop_b:
    base_rate_a += [(total * 1 /(1+p))/num_a]
    base_rate_b += [(total * p /(1+p))/num_b]
# base_rate_a: [0.05, 0.03333333333333333, 0.025, 0.02]
# base_rate_b: [0.05, 0.06666666666666667, 0.075, 0.08]
print(base_rate_a)
dimension = 5


# initialize
tpr = []
fpr = []
ppr = []

tpr_whole = []
fpr_whole = []
ppr_whole = []

flag_ratio = [] 
true_positive_ratio = []
fpr_ratio = []
ppr_ratio = []

flag_rate_a = []
flag_rate_b = []

recall_a = []
recall_b = []

fpr_a = []
fpr_b = []

ppr_a = []
ppr_b = []

flag_rate_whole = []

num_experiments = 10

violin_df = pd.DataFrame()
violin_true = pd.DataFrame()

auroc = []

# execute 10 time
for e in range(len(seed)):
    y_tpr = []
    y_fpr = []
    y_ppr = []
    
    y_flag_ratio = [] 
    y_true_positive_ratio = []
    y_fpr_ratio = []
    y_ppr_ratio = []
    
    #flag
    y_flag_a = []
    y_flag_b = []
    y_flag_whole = []
    

    #false positive 
    y_fpr_a = []
    y_fpr_b = []
    
    #true positive
    y_recall_a = []
    y_recall_b = []
    
    #percision
    y_ppr_a = []
    y_ppr_b = []
    
    y_criminal = []
    neighbor_list = []
    
    auroc_list = []
    
    for i in range(len(base_rate_a)):
        # create data
        if data_type == "cluster":
            # cluster data
            df = cluster_data.Data(num_a, num_b, base_rate_a[i], base_rate_b[i], dimension, seed[e])
        else:
            # scatter data
            df = scatter_data.Data(num_a, num_b, base_rate_a[i], base_rate_b[i], dimension, inflate_list, seed[e])
            
        # flag rate
        y_flag_whole.append(df.total_criminal/(num_a + num_b))
        
        # algorithm
        if method == "LOF":
            #LOF
            # find optimal hyperparameters
            neighbor = df.decide_neighbors(list(range(df.total_criminal-50, df.total_criminal+50, 10)))
            neighbor_list = neighbor_list + [neighbor]
            # LOF
            df.LOF(neighbor)
            violin_title = "LOF"  
            auroc_list.append(roc_auc_score(df.data_pred['Y'], df.data_pred['outlier_score']))
        # Isolation Forest
        else:
            df.iForest()
            violin_title = "Isolation Forest"  
            auroc_list.append(roc_auc_score(df.data_pred['Y'], df.data_pred['outlier_score']))
    
        if e == 0:
            # select = ['group', 'outlier_score']
            # temp = df.data_pred[select]
            temp = df.data_pred.copy()
            temp['rate'] = prop_b[i]
            violin_df = pd.concat([violin_df, temp], axis=0)
            
            t = df.data_pred[df.data_pred['Y'] == 1]
            t['rate'] = prop_b[i]
            violin_true = pd.concat([violin_true, t], axis=0)
            
            
        #flagrate
        y_flag_ratio.append(matrix.flag_rate_ratio(df.data_pred))
        a, b = matrix.flag_rate(df.data_pred)
        y_flag_a.append(a)
        y_flag_b.append(b)

        #equalize odd
        y_true_positive_ratio.append(matrix.true_positive_rate_ratio(df.data_pred))
        r_a, r_b = matrix.true_positive_rate(df.data_pred)
        y_recall_a.append(r_a)
        y_recall_b.append(r_b)
        y_tpr.append(matrix.true_positive_rate_whole(df.data_pred))

        #false positive rate
        y_fpr_ratio.append(matrix.false_positive_rate_ratio(df.data_pred))
        a, b = matrix.false_positive_rate(df.data_pred)
        y_fpr_a.append(a)
        y_fpr_b.append(b)
        y_fpr.append(matrix.false_positive_rate_whole(df.data_pred))


        #positive predictive value rate
        y_ppr_ratio.append(matrix.positive_predict_ratio(df.data_pred))
        a, b = matrix.positive_predict(df.data_pred)
        y_ppr_a.append(a)
        y_ppr_b.append(b)
        y_ppr.append(matrix.positive_predict_whole(df.data_pred))
        
        if e == 0:
            # matrix_plot.plot_data(violin_df, prop_b[i])
            rate_data = violin_df[violin_df['rate'] == prop_b[i]]
            groupa_anomaly = ((rate_data["y_pred"] == 1) & (rate_data['group'] == 0)).sum()
            groupb_anomaly = ((rate_data["y_pred"] == 1) & (rate_data['group'] == 1)).sum()
            anomaly_preAnomaly = ((rate_data["y_pred"] == 1) & (rate_data['Y'] == 1)).sum()
            normal_preNormal = ((rate_data["y_pred"] == 1) & (rate_data['Y'] == 0)).sum()
            groupa_bothAnomaly = ((rate_data["y_pred"] == 1) & (rate_data['group'] == 0) & (rate_data['Y'] == 1)).sum()
            groupb_bothAnomaly = ((rate_data["y_pred"] == 1) & (rate_data['group'] == 1) & (rate_data['Y'] == 1)).sum()
        
    tpr_whole = tpr_whole + [y_tpr]
    fpr_whole = fpr_whole + [y_fpr]
    ppr_whole = ppr_whole + [y_ppr]
    
    flag_ratio = flag_ratio + [y_flag_ratio]
    true_positive_ratio = true_positive_ratio + [y_true_positive_ratio]
    fpr_ratio = fpr_ratio + [y_fpr_ratio]
    ppr_ratio = ppr_ratio + [y_ppr_ratio]
    
    flag_rate_a = flag_rate_a + [y_flag_a]
    flag_rate_b = flag_rate_b + [y_flag_b]
    recall_a = recall_a + [y_recall_a]
    recall_b = recall_b + [y_recall_b]
    fpr_a = fpr_a + [y_fpr_a]
    fpr_b = fpr_b + [y_fpr_b]
    ppr_a = ppr_a + [y_ppr_a]
    ppr_b = ppr_b + [y_ppr_b]
    flag_rate_whole = flag_rate_whole + [y_flag_whole]
    auroc = auroc + [auroc_list]

## # x-axis    
# x_value = [round(num/dimension,2) for num in x]
tpr_whole_mean = matrix.get_mean(tpr_whole)
fpr_whole_mean = matrix.get_mean(fpr_whole)
ppr_whole_mean = matrix.get_mean(ppr_whole)

flag_ratio_mean = matrix.get_mean(flag_ratio)
true_positive_mean = matrix.get_mean(true_positive_ratio)
fpr_ratio_mean = matrix.get_mean(fpr_ratio)
ppr_ratio_mean = matrix.get_mean(ppr_ratio)

flag_ratio_var = matrix.get_var(flag_ratio_mean, flag_ratio)
true_positive_var = matrix.get_var(true_positive_mean, true_positive_ratio)
fpr_ratio_var = matrix.get_var(fpr_ratio_mean, fpr_ratio)
ppr_ratio_var = matrix.get_var(ppr_ratio_mean, ppr_ratio)


# plot 
axis_title = r'$\beta_b$'
violin_title = method
matrix_plot.plot_line_final(axis_title, prop_b, flag_rate_b, flag_rate_a, base_rate_b, base_rate_a, recall_b, recall_a, tpr_whole, fpr_b, fpr_a, fpr_whole, ppr_b, ppr_a, ppr_whole)
matrix_plot.auroc(auroc, prop_b, axis_title)

bias_type = "base rate change"


# save.save(bias_type, data_type, method, prop_b, 
#           base_rate_a, base_rate_b, flag_rate_a, flag_rate_b, flag_rate_whole,
#           recall_a, recall_b, tpr_whole, fpr_a, fpr_b, fpr_whole,
#           ppr_a, ppr_b, ppr_whole, auroc)
        
        
        