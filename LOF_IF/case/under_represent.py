#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:23:16 2023

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

# group_a and group_b have respectively 1000 data
# decrease the size of group b anomaly

# setting 
seed = list(range(123, 133))
method = "LOF" # LOF / Isolation Forest
data_type = "cluster" # cluster / scatter

dimension = 5
inflate_list = list(range(10,10+5*3,3))

beta_u = [ 0.01, 0.05, 0.10, 0.2, 0.4,  0.6, 0.8]

#initialize
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
    # print(seed[e])
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
    
    # auroc
    auroc_list = []
    
    for i in range(len(beta_u)):
        num_a = 1000
        num_b = 1000
        base_ratea = 0.1
        base_rateb = 0.1
        
        # create data
        if data_type == "cluster":
            # cluster data
            df = cluster_data.Data(num_a, num_b, base_ratea, base_rateb, dimension, seed[e])
        else:   
            # scatter data
            df = scatter_data.Data(num_a, num_b, base_ratea, base_rateb, dimension,  inflate_list, seed[e])
           
        # flag rate
        y_flag_whole.append(df.total_criminal/(num_a + num_b))
        
        
        # bias: random drop beta_u
        df.under_representation(beta_u[i])
        
        if method == "LOF":
            #LOF
            neighbor = df.decide_neighbors(list(range(df.total_criminal-50, df.total_criminal+50, 10)))
            neighbor_list = neighbor_list + [neighbor]
            df.LOF(neighbor)
            violin_title = "LOF"  
            auroc_list.append(roc_auc_score(df.data_pred['Y'], df.data_pred['outlier_score']))
        # Isolation Forest
        else:
            df.iForest()
            violin_title = "Isolation Forest"  
            auroc_list.append(roc_auc_score(df.data_pred['Y'], df.data_pred['outlier_score']))
        
        # number of criminal in each group: [(criminal_a, criminal_b)]
        y_criminal = y_criminal + [(((df.data_pred["Y"] == 1) & (df.data_pred["group"] == 0)).sum(), 
                                   ((df.data_pred["Y"] == 1) & (df.data_pred["group"] == 1)).sum())]

        
        #violin graph
        if e == 0:
            temp = df.data_pred.copy()
            temp['rate'] = beta_u[i]
            violin_df = pd.concat([violin_df, temp], axis=0)
            
            # for true anomaly
            t = df.data_pred[df.data_pred['Y'] == 1]
            t['rate'] = beta_u[i]
            violin_true = pd.concat([violin_true, t], axis=0)
            
            
        #flagrate
        y_flag_ratio.append(matrix.flag_rate_ratio(df.data_pred))
        a, b = matrix.flag_rate(df.data_pred)
        y_flag_a.append(a)
        y_flag_b.append(b)

        #true positive rate
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

axis_title = r'$\beta_u$'
 
# prepare group b base rate list
base_rateb = []
for i in beta_u:
    base_rateb.append(100*(1-i)/1000)
base_ratea = [base_ratea] * len(beta_u)

matrix_plot.auroc(auroc, beta_u, axis_title)


matrix_plot.plot_line_final(axis_title, beta_u, flag_rate_a, flag_rate_b, base_ratea, base_rateb, recall_a, recall_b, tpr_whole, fpr_a, fpr_b, fpr_whole, ppr_a, ppr_b, ppr_whole)


bias_type = "under representation"


# save.save(bias_type, data_type, method, beta_u, 
#           base_ratea, base_rateb, flag_rate_a, flag_rate_b, flag_rate_whole,
#           recall_a, recall_b, tpr_whole, fpr_a, fpr_b, fpr_whole,
#           ppr_a, ppr_b, ppr_whole, auroc)
