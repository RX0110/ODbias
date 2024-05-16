# -*- coding: utf-8 -*-

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

# setting
data_type = "cluster" #scatter / cluster
method = "LOF" #Isolation Forest/ LOF

seed = list(range(123, 133))
num_a = 1000
num_b = 1000
base_ratea = 0.1
base_rateb = 0.1

dimension = 5
inflate_list = list(range(10,10+5*3,3))

meanShift = [2*i for i in range(5)]

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
for e in range(num_experiments):
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
    
    for i in range(len(meanShift)):
        # cluster data
        if data_type == "cluster":
            df = cluster_data.Data(num_a, num_b, base_ratea, base_rateb, dimension, seed[e])
        else:   
            # scatter data
            df = scatter_data.Data(num_a, num_b, base_ratea, base_rateb, dimension,  inflate_list, seed[e])
           
       
        # flag rate
        y_flag_whole.append(df.total_criminal/(num_a + num_b))
        
        #shift mean
        df.mean_shift(meanShift)
        
        if method == "LOF":
            # LOF
            neighbor = df.decide_neighbors(list(range(df.total_criminal-190, df.total_criminal+50, 10)))
            neighbor_list = neighbor_list + [neighbor]
            df.LOF(neighbor)
            auroc_list.append(roc_auc_score(df.data_pred['Y'], df.data_pred['y_pred']))
            violin_title = "LOF" 
        else:
            # IForest
            df.iForest()
            auroc_list.append(roc_auc_score(df.data_pred['Y'], df.data_pred['outlier_score']))
            violin_title = "Isolation Forest" 

        
        # number of criminal in each group: [(criminal_a, criminal_b)]
        y_criminal = y_criminal + [(((df.data_pred["Y"] == 1) & (df.data_pred["group"] == 0)).sum(), 
                                   ((df.data_pred["Y"] == 1) & (df.data_pred["group"] == 1)).sum())]

        
        #violin graph
        if e == num_experiments-1:
            select = ['group', 'outlier_score']
            temp = df.data_pred[select]
            temp['rate'] = meanShift[i]
            violin_df = pd.concat([violin_df, temp], axis=0)
            
            t = df.data_pred[df.data_pred['Y'] == 1]
            t['rate'] = meanShift[i]
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

# # x-axis    
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


axis_title = r'$\beta_m$'

# plot
matrix_plot.plot_line_final(axis_title, meanShift, flag_rate_a, flag_rate_b, base_ratea, base_rateb, recall_a, recall_b, tpr_whole, fpr_a, fpr_b, fpr_whole, ppr_a, ppr_b, ppr_whole)
matrix_plot.auroc(auroc, meanShift, axis_title)

bias_type = "mean shift"

# save.save(bias_type, data_type, method, meanShift, 
#           base_ratea, base_rateb, flag_rate_a, flag_rate_b, flag_rate_whole,
#           recall_a, recall_b, tpr_whole, fpr_a, fpr_b, fpr_whole,
#           ppr_a, ppr_b, ppr_whole, auroc)
        