
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')

import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
print(sys.path)
import matrix
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import itertools

def prepareForScatter(l1, l2, x_axis):
    a = list(zip(*l1))
    b = list(zip(*l2))
    d = pd.DataFrame()
    for i in range(len(a)):
        df_x = pd.DataFrame(a[i], columns=["x"]) 
        df_y = pd.DataFrame(b[i], columns=["y"]) 
        df_beta = pd.DataFrame( np.full([len(a[i]), 1], x_axis[i]), columns=["beta"])
        g = pd.concat([df_x, df_y, df_beta], axis = 1)
        d = pd.concat([d, g], axis=0)
    return d

def plot(violin_title, axis_title, x_axis, violin_true, violin_df, 
         flag_ratio_var, flag_ratio_mean, flag_rate_a, flag_rate_b,
         tpr_ratio_var, tpr_ratio_mean, recall_a, recall_b,
         fpr_ratio_var, fpr_ratio_mean, fpr_a, fpr_b,
         ppr_ratio_var, ppr_ratio_mean, ppr_a, ppr_b
         ):
    
    colors = cm.rainbow(np.linspace(0, 1, len(x_axis)))

    #flag rate
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5,2,figsize=(15, 35))
    # for i in range(len(flag)):
    # #     label = f'Experiment {i+1}'
    #     plt.plot(x_value, flag[i], marker='o')
    # # plt.plot(x_value, y_flag, marker='o')
    # # plt.ylim(0,2)
    # # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Calculate upper and lower bounds for the area chart
    upper_bound = flag_ratio_mean + np.sqrt(flag_ratio_var)
    lower_bound = flag_ratio_mean - np.sqrt(flag_ratio_var)
    
    # Create the area chart
    ax1.fill_between(x_axis, lower_bound, upper_bound, color='gray', alpha=0.5, label='Variance Area')
    ax1.plot(x_axis, flag_ratio_mean, marker='o')
    ax1.set_xlabel(axis_title)
    ax1.set_ylabel('flag rate ratio: group_a/group_b')
    ax1.set_title('flag rate ratio')
    
    # for i in range(len(flag_rate_a)):
        # scatter = ax2.scatter(flag_rate_a[i], flag_rate_b[i], marker='o',c = x_axis, alpha=0.5, cmap = 'gist_rainbow')
        # ax2.legend(handles=scatter.legend_elements()[0], labels = x_axis)
    d = prepareForScatter(flag_rate_a, flag_rate_b, x_axis)
    groups = d.groupby('beta')
    for i, (name, group) in enumerate(groups):
        ax2.scatter(group.x, group.y,  c=[colors[i]],marker='o', linestyle='', label=name)
    ax2.legend()
    ax2.set_xlabel('group_a flag_rate')
    ax2.set_ylabel('group_b flag_rate')
    ax2.set_title('flag rate')
    ax2.axline((0, 0), slope=1)
    
    
       
    #recall rate
    # fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 7))
    # for i in range(len(flag)):
    # #     label = f'Experiment {i+1}'
    #     plt.plot(x_value, tpr_ratio[i], marker='o')
    # # plt.plot(x_value, y_tpr_ratio, marker='o')
    # # plt.ylim(0,2)
    # # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.figure(figsize=(6, 4))
    # Calculate upper and lower bounds for the area chart
    upper_bound = tpr_ratio_mean + np.sqrt(tpr_ratio_var) # 1 std
    lower_bound = tpr_ratio_mean - np.sqrt(tpr_ratio_var)
    
    # Create the area chart
    ax3.fill_between(x_axis, lower_bound, upper_bound, color='gray', alpha=0.5, label='Variance Area')
    ax3.plot(x_axis, tpr_ratio_mean, marker='o')
    ax3.set_xlabel(axis_title)
    ax3.set_ylabel('true positive rate ratio: group_a/group_b')
    ax3.set_title('recall ratio')
    
    
    # for i in range(len(recall_a)):
    # #     label = f'Experiment {i+1}'
    #     scatter = ax4.scatter(recall_a[i], recall_b[i], marker='o',c = x_axis, alpha=0.5, cmap = 'gist_rainbow')
        # ax4.legend(handles=scatter.legend_elements(num=len(x_axis)+1)[0], labels = x_axis)
    #     for j, label in enumerate(x_value):
    #         plt.annotate(label, (recall_a[i][j], recall_b[i][j]), textcoords="offset points", xytext=(0,10), ha='center')
    # scatter = ax2.scatter(recall_a, recall_b, marker='o', c = beta_s, cmap="Spectral")
    # legend = ax2.legend(*scatter.legend_elements(), title="beta_s")
    # ax2.add_artist(legend)
    d = prepareForScatter(recall_a, recall_b, x_axis)
    groups = d.groupby('beta')
    for i, (name, group) in enumerate(groups):
        ax4.scatter(group.x, group.y,  c=[colors[i]],marker='o', linestyle='', label=name)
    ax4.legend()
    ax4.set_xlabel('group_a TPR')
    ax4.set_ylabel('group_b TPR')
    ax4.set_title('true positive rate')
    # ax2.set_xlim(0,0.04)
    # ax2.set_ylim(-0.02,0.02)
    ax4.axline((0, 0), slope=1)
                
    
    
    upper_bound = fpr_ratio_mean + np.sqrt(fpr_ratio_var)
    lower_bound = fpr_ratio_mean - np.sqrt(fpr_ratio_var)
    ax5.fill_between(x_axis, lower_bound, upper_bound, color='gray', alpha=0.5, label='Variance Area')
    ax5.plot(x_axis, fpr_ratio_mean , marker='o')
    ax5.set_xlabel(axis_title)
    ax5.set_ylabel('false positive rate ratio: groupa/groupb')
    ax5.set_title('false positive rate ratio')
    
    # plt.subplot(2,2,2)
    # for i in range(len(flag)):
    # #     label = f'Experiment {i+1}'
    #     plt.scatter(fpr_a[i], fpr_b[i], marker='o', c = x_value, alpha=0.5)
    #     for j, label in enumerate(x_value):
    #         plt.annotate(label, (fpr_a[i][j], fpr_b[i][j]), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.xlabel('false positive of group_a')
    # plt.ylabel('false positive of group_b')
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.show()
    
    # for i in range(len(recall_a)):
    #     scatter = ax6.scatter(fpr_a[i], fpr_b[i], marker='o',c = x_axis, alpha=0.5, cmap = 'gist_rainbow')
    #     # for j, label in enumerate(x_axis):
    #     #     ax6.annotate(label, (fpr_a[i][j], fpr_b[i][j]), textcoords="offset points", xytext=(0,10), ha='center')
    #     # legend = ax6.legend(*scatter.legend_elements(), title="base_rate_b/base_rate_a")
    #     ax6.legend(handles=scatter.legend_elements(num=len(x_axis)+1)[0], labels = x_axis)
    d = prepareForScatter(fpr_a, fpr_b, x_axis)
    groups = d.groupby('beta')
    for i, (name, group) in enumerate(groups):
        ax6.scatter(group.x, group.y,  c=[colors[i]],marker='o', linestyle='', label=name)
    ax6.legend()
    # ax6.add_artist(legend)
    ax6.set_xlabel('group_a FPR')
    ax6.set_ylabel('group_b FPR')
    ax6.set_title('false positive rate')
    ax6.axline((0, 0), slope=1)
    
    #precision
    # fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15, 7))
    upper_bound = ppr_ratio_mean + np.sqrt(ppr_ratio_var)
    lower_bound = ppr_ratio_mean - np.sqrt(ppr_ratio_var)
    ax7.fill_between(x_axis, lower_bound, upper_bound, color='gray', alpha=0.5, label='Variance Area')
    ax7.plot(x_axis, ppr_ratio_mean, marker='o')
    ax7.set_xlabel(axis_title)
    ax7.set_ylabel('positive predixt rate ratio: group_a/group_b')
    ax7.set_title('precision ratio')
    
    # for i in range(len(recall_a)):
    #     scatter = ax8.scatter(ppr_a[i], ppr_b[i], marker='o',c = x_axis, alpha=0.5, cmap = 'gist_rainbow')
    #     ax8.legend(handles=scatter.legend_elements(num=len(x_axis)+1)[0], labels = x_axis)
    d = prepareForScatter(ppr_a, ppr_b, x_axis)
    groups = d.groupby('beta')
    for i, (name, group) in enumerate(groups):
        ax8.scatter(group.x, group.y,  c=[colors[i]],marker='o', linestyle='', label=name)
    ax8.legend()
    ax8.set_xlabel('group_a PPR')
    ax8.set_ylabel('group_b PPR')
    ax8.set_title('positive predict rate')
    ax8.axline((0, 0), slope=1)
    
    
    #violin plot
    
    p = sns.violinplot(data=violin_true, x="rate", y="outlier_score", hue="group", split=True, ax = ax9)
    legend = ax9.legend(title="group")
    legend.set_bbox_to_anchor((1.15, 1))
    ax9.set_xlabel(axis_title)
    title = violin_title + " scores only for the true anomalies"
    ax9.set_title(title)
    
    
    p = sns.violinplot(data=violin_df, x="rate", y="outlier_score", hue="group", split=True, ax = ax10)
    legend = ax10.legend(title="group")
    legend.set_bbox_to_anchor((1.2, 1))
    ax10.set_xlabel(axis_title)
    title = violin_title + " scores for all data"
    ax10.set_title(title)
        
    # # Specify the directory and file name for saving
    # save_path = '/Users/hahaha/Desktop/research-cmu/pic/'  # Change this to your desired directory path
    # file_name = 'base_rate_k=60.png'  # Change this to your desired file name and extension
    
    # # Save the figure to the specified path and file name
    # fig.savefig(save_path + file_name)
    plt.show()    
    
    
def plot_whole(axis_title, x_axis, 
               tpr_whole_mean, fpr_whole_mean, ppr_whole_mean, auroc_mean):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15, 15))
   
    #true positive rate
    ax1.plot(x_axis, tpr_whole_mean, marker='o')
    ax1.set_xlabel(axis_title)
    ax1.set_ylabel('true positive rate')
    ax1.set_title('recall')
    # ax1.set_ylim(0.9, 1.1)
    
    #false positive rate
    ax2.plot(x_axis, fpr_whole_mean, marker='o')
    ax2.set_xlabel(axis_title)
    ax2.set_ylabel('false positive rate')
    ax2.set_title('false positive rate')
    # ax2.set_ylim(-0.1, 0.1)
    
    #positive prediction value
    ax3.plot(x_axis, ppr_whole_mean, marker='o')
    ax3.set_xlabel(axis_title)
    ax3.set_ylabel('positive prediction value')
    ax3.set_title('precision')
    # ax3.set_ylim(0.8,1.2)

    # auroc
    ax4.plot(x_axis, auroc_mean, marker='o')
    ax4.set_xlabel(axis_title)
    ax4.set_ylabel('AUROC score')
    plt.show()

def plot_diff(axis_title, x_axis, 
              flag_rate_a, flag_rate_b, base_ratea, base_rateb,
              recall_a, recall_b, tpr_whole,
              fpr_a, fpr_b, fpr_whole,
              ppr_a, ppr_b, ppr_whole):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15, 15))
    
    # flag rate difference
    flag_rate_diff_a = [[0 for j in range(len(flag_rate_a[0]))] for i in range(len(flag_rate_a))]
    flag_rate_diff_b = [[0 for j in range(len(flag_rate_a[0]))] for i in range(len(flag_rate_a))]
    
    # base rate difference and other case 
        # other case
    if isinstance(base_ratea, float) and isinstance(base_rateb, float):
    # if len(base_ratea) == 1 and len(base_rateb) == 1:
        for i in range(len(flag_rate_a)):
            for j in range(len(flag_rate_a[0])):
                flag_rate_diff_a[i][j] = flag_rate_a[i][j] - base_ratea #base_ratea[0]
                flag_rate_diff_b[i][j] = flag_rate_b[i][j] - base_ratea #base_rateb[0]
        #flag rate difference mean
        flag_rate_a_new = [rate_a - base_ratea for index, rate_a in enumerate(np.mean(flag_rate_a, axis=0))] # base_ratea[0]
        flag_rate_b_new = [rate_b - base_rateb for index, rate_b in enumerate(np.mean(flag_rate_b, axis=0))] # base_ratea[0]
    # base rate difference
    else:
        for i in range(len(flag_rate_a)):
            for j in range(len(flag_rate_a[0])):
                flag_rate_diff_a[i][j] = flag_rate_a[i][j] - base_ratea[j]
                flag_rate_diff_b[i][j] = flag_rate_b[i][j] - base_rateb[j]
        #flag rate difference mean
        flag_rate_a_new = [rate_a - base_ratea[index] for index, rate_a in enumerate(np.mean(flag_rate_a, axis=0))]
        flag_rate_b_new = [rate_b - base_rateb[index] for index, rate_b in enumerate(np.mean(flag_rate_b, axis=0))]

    # standard error
    std_err_a = np.std(flag_rate_diff_a, axis = 0)/np.sqrt(len(flag_rate_diff_a))
    std_err_b = np.std(flag_rate_diff_b, axis = 0)/np.sqrt(len(flag_rate_diff_b))


    # # get variance 
    # flag_rate_var_a = matrix.get_var(flag_rate_a_new, flag_rate_diff_a)
    # flag_rate_var_b = matrix.get_var(flag_rate_b_new, flag_rate_diff_b)
    
    # upper bound & lower bound
    # upper_bound_a = flag_rate_a_new + np.sqrt(flag_rate_var_a)
    # lower_bound_a = flag_rate_a_new - np.sqrt(flag_rate_var_a)
    
    # upper_bound_b = flag_rate_b_new + np.sqrt(flag_rate_var_b)
    # lower_bound_b = flag_rate_b_new - np.sqrt(flag_rate_var_b)
    
    # ax1.plot(x_axis, flag_rate_a_new, color='orange', marker='o', label='group A')
    # ax1.plot(x_axis, flag_rate_b_new, color='blue', marker='o', label='group B')
    ax1.plot(x_axis, [0]*len(x_axis), marker='o', label='overall - overall')
    ax1.errorbar(x_axis, flag_rate_a_new, yerr=std_err_a, fmt='o-', label='group A')
    ax1.errorbar(x_axis, flag_rate_b_new, yerr=std_err_b, fmt='o-', label='group B')
    # ax1.fill_between(x_axis, lower_bound_a, upper_bound_a, color='gray', alpha=0.5)
    # ax1.fill_between(x_axis, lower_bound_b, upper_bound_b, color='gray', alpha=0.5)
    ax1.set_xlabel(axis_title)
    # ax1.set_ylabel('flag rate')
    # ax1.set_ylim(-0.5, 0.5)
    ax1.set_title('Distance between individual flag rate to overall base rate')
    ax1.legend()
    
    
    # recall
    # get difference
    recall_diff_a = matrix.get_difference(recall_a, tpr_whole)
    recall_diff_b = matrix.get_difference(recall_b, tpr_whole)
    
    # compute mean
    recall_diff_a_mean = np.mean(recall_diff_a, axis=0)
    recall_diff_b_mean = np.mean(recall_diff_b, axis=0)
    
    # compute standard error
    std_err_a = np.std(recall_diff_a, axis = 0)/np.sqrt(len(recall_diff_a))
    std_err_b = np.std(recall_diff_b, axis = 0)/np.sqrt(len(recall_diff_b))
    
    ax2.plot(x_axis, [0]*len(x_axis), marker='o', label='overall - overall')
    ax2.errorbar(x_axis, recall_diff_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax2.errorbar(x_axis, recall_diff_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax2.set_xlabel(axis_title)
    ax2.set_title('Distance between individual recall to overall recall')
    ax2.legend()
    
    
    # false positive rate
    # get difference
    fpr_diff_a = matrix.get_difference(fpr_a, fpr_whole)
    fpr_diff_b = matrix.get_difference(fpr_b, fpr_whole)
    
    # compute mean
    fpr_diff_a_mean = np.mean(fpr_diff_a, axis=0)
    fpr_diff_b_mean = np.mean(fpr_diff_b, axis=0)
    
    # compute standard error
    std_err_a = np.std(fpr_diff_a, axis = 0)/np.sqrt(len(fpr_diff_a))
    std_err_b = np.std(fpr_diff_b, axis = 0)/np.sqrt(len(fpr_diff_b))
    
    # fpr_a_mean = np.mean(fpr_a, axis=0)
    # fpr_b_mean = np.mean(fpr_b, axis=0)
    # fpr_a_new = [fpr_a_mean[i] - fpr_whole_mean[i] for i in range(len(fpr_a_mean))]
    # fpr_b_new = [fpr_b_mean[i] - fpr_whole_mean[i] for i in range(len(fpr_b_mean))]
    
    # ax3.plot(x_axis, fpr_a_new, color='orange', marker='o', label='group A')
    # ax3.plot(x_axis, fpr_b_new, color='blue', marker='o', label='group B')
    
    ax3.plot(x_axis, [0]*len(x_axis), marker='o', label='overall - overall')
    ax3.errorbar(x_axis, fpr_diff_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax3.errorbar(x_axis, fpr_diff_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax3.set_xlabel(axis_title)
    # ax3.set_ylabel('false positive rate')
    ax3.set_title('Distance between individual FPR to overall FPR')
    ax3.legend()
    
    # positive prediction value
    # get difference
    ppr_diff_a = matrix.get_difference(ppr_a, ppr_whole)
    ppr_diff_b = matrix.get_difference(ppr_b, ppr_whole)
    
    # compute mean
    ppr_diff_a_mean = np.mean(ppr_diff_a, axis=0)
    ppr_diff_b_mean = np.mean(ppr_diff_b, axis=0)
    
    # compute standard error
    std_err_a = np.std(ppr_diff_a, axis = 0)/np.sqrt(len(ppr_diff_a))
    std_err_b = np.std(ppr_diff_b, axis = 0)/np.sqrt(len(ppr_diff_b))
    
    # ppr_a_mean = np.mean(ppr_a, axis=0)
    # ppr_b_mean = np.mean(ppr_b, axis=0)
    # ppr_a_new = [ppr_a_mean[i] - ppr_whole_mean[i] for i in range(len(ppr_a_mean))]
    # ppr_b_new = [ppr_b_mean[i] - ppr_whole_mean[i] for i in range(len(ppr_b_mean))]
    
    # ax4.plot(x_axis, ppr_a_new, color='orange', marker='o', label='group A')
    # ax4.plot(x_axis, ppr_b_new, color='blue', marker='o', label='group B')
    ax4.plot(x_axis, [0]*len(x_axis), marker='o', label='overall - overall')
    ax4.errorbar(x_axis, ppr_diff_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax4.errorbar(x_axis, ppr_diff_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax4.set_xlabel(axis_title)
    # ax4.set_ylabel('false positive rate')
    ax4.set_title('Distance between individual precision to overall precision')
    ax4.legend()
    plt.show()    
    
def plot_line(axis_title, x_axis, 
              flag_rate_a, flag_rate_b, base_rate_a, base_rate_b,
              recall_a, recall_b, tpr_whole,
              fpr_a, fpr_b, fpr_whole,
              ppr_a, ppr_b, ppr_whole):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15, 15))
    
    # flag rate
    flag_rate_a_mean = np.mean(flag_rate_a, axis=0)
    flag_rate_b_mean = np.mean(flag_rate_b, axis=0)
    
    # compute standard error
    std_err_a = np.std(flag_rate_a, axis = 0)/np.sqrt(len(flag_rate_a))
    std_err_b = np.std(flag_rate_b, axis = 0)/np.sqrt(len(flag_rate_b))
    
    if len(base_rate_a) == 1 and len(base_rate_b) == 1:
        base_rate_a = base_rate_a * len(x_axis)
        base_rate_b = base_rate_b * len(x_axis)
    if base_rate_a == base_rate_b:
        ax1.plot(x_axis, base_rate_a, marker='o', label='overall')
    else:
        ax1.plot(x_axis, base_rate_a, marker='o', label='base_rate_a')
        ax1.plot(x_axis, base_rate_b, marker='o', label='base_rate_b')
    ax1.errorbar(x_axis, flag_rate_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax1.errorbar(x_axis, flag_rate_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax1.set_xlabel(axis_title)
    ax1.set_ylabel('flag rate')
    ax1.set_title('flag rate of group A, B and base rate of group A, B')
    ax1.legend()
   
    # recall
    recall_a_mean = np.mean(recall_a, axis=0)
    recall_b_mean = np.mean(recall_b, axis=0)
    tpr_whole_mean = np.mean(tpr_whole, axis=0)
    
    # compute standard error
    std_err_a = np.std(recall_a, axis = 0)/np.sqrt(len(recall_a))
    std_err_b = np.std(recall_b, axis = 0)/np.sqrt(len(recall_b))
    std_err_whole = np.std(tpr_whole, axis = 0)/np.sqrt(len(tpr_whole))
    
    ax2.errorbar(x_axis, tpr_whole_mean, yerr= std_err_whole, marker='o', label='overall')
    ax2.errorbar(x_axis, recall_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax2.errorbar(x_axis, recall_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax2.set_xlabel(axis_title)
    ax2.set_ylabel('true positive rate')
    ax2.set_title('true positive rate of group A, B and overall')
    ax2.legend()
    
    # fpr
    fpr_a_mean = np.mean(fpr_a, axis=0)
    fpr_b_mean = np.mean(fpr_b, axis=0)
    fpr_whole_mean = np.mean(fpr_whole, axis=0)
    
    # compute standard error
    std_err_a = np.std(fpr_a, axis = 0)/np.sqrt(len(fpr_a))
    std_err_b = np.std(fpr_b, axis = 0)/np.sqrt(len(fpr_b))
    std_err_whole = np.std(fpr_whole, axis = 0)/np.sqrt(len(fpr_whole))
    
    ax3.errorbar(x_axis, fpr_whole_mean, yerr= std_err_whole, marker='o', label='overall')
    ax3.errorbar(x_axis, fpr_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax3.errorbar(x_axis, fpr_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax3.set_xlabel(axis_title)
    ax3.set_ylabel('false positive rate')
    ax3.set_title('false positive rate of group A, B and overall')
    ax3.legend()
    
    # precision
    # fpr
    ppr_a_mean = np.mean(ppr_a, axis=0)
    ppr_b_mean = np.mean(ppr_b, axis=0)
    ppr_whole_mean = np.mean(ppr_whole, axis=0)
    
    # compute standard error
    std_err_a = np.std(ppr_a, axis = 0)/np.sqrt(len(ppr_a))
    std_err_b = np.std(ppr_b, axis = 0)/np.sqrt(len(ppr_b))
    std_err_whole = np.std(ppr_whole, axis = 0)/np.sqrt(len(ppr_whole))
    
    ax4.errorbar(x_axis, ppr_whole_mean, yerr= std_err_whole, marker='o', label='overall')
    ax4.errorbar(x_axis, ppr_a_mean, yerr=std_err_a, fmt='o-', label='group A')
    ax4.errorbar(x_axis, ppr_b_mean, yerr=std_err_b, fmt='o-', label='group B')
    ax4.set_xlabel(axis_title)
    ax4.set_ylabel('precision rate')
    ax4.set_title('precision rate of group A, B and overall')
    ax4.legend()
    plt.show()
    
    
    
def plot_hparam(roc, flag_a, flag_b, tpr_a, tpr_b, hparams):
    flag_ratio = [min(flag_a[i]/flag_b[i], flag_b[i]/flag_a[i]) if flag_a[i] != 0 and flag_b[i] != 0 else 0 for i in range(len(flag_a))]
    tpr_ratio = [min(tpr_a[i]/tpr_b[i], tpr_b[i]/tpr_a[i]) if tpr_a[i] != 0 and tpr_b[i] != 0 else 0 for i in range(len(tpr_a))]

    print("flag_ratio:",flag_ratio)
    print("tpr_ratio:", tpr_ratio)
    print("roc:", roc)
    colors = cm.rainbow(np.linspace(0, 1, len(hparams)))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12.5, 5))
    plt.subplots_adjust(wspace=0.6)

    # Map string categories to numeric values
    # unique_hparams = np.unique(hparams)  # Find all unique categories
    # hparam_to_numeric = {hparam: i for i, hparam in enumerate(unique_hparams)}  # Create a mapping
    # hparam_numeric = [hparam_to_numeric[hparam] for hparam in hparams]  # Convert all categories to numbers


    # scatter = plt.scatter(flag_ratio, roc, c=hparam_numeric, cmap='viridis')

    # # Adding color bar
    # plt.colorbar(scatter)
    
    
    for i, value in enumerate(hparams): 
        noise1 = np.random.rand(1,1)*0.1
        noise2 = np.random.rand(1,1)*0.1
        ax1.scatter(flag_ratio[i]+ noise1[0], roc[i]+noise2[0], c=[colors[i]], label=value)
    ax1.legend(title = 'alpha-gamma')
    # Labeling the axes
    ax1.set_xlabel('Flag Rate Ratio')
    ax1.set_ylabel('ROC')
    ax1.set_title('Flag Rate Ratio vs ROC')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.32,1), title = 'alpha-gamma')
    
    for i, value in enumerate(hparams): 
        ax2.scatter(tpr_ratio[i], roc[i], c=[colors[i]], label=value)
    ax2.legend(title = 'alpha-gamma')
    ax2.set_xlabel('TPR ratio')
    ax2.set_ylabel('ROC')
    ax2.set_title('TPR Ratio vs ROC')
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1), title = 'alpha-gamma')

    # min_x = min(min(flag_ratio), min(tpr_ratio), min(roc))
    # max_x = max(max(flag_ratio), max(tpr_ratio), max(roc))  
    min_x = min(min(flag_ratio), min(tpr_ratio))
    max_x = max(max(flag_ratio), max(tpr_ratio))    
    margin = (max_x - min_x) * 0.05  # 5% margin on each side
    min_x -= margin
    max_x += margin
    ax1.set_xlim([min_x, max_x])
    ax2.set_xlim([min_x, max_x]) 
    # ax1.set_ylim([min_x, max_x])
    # ax2.set_ylim([min_x, max_x]) 
    # ax1.set_xlim([0, 1])
    # ax2.set_xlim([0, 1]) 
    # ax1.set_ylim([0, 1])
    # ax2.set_ylim([0, 1]) 
    fig = plt.figure()

    # # Add a 3D subplot
    # ax = fig.add_subplot(111, projection='3d')
    # for i, value in enumerate(hparams): 
    #     ax.scatter(flag_ratio[i], tpr_ratio[i], roc[i], c=[colors[i]], label=value)
    # ax.set_xlabel('Flag Rate Ratio')
    # ax.set_ylabel('TPR ratio')
    # ax.set_zlabel('ROC')    
    # ax.legend(loc='upper left', bbox_to_anchor=(1.3,1), title = 'alpha-gamma')

def plot_line_final(axis_title, x_axis, 
              flag_rate_a, flag_rate_b, base_rate_a, base_rate_b,
              recall_a, recall_b, tpr_whole,
              fpr_a, fpr_b, fpr_whole,
              ppr_a, ppr_b, ppr_whole):
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15, 10))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(54, 8.5))
    
    # flag rate
    flag_rate_a_mean = np.mean(flag_rate_a, axis=0)
    flag_rate_b_mean = np.mean(flag_rate_b, axis=0)
    
    # if len(base_rate_a) == 1 and len(base_rate_b) == 1:
    if isinstance(base_rate_a, float) and isinstance(base_rate_b, float):
        base_rate_a = [base_rate_a] * len(x_axis)
        base_rate_b = [base_rate_b] * len(x_axis)
        base_rate_a = np.array(base_rate_a)
        base_rate_b = np.array(base_rate_b)
    if np.array_equal(base_rate_a, base_rate_b):
    # if base_rate_a == base_rate_b:
        ax1.plot(x_axis, base_rate_a, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    else:
        ax1.plot(x_axis, base_rate_a, marker='o', markersize=10, c='#808080', label='base rate a', linewidth=6)
        ax1.plot(x_axis, base_rate_b, marker='o', markersize=10, c='#404040', label='base rate b', linewidth=6)

    # compute standard deviation
    std_dev_a = np.std(flag_rate_a, axis=0)
    std_dev_b = np.std(flag_rate_b, axis=0)
    
    # Plot mean lines
    ax1.plot(x_axis, flag_rate_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax1.plot(x_axis, flag_rate_b_mean, marker='o',  markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax1.fill_between(x_axis, flag_rate_a_mean - std_dev_a, flag_rate_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax1.fill_between(x_axis, flag_rate_b_mean - std_dev_b, flag_rate_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax1.tick_params(axis='both', labelsize=38)
    ax1.set_xlabel(axis_title, size = 38)
    ax1.set_ylabel('flag rate', size = 38)
    # ax1.set_title('flag rate of group A, B and base rate of group A, B')
    # ax1.legend(fontsize=28)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
   
    # recall
    recall_a_mean = np.mean(recall_a, axis=0)
    recall_b_mean = np.mean(recall_b, axis=0)
    tpr_whole_mean = np.mean(tpr_whole, axis=0)
    
    # compute standard error
    std_dev_a = np.std(recall_a, axis=0)
    std_dev_b = np.std(recall_b, axis=0)
    std_dev_whole = np.std(tpr_whole, axis=0)
    
    # Plot mean lines
    ax2.plot(x_axis, tpr_whole_mean,marker='o', markersize=10, c='#404040', label='overall',linewidth=6 )
    ax2.plot(x_axis, recall_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax2.plot(x_axis, recall_b_mean, marker='o', markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax2.fill_between(x_axis, tpr_whole_mean - std_dev_whole, tpr_whole_mean + std_dev_whole, color='#404040', alpha=0.15)
    ax2.fill_between(x_axis, recall_a_mean - std_dev_a, recall_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax2.fill_between(x_axis, recall_b_mean - std_dev_b, recall_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax2.tick_params(axis='both', labelsize=38)
    ax2.set_xlabel(axis_title, size = 38)
    ax2.set_ylabel('true positive rate', size=38)
    # ax2.set_title('true positive rate of group A, B and overall')
    #ax2.legend(fontsize=12)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    

    # fpr
    fpr_a_mean = np.mean(fpr_a, axis=0)
    fpr_b_mean = np.mean(fpr_b, axis=0)
    fpr_whole_mean = np.mean(fpr_whole, axis=0)
    
   # compute standard error
    std_dev_a = np.std(fpr_a, axis=0)
    std_dev_b = np.std(fpr_b, axis=0)
    std_dev_whole = np.std(fpr_whole, axis=0)
    
    # Plot mean lines
    ax3.plot(x_axis, fpr_whole_mean, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    ax3.plot(x_axis, fpr_a_mean, marker='o', markersize=10,  c='#FC5A50', label='group a', linewidth=6)
    ax3.plot(x_axis, fpr_b_mean, marker='o', markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax3.fill_between(x_axis, fpr_whole_mean - std_dev_whole, fpr_whole_mean + std_dev_whole, color='#404040', alpha=0.15)
    ax3.fill_between(x_axis, fpr_a_mean - std_dev_a, fpr_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax3.fill_between(x_axis, fpr_b_mean - std_dev_b, fpr_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax3.tick_params(axis='both', labelsize=38)
    ax3.set_xlabel(axis_title, size=38)
    ax3.set_ylabel('false positive rate', size=38)
    # ax3.set_title('false positive rate of group A, B and overall')
    # ax3.legend(fontsize=12)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    

    # precision
    # ppr
    ppr_a_mean = np.mean(ppr_a, axis=0)
    ppr_b_mean = np.mean(ppr_b, axis=0)
    ppr_whole_mean = np.mean(ppr_whole, axis=0)
    
    # compute standard error
    std_dev_a = np.std(ppr_a, axis=0)
    std_dev_b = np.std(ppr_b, axis=0)
    std_dev_whole = np.std(ppr_whole, axis=0)
    
    # Plot mean lines
    ax4.plot(x_axis, ppr_whole_mean, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    ax4.plot(x_axis, ppr_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax4.plot(x_axis, ppr_b_mean, marker='o', markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax4.fill_between(x_axis, ppr_whole_mean - std_dev_whole, ppr_whole_mean + std_dev_whole, color='#404040', alpha=0.15)
    ax4.fill_between(x_axis, ppr_a_mean - std_dev_a, ppr_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax4.fill_between(x_axis, ppr_b_mean - std_dev_b, ppr_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax4.tick_params(axis='both', labelsize=38)
    ax4.set_xlabel(axis_title, size = 38)
    ax4.set_ylabel('precision', size= 38)
    # ax4.set_title('precision rate of group A, B and overall')
    #ax4.legend(fontsize=12)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=38)
    # fig.tight_layout()
    #plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.05)
    plt.show()
    
def fr(axis_title, x_axis, 
              flag_rate_a, flag_rate_b, base_rate_a, base_rate_b):
    fig, ax = plt.subplots(1,1,figsize=(13.5, 8.5))
    # flag rate
    flag_rate_a_mean = np.mean(flag_rate_a, axis=0)
    flag_rate_b_mean = np.mean(flag_rate_b, axis=0)
    
    # if len(base_rate_a) == 1 and len(base_rate_b) == 1:
    if isinstance(base_rate_a, float) and isinstance(base_rate_b, float):
        # base_rate_a = base_rate_a * len(x_axis)
        # base_rate_b = base_rate_b * len(x_axis)
        base_rate_a = [base_rate_a] * len(x_axis)
        base_rate_b = [base_rate_b] * len(x_axis)
    if base_rate_a == base_rate_b:
        ax.plot(x_axis, base_rate_a, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    else:
        ax.plot(x_axis, base_rate_a, marker='o', markersize=10, c='#808080', label='base_rate_a', linewidth=6)
        ax.plot(x_axis, base_rate_b, marker='o', markersize=10, c='#404040', label='base_rate_b', linewidth=6)

    # compute standard deviation
    std_dev_a = np.std(flag_rate_a, axis=0)
    std_dev_b = np.std(flag_rate_b, axis=0)
    
    # Plot mean lines
    ax.plot(x_axis, flag_rate_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax.plot(x_axis, flag_rate_b_mean, marker='o',  markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax.fill_between(x_axis, flag_rate_a_mean - std_dev_a, flag_rate_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax.fill_between(x_axis, flag_rate_b_mean - std_dev_b, flag_rate_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax.tick_params(axis='both', labelsize=38)
    ax.set_xlabel(axis_title, size = 38)
    ax.set_ylabel('flag rate', size = 38)
    # ax1.set_title('flag rate of group A, B and base rate of group A, B')
    # ax1.legend(fontsize=28)
    handles, labels = ax.get_legend_handles_labels()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
def tpr(axis_title, x_axis, 
              flag_rate_a, flag_rate_b,
              recall_a, recall_b, tpr_whole):
    fig, ax = plt.subplots(1,1,figsize=(13.5, 8.5))
    # recall
    recall_a_mean = np.mean(recall_a, axis=0)
    recall_b_mean = np.mean(recall_b, axis=0)
    tpr_whole_mean = np.mean(tpr_whole, axis=0)
    
    # compute standard error
    std_dev_a = np.std(recall_a, axis=0)
    std_dev_b = np.std(recall_b, axis=0)
    std_dev_whole = np.std(tpr_whole, axis=0)
    
    # Plot mean lines
    ax.plot(x_axis, tpr_whole_mean,marker='o', markersize=10, c='#404040', label='overall',linewidth=6 )
    ax.plot(x_axis, recall_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax.plot(x_axis, recall_b_mean, marker='o', markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax.fill_between(x_axis, tpr_whole_mean - std_dev_whole, tpr_whole_mean + std_dev_whole, color='#404040', alpha=0.15)
    ax.fill_between(x_axis, recall_a_mean - std_dev_a, recall_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax.fill_between(x_axis, recall_b_mean - std_dev_b, recall_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax.tick_params(axis='both', labelsize=38)
    ax.set_xlabel(axis_title, size = 38)
    ax.set_ylabel('true positive rate', size=38)
    # ax2.set_title('true positive rate of group A, B and overall')
    #ax2.legend(fontsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
def fpr(axis_title, x_axis, 
              flag_rate_a, flag_rate_b,
              fpr_a, fpr_b, fpr_whole):
    fig, ax = plt.subplots(1,1,figsize=(13.5, 8.5))
    # fpr
    fpr_a_mean = np.mean(fpr_a, axis=0)
    fpr_b_mean = np.mean(fpr_b, axis=0)
    fpr_whole_mean = np.mean(fpr_whole, axis=0)
    
   # compute standard error
    std_dev_a = np.std(fpr_a, axis=0)
    std_dev_b = np.std(fpr_b, axis=0)
    std_dev_whole = np.std(fpr_whole, axis=0)
    
    # Plot mean lines
    ax.plot(x_axis, fpr_whole_mean, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    ax.plot(x_axis, fpr_a_mean, marker='o', markersize=10,  c='#FC5A50', label='group a', linewidth=6)
    ax.plot(x_axis, fpr_b_mean, marker='o', markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax.fill_between(x_axis, fpr_whole_mean - std_dev_whole, fpr_whole_mean + std_dev_whole, color='#404040', alpha=0.15)
    ax.fill_between(x_axis, fpr_a_mean - std_dev_a, fpr_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax.fill_between(x_axis, fpr_b_mean - std_dev_b, fpr_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax.tick_params(axis='both', labelsize=38)
    ax.set_xlabel(axis_title, size=38)
    ax.set_ylabel('false positive rate', size=38)
    # ax3.set_title('false positive rate of group A, B and overall')
    # ax3.legend(fontsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def prec(axis_title, x_axis, 
              flag_rate_a, flag_rate_b,
              ppr_a, ppr_b, ppr_whole):
    fig, ax = plt.subplots(1,1,figsize=(13.5, 8.5))
    # precision
    # ppr
    ppr_a_mean = np.mean(ppr_a, axis=0)
    ppr_b_mean = np.mean(ppr_b, axis=0)
    ppr_whole_mean = np.mean(ppr_whole, axis=0)
    
    # compute standard error
    std_dev_a = np.std(ppr_a, axis=0)
    std_dev_b = np.std(ppr_b, axis=0)
    std_dev_whole = np.std(ppr_whole, axis=0)
    
    # Plot mean lines
    ax.plot(x_axis, ppr_whole_mean, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    ax.plot(x_axis, ppr_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax.plot(x_axis, ppr_b_mean, marker='o', markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax.fill_between(x_axis, ppr_whole_mean - std_dev_whole, ppr_whole_mean + std_dev_whole, color='#404040', alpha=0.15)
    ax.fill_between(x_axis, ppr_a_mean - std_dev_a, ppr_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax.fill_between(x_axis, ppr_b_mean - std_dev_b, ppr_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax.tick_params(axis='both', labelsize=38)
    ax.set_xlabel(axis_title, size = 38)
    ax.set_ylabel('precision', size= 38)
    # ax4.set_title('precision rate of group A, B and overall')
    #ax4.legend(fontsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
def auroc(auroc, x_axis, axis_title):
    fig, ax = plt.subplots(1,1,figsize=(13.5, 8.5))
    
    auroc_mean = np.mean(auroc, axis=0)
    # compute standard error
    std_dev_auroc = np.std(auroc, axis=0)
    
   
    # Fill area for 1 standard deviation error
    ax.fill_between(x_axis, auroc_mean - std_dev_auroc, auroc_mean + std_dev_auroc, color='#404040', alpha=0.15)
    
    ax.tick_params(axis='both', labelsize=38)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(x_axis, auroc_mean, marker='o', markersize=10, c='#404040', linewidth=6)
    ax.set_xlabel(axis_title, size=38)
    ax.set_ylabel('auroc score', size=38)
    plt.show()


def fr_legend(axis_title, x_axis, 
              flag_rate_a, flag_rate_b, base_rate_a, base_rate_b):
    fig, ax = plt.subplots(1,1,figsize=(13.5, 8.5))
    # flag rate
    flag_rate_a_mean = np.mean(flag_rate_a, axis=0)
    flag_rate_b_mean = np.mean(flag_rate_b, axis=0)
    
    # if len(base_rate_a) == 1 and len(base_rate_b) == 1:
    if isinstance(base_rate_a, float) and isinstance(base_rate_a, float):
        # base_rate_a = base_rate_a * len(x_axis)
        # base_rate_b = base_rate_b * len(x_axis)
        base_rate_a = [base_rate_a] * len(x_axis)
        base_rate_b = [base_rate_b] * len(x_axis)
    if base_rate_a == base_rate_b:
        ax.plot(x_axis, base_rate_a, marker='o', markersize=10, c='#404040', label='overall', linewidth=6)
    else:
        ax.plot(x_axis, base_rate_a, marker='o', markersize=10, c='#808080', label='base rate a', linewidth=6)
        ax.plot(x_axis, base_rate_b, marker='o', markersize=10, c='#404040', label='base rate b', linewidth=6)

    # compute standard deviation
    std_dev_a = np.std(flag_rate_a, axis=0)
    std_dev_b = np.std(flag_rate_b, axis=0)
    
    # Plot mean lines
    ax.plot(x_axis, flag_rate_a_mean, marker='o', markersize=10, c='#FC5A50', label='group a', linewidth=6)
    ax.plot(x_axis, flag_rate_b_mean, marker='o',  markersize=10, c='#069AF3', label='group b', linewidth=6)

    # Fill area for 1 standard deviation error
    ax.fill_between(x_axis, flag_rate_a_mean - std_dev_a, flag_rate_a_mean + std_dev_a, color='#FC5A50', alpha=0.15)
    ax.fill_between(x_axis, flag_rate_b_mean - std_dev_b, flag_rate_b_mean + std_dev_b, color='#069AF3', alpha=0.15)
    ax.tick_params(axis='both', labelsize=38)
    ax.set_xlabel(axis_title, size = 38)
    ax.set_ylabel('flag rate', size = 38)
    # ax1.set_title('flag rate of group A, B and base rate of group A, B')
    # ax1.legend(fontsize=28)
    handles, labels = ax.get_legend_handles_labels()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend(loc='best', fontsize=32)
    


def ae_score(violin_df):
    axis_title = "ae score for normal and anomaly"
    fig, ax = plt.subplots(1,1)
    p = sns.violinplot(data=violin_df, x="rate", y="outlier_score", hue="Y", split=True, ax = ax)
    legend = ax.legend(title="Y")
    legend.set_bbox_to_anchor((1.15, 1))
    ax.set_xlabel("rate")
    ax.set_title(axis_title)

def ae_predict(violin_df):
    axis_title = "ae score for data predict as anomaly"
    fig, ax = plt.subplots(1,1)
    p = sns.violinplot(data=violin_df[violin_df['y_pred']==1], x="rate", y="outlier_score", hue="Y", split=True, ax = ax)
    legend = ax.legend(title="Y")
    legend.set_bbox_to_anchor((1.15, 1))
    ax.set_xlabel("rate")
    ax.set_title(axis_title)
    
# def data(violin_df):
#     plt.scatter(df.data["Xg_1"][df.data["Y"] == 0], df.data["Xc_1"][df.data["Y"] == 0], c= 'blue')
#     plt.scatter(df.data["Xg_1"][df.data["Y"] == 1], df.data["Xc_1"][df.data["Y"] == 1], c='red')
#     plt.xlabel("Xg_1")
#     plt.ylabel("Xc_1")

def plot_data(data, rate):
    # rate_list = list(set(data["rate"]))
    # number = len(rate_list)
    # for i in range(number):
    #     fig, ax = plt.subplots(5,5)
    #     for a, b in itertools.product([1,2,3,4,5], repeat=1):
    #         ax_{i}.plot(data[f'Xg_{a}'], data[f'Xc_{b}'], colorby=data["Y"])

    # Creating a figure with subplots
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))  # Adjust figsize as needed

    rate_data = data[data['rate'] == rate]
        
    for (a, b), ax in zip(itertools.product(range(5), repeat=2), axs.ravel()):
        # Filter the specific data for Xg_a and Xc_b
        Xg = rate_data[f'Xg_{a+1}']  # Assuming your columns start with Xg_1, Xg_2, etc.
        Xc = rate_data[f'Xc_{b+1}']  # Assuming your columns start with Xc_1, Xc_2, etc.
        Y = rate_data["Y"]

        Xg_pred = rate_data[rate_data["y_pred"] == 1][f'Xg_{a+1}']
        Xc_pred = rate_data[rate_data["y_pred"] == 1][f'Xc_{b+1}']
        group = rate_data[rate_data["y_pred"] == 1]["group"]
        
        # Now, plot on the respective axes
        ax.scatter(Xg, Xc, c=Y, cmap='viridis')  # Choose an appropriate colormap
        ax.scatter(Xg_pred, Xc_pred, c=group, cmap='Accent')

        # Setting labels for the first column and last row for clarity
        if b == 0:
            ax.set_ylabel(f'Xc_{a+1}')
        if a == 4:
            ax.set_xlabel(f'Xg_{b+1}')
    
    # Adjusting the layout
    plt.tight_layout()

    # # If you want to display a colorbar
    # fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.01, pad=0.02)

    # Show or save the figure for the current rate
    plt.suptitle(f'Plots for rate: {rate}')
    plt.show()


def plot_dot(flag_ratio, tpr_ratio, auroc, key):
    colors = cm.rainbow(np.linspace(0, 1, len(key)))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12.5, 5))
    plt.subplots_adjust(wspace=0.6)


    for i, value in enumerate(key): 
        noise1 = np.random.rand(1,1)*0.001
        noise2 = np.random.rand(1,1)*0.001
        ax1.scatter(flag_ratio[i]+ noise1[0], auroc[i]+noise2[0], c=[colors[i]], label=value)
    ax1.legend(title = 'alpha-gamma')
    # Labeling the axes
    ax1.set_xlabel('Flag Rate Ratio')
    ax1.set_ylabel('ROC')
    ax1.set_title('Flag Rate Ratio vs ROC')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.32,1), title = 'alpha-gamma')
    
    for i, value in enumerate(key): 
        ax2.scatter(tpr_ratio[i], auroc[i], c=[colors[i]], label=value)
    ax2.legend(title = 'alpha-gamma')
    ax2.set_xlabel('TPR ratio')
    ax2.set_ylabel('ROC')
    ax2.set_title('TPR Ratio vs ROC')
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1), title = 'alpha-gamma')

    # min_x = min(min(flag_ratio), min(tpr_ratio), min(roc))
    # max_x = max(max(flag_ratio), max(tpr_ratio), max(roc))  
    min_x = min(min(flag_ratio), min(tpr_ratio))
    max_x = max(max(flag_ratio), max(tpr_ratio))    
    margin = (max_x - min_x) * 0.05  # 5% margin on each side
    min_x -= margin
    max_x += margin
    ax1.set_xlim([min_x, max_x])
    ax2.set_xlim([min_x, max_x]) 
    # ax1.set_ylim([min_x, max_x])
    # ax2.set_ylim([min_x, max_x]) 
    # ax1.set_xlim([0, 1])
    # ax2.set_xlim([0, 1]) 
    # ax1.set_ylim([0, 1])
    # ax2.set_ylim([0, 1]) 
    fig = plt.figure()