import numpy as np
import math
import json
import os
import pandas as pd

#def how to compute flag_rate
def flag_rate_ratio(df):
    group_a = ((df["group"] == 0) & (df["y_pred"] == 1)).sum() / ((df["group"] == 0).sum())
    group_b = ((df["group"] == 1) & (df["y_pred"] == 1)).sum() / ((df["group"] == 1).sum())
    return group_a/group_b

#def how to compute flag_rate
def flag_rate(df):
    group_a = ((df["group"] == 0) & (df["y_pred"] == 1)).sum() / ((df["group"] == 0).sum())
    group_b = ((df["group"] == 1) & (df["y_pred"] == 1)).sum() / ((df["group"] == 1).sum())
    return group_a, group_b

def true_positive_rate_ratio(df):
    group_a = ( (df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 0)).sum() / ( (df["Y"] == 1) & (df["group"] == 0)).sum()
    group_b = ( (df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 1)).sum() / ( (df["Y"] == 1) & (df["group"] == 1)).sum()
    return group_a / group_b

def true_positive_rate(df):
    group_a = ( (df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 0)).sum() / ( (df["Y"] == 1) & (df["group"] == 0)).sum()
    group_b = ( (df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 1)).sum() / ( (df["Y"] == 1) & (df["group"] == 1)).sum()
    return group_a, group_b

def true_positive_rate_whole(df):
    return ( (df["Y"] == 1) & (df["y_pred"] == 1) ).sum() / ((df["Y"] == 1).sum())

def false_positive_rate_ratio(df):
    group_a = ( (df["Y"] == 0) & (df["y_pred"] == 1) & (df["group"] == 0)).sum() / ( (df["Y"] == 0) & (df["group"] == 0)).sum()
    group_b = ( (df["Y"] == 0) & (df["y_pred"] == 1) & (df["group"] == 1)).sum() / ( (df["Y"] == 0) & (df["group"] == 1)).sum()
    epsilon = 1e5 
    if group_b == 0:
        return group_a / (group_b + epsilon)
    return group_a / group_b

def false_positive_rate(df):
    group_a = ((df["Y"] == 0) & (df["y_pred"] == 1) & (df["group"] == 0)).sum() / ( (df["Y"] == 0) & (df["group"] == 0)).sum()
    group_b = ((df["Y"] == 0) & (df["y_pred"] == 1) & (df["group"] == 1)).sum() / ( (df["Y"] == 0) & (df["group"] == 1)).sum()
    return group_a, group_b

def false_positive_rate_whole(df):
    return ( (df["Y"] == 0) & (df["y_pred"] == 1) ).sum() / ((df["Y"] == 0).sum())

def positive_predict_ratio(df):
    group_a = ( (df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 0)).sum() / ( (df["y_pred"] == 1) & (df["group"] == 0)).sum()
    group_b = ( (df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 1)).sum() / ( (df["y_pred"] == 1) & (df["group"] == 1)).sum()
    return group_a / group_b

def positive_predict(df):
    group_a = ((df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 0)).sum() / ( (df["y_pred"] == 1) & (df["group"] == 0)).sum()
    group_b = ((df["Y"] == 1) & (df["y_pred"] == 1) & (df["group"] == 1)).sum()/ ( (df["y_pred"] == 1) & (df["group"] == 1)).sum()
    return group_a, group_b

def positive_predict_whole(df):
    return ((df["Y"] == 1) & (df["y_pred"] == 1)).sum() / ((df["y_pred"] == 1).sum())

def get_mean(ll):
    res = []
    transposed_fr = list(zip(*ll))
    for col in transposed_fr:
        m = 0
        l = 0
        for x in col :
            if x != np.inf and not math.isnan(x):
                m += x
                l += 1
        if l == 0:
            res = res + [0]
        else:
            res = res + [m/l]
    return res


# input list of mean value and list of list result
def get_var(m, ll):
    res = []
    transposed_fr = list(zip(*ll))
    for i in range(len(transposed_fr)):
        v = 0
        l = 0
        for x in transposed_fr[i]:
            if x != np.inf and not math.isnan(x):
                v += (x - m[i]) ** 2
                l += 1
        if l == 0:
            res = res + [0]
        else:
            res = res + [v/l]
    return res

def get_difference(matrix1, matrix2):
    res = [[0 for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            res[i][j] = matrix1[i][j] - matrix2[i][j]
    return res

# read json file
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        base_rate_a = data["Base Rate A"]
        base_rate_b = data["Base Rate B"]
        base_rate_whole = data["Base Rate whole"]
        
        exp_num = 0
        # find exp num
        for key, value in data.items():
            # find value pair for parameter
            if key.startswith("parameter"):
                for index, v in enumerate( value["Flag_rate_A"].values()):
                    exp_num = index
                break
        exp_num += 1
        # initialize
        x_axis = []
        flag_rate_a = [[] for item in range(exp_num)]
        flag_rate_b = [[] for item in range(exp_num)]
        # flag_rate_whole = [[] for item in range(exp_num)]
        recall_a = [[] for item in range(exp_num)]
        recall_b = [[] for item in range(exp_num)]
        tpr_whole = [[] for item in range(exp_num)]
        fpr_a = [[] for item in range(exp_num)]
        fpr_b = [[] for item in range(exp_num)]
        fpr_whole = [[] for item in range(exp_num)]
        ppr_a = [[] for item in range(exp_num)]
        ppr_b = [[] for item in range(exp_num)]
        ppr_whole = [[] for item in range(exp_num)]
        flag_ratio = [[] for item in range(exp_num)]
        true_positive_ratio = [[] for item in range(exp_num)]
        fpr_ratio = [[] for item in range(exp_num)]
        ppr_ratio = [[] for item in range(exp_num)]
        auroc = [[] for item in range(exp_num)]
        
        # read
        for key, value in data.items():
            # find value pair for parameter
            if key.startswith("parameter"):
                # read information from each parameter
                # paramater
                x_axis.append(value["parameter"])
                
                # flag rate A
                for index, v in enumerate( value["Flag_rate_A"].values()):
                    flag_rate_a[index].append(v)

                # recall A
                for index, v in enumerate( value["Recall_A"].values()):
                    recall_a[index].append(v)
            
                # fpr A
                for index, v in enumerate( value["FPR_A"].values()):
                    fpr_a[index].append(v)
                # fpr_a = fpr_a + [[v for v in value["FPR_A"].values()]]
                
                # ppr A
                for index, v in enumerate( value["Precision_A"].values()):
                    ppr_a[index].append(v)
                
                # flag rate B
                for index, v in enumerate( value["Flag_rate_B"].values()):
                    flag_rate_b[index].append(v)
                
                # recall B
                for index, v in enumerate( value["Recall_B"].values()):
                    recall_b[index].append(v)
               
                # fpr B
                for index, v in enumerate( value["FPR_B"].values()):
                    fpr_b[index].append(v)
                    
                # ppr B
                for index, v in enumerate( value["Precision_B"].values()):
                    ppr_b[index].append(v)
                    
                # # flag rate whole
                # for index, v in enumerate( value["Flag_rate_overall"].values()):
                #     flag_rate_whole[index].append(v)
               
                # recall whole
                for index, v in enumerate( value["Recall_overall"].values()):
                    tpr_whole[index].append(v)
                
                # fpr overall
                for index, v in enumerate( value["FPR_ovrall"].values()):
                    fpr_whole[index].append(v)
                    
                # precision whole
                for index, v in enumerate( value["Precision_overall"].values()):
                   ppr_whole[index].append(v)
                
                # flag rate ratio
                for index, v in enumerate( value["Flag_rate_ratio"].values()):
                   flag_ratio[index].append(v)
                
                # tpr ratio
                for index, v in enumerate( value["TPR_ratio"].values()):
                   true_positive_ratio[index].append(v)
                
                # fpr ratio
                for index, v in enumerate( value["FPR_ratio"].values()):
                   fpr_ratio[index].append(v)
                
                # ppr ratio 
                for index, v in enumerate( value["PPR_ratio"].values()):
                   ppr_ratio[index].append(v)

                # auroc
                for index, v in enumerate( value["auroc"].values()):
                   auroc[index].append(v)
                   
        
    return x_axis, base_rate_a, base_rate_b, base_rate_whole, flag_rate_a, flag_rate_b, recall_a, \
        recall_b, tpr_whole, fpr_a, fpr_b, fpr_whole, ppr_a, ppr_b, ppr_whole, \
            flag_ratio, true_positive_ratio, fpr_ratio, ppr_ratio, auroc
                
                
def read_hparam(file_path):
    print(os.getcwd())
    print(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
        hparam = data['hparams']
        auroc = data['auroc']
    return hparam,auroc
        
        

def read_violin(file_path):
    with open(file_path, 'r') as file:
        combined_json = json.load(file)
        
     # Convert the JSON strings back to DataFrames
    violin_true = pd.read_json(combined_json['violin_true'], orient='records')
    violin_df = pd.read_json(combined_json['violin_df'], orient='records')

    return violin_true, violin_df


def read_fairod(file_path, key):
    # hparam = []
    # tpr
    tpr_whole = []
    recall_a = []
    recall_b = []
    true_positive_ratio = []
    # fpr
    fpr_whole = []
    fpr_a = []
    fpr_b = []
    fpr_ratio = []
    # ppr
    ppr_whole = []
    ppr_a = []
    ppr_b = []
    ppr_ratio = []
    # flag rate
    #flag_rate_whole = []
    flag_rate_a = []
    flag_rate_b = []
    flag_rate_ratio = [] 

    roc_list = []
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        # for key, dic in data.items():
        dic = data[key]
        # print(dic)
        # hparam.append(key)
        roc_list.append(dic['roc'])
        flag_rate_a.append(dic['flag_rate'][0])
        flag_rate_b.append(dic['flag_rate'][1])
        flag_rate_ratio.append(dic['flag_rate_ratio'])
        recall_a.append(dic['TPR'][0])
        recall_b.append(dic['TPR'][1])
        tpr_whole.append(dic["TPR whole"])
        true_positive_ratio.append(dic["TPR ratio"])
        fpr_a.append(dic["FPR"][0])
        fpr_b.append(dic["FPR"][1])
        fpr_ratio.append(dic["FPR ratio"])
        fpr_whole.append(dic["FPR whole"])
        ppr_a.append(dic["PPR"][0])
        ppr_b.append(dic["PPR"][1])
        ppr_ratio.append(dic["PPR ratio"])
        ppr_whole.append(dic["PPR whole"])
    return roc_list, flag_rate_a, flag_rate_b, flag_rate_ratio, recall_a,\
          recall_b, tpr_whole, true_positive_ratio, fpr_a, fpr_b, fpr_ratio, fpr_whole,\
          ppr_a, ppr_b, ppr_ratio, ppr_whole

def read_flagtprroc(file_path, alpha, gamma):
    with open(file_path, 'r') as file:
        data = json.load(file)
        print(data)
        key = str(alpha)+'-'+str(gamma)
        return data[key]['flag_rate'][0], data[key]['flag_rate'][1], data[key]['TPR'][0], data[key]['TPR'][1], data[key]['roc']


def read_fairod_param(file_path):
    alpha = []
    gamma = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        alpha = data['alpha_list']
        gamma = data['gamma_list']
    
    return alpha, gamma

def new_read_violin(folder_path, key):
    with open(os.path.join(folder_path, f'{key}.json'), 'r') as file:
        data_json = json.load(file)
        df = pd.read_json(data_json['data'], orient='records')
        return df


def read_hparam(file_path):
    print(os.getcwd())
    print(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
        hparam = data['hparams']
        auroc = data['auroc']
    return hparam,auroc

def read_dot(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        flag_ratio = data['flag_ratio']
        tpr_ratio = data['tpr_ratio']
        auroc = data['auroc']
        key = data['key']
    return flag_ratio, tpr_ratio, auroc, key

def get_optimal_from_distance(file_path):
    alpha = []
    gamma = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            min_dis = float('inf')
            optimal = ''
            for k, v in value.items():
                if v < min_dis:
                    optimal = k
                    min_dis = v
            l = optimal.split('-')
            alpha.append(l[0])
            gamma.append(l[1])
    return alpha, gamma

