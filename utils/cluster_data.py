#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:13:41 2023

@author: hahaha
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import math
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, roc_auc_score, auc
import random


#create group
def create_group(num_a, num_b):
    return pd.DataFrame([0.0] * num_a + [1.0] * num_b, columns = ["group"])
    
#create G
#group a: mean: half 20s, half 5s | var: 1,1,1,1,1
#group b: mean: half 5s, half 20s | var: 1,1,1,1,1
def create_g(dimension, num_a, num_b, seed_value):
    np.random.seed(seed_value + 100)
    # Define means and covariance matrices for the two distributions
    n = round(dimension/2)
    mean1 = np.array([5.0] * n + [20.0] * (dimension - n)) # 0, 3
    cov1 = np.diag([1.0] * dimension) # Diagonal covariance matrix for independence
    
    mean2 = np.array([20.0] * n + [5.0] * (dimension - n))
    cov2 = np.diag([1.0] * dimension) 


    # Generate random data points for the two distributions
    data_a = np.random.multivariate_normal(mean1, cov1, num_a)
    data_b = np.random.multivariate_normal(mean2, cov2, num_b)
    
    #combine two group
    column = ['Xg_' + str(i+1) for i in range(dimension)]
    dfg_a = pd.DataFrame(data_a, columns=column)
    dfg_b = pd.DataFrame(data_b, columns=column)
    df_ab = pd.concat([dfg_a, dfg_b], axis=0)
    return df_ab

# #Create C
# #criminal: all 0s | var: 1,1,1,1,1
# #non_criminal: from 3 to dimension ｜ var: 1,1,1,1,1
def create_c(dimension, num_a, num_b, base_ratea, base_rateb, seed_value):
    np.random.seed(seed_value)
#     mean1 = np.array([(2.0) * i + np.random.randint(1,4) for i in range(dimension)])
    mean1 = np.array([0.0] * dimension) #20
    cov1 = np.diag([1.0] * dimension) # Diagonal covariance matrix for independence
    # mean2 = np.array(10.0] * dimension) 
    # variance shift
    mean2 = np.array([3.0] * dimension) # 3.0 / 10.0
#     mean2 = np.array([(2.0) * i + 8 for i in range(dimension)])
    cov2 = np.diag([1.0] * dimension) 
    
    criminal_a = round(num_a * base_ratea)
    criminal_b = round(num_b * base_rateb)
                               
                     
    # Generate random data points for the two distributions
    data_a_criminal = np.random.multivariate_normal(mean2, cov2, criminal_a)
    data_a_nonc = np.random.multivariate_normal(mean1, cov1, num_a - criminal_a)
    data_b_criminal = np.random.multivariate_normal(mean2, cov2, criminal_b)
    data_b_nonc = np.random.multivariate_normal(mean1, cov1, num_b - criminal_b)
    
    #combine two group
    
    column = ['Xc_' + str(i+1) for i in range(dimension)]
    dfc_a_criminal = pd.DataFrame(data_a_criminal, columns=column)
    dfc_a_nonc = pd.DataFrame(data_a_nonc, columns=column)
    dfc_b_criminal = pd.DataFrame(data_b_criminal, columns=column)
    dfc_b_nonc = pd.DataFrame(data_b_nonc, columns=column)
    df = pd.concat([dfc_a_criminal,dfc_a_nonc, dfc_b_criminal, dfc_b_nonc], axis=0)
    return df

#create O:
#mean:all 50s | var: all 1s
def create_o(dimension, num_a, num_b, seed_value):
    np.random.seed(seed_value)
    n  = num_a + num_b
    mean = np.array([50.0] * dimension)
    var = np.diag([1.0] * dimension)
    
    data = np.random.multivariate_normal(mean, var, n)
    column = ['Xo_' + str(i+1) for i in range(dimension)]
    dfo = pd.DataFrame(data, columns=column)
    return dfo


#create Y
def create_y(num_a, num_b, base_ratea, base_rateb):
    y = np.array([1] * round(num_a * base_ratea) + [0] * (num_a - round(num_a * base_ratea)) + 
                  [1] * round(num_b * base_rateb) + [0] * (num_b - round(num_b * base_rateb)))
    df_y = pd.DataFrame(y, columns = ["Y"])
    return df_y

class Data:
    def __init__(self, num_a, num_b, base_ratea, base_rateb, dimension, seed_value):
        self.num_a = num_a
        self.num_b = num_b
        self.base_ratea = base_ratea
        self.base_rateb = base_rateb
        self.dimension = dimension
        self.total_criminal = math.ceil(num_a * base_ratea + num_b * base_rateb)
        group = create_group(num_a, num_b)
        df_g = create_g(dimension, num_a, num_b, seed_value)
        df_y = create_y(num_a, num_b, base_ratea, base_rateb)
        df_c = create_c(dimension, num_a, num_b, base_ratea, base_rateb, seed_value)
        df_o = create_o(dimension, num_a, num_b, seed_value)
        g = group.reset_index(drop=True)
        df_g = df_g.reset_index(drop=True)
        df_c = df_c.reset_index(drop=True)
        df_o = df_o.reset_index(drop=True)
        df_y = df_y.reset_index(drop=True)
        self.data = pd.concat([g, df_g, df_c, df_o, df_y], axis= 1)
        self.data_pred = pd.DataFrame()

    # LOF algorithm
    def LOF(self, n):
        clf = LocalOutlierFactor(n_neighbors=n, contamination=0.1)
        df_nogroup = self.data.loc[:, ~(self.data.columns.isin(['group', 'Y']))]
        clf.fit_predict(df_nogroup)
        outlier_scores = -clf.negative_outlier_factor_ #the larger the abnormal
        data_w_ypred = pd.concat([self.data, pd.DataFrame(outlier_scores, columns = ["outlier_score"])], axis = 1)
        data_w_ypred_sorted = data_w_ypred.sort_values(by = "outlier_score", ascending=False)
        data_w_ypred_sorted["y_pred"] = [1 if i < self.total_criminal else 0 for i in range(len(data_w_ypred_sorted))]
        self.data_pred = data_w_ypred_sorted
        return self.data_pred
    
    # input a list of possible neighbors number, return the optimal number 
    def decide_neighbors(self, l):
        n_roc = 0
        num = 0
        for n in l:
            self.LOF(n)
           #  fpr, tpr, thresholds = roc_curve(self.data_pred["Y"], self.data_pred["y_pred"])
           # # compute auc value
           #  roc_auc = auc(fpr, tpr)
            roc_auc = roc_auc_score(self.data_pred['Y'], self.data_pred['outlier_score'])
            if n_roc < roc_auc:
                num = n
                n_roc = roc_auc
        return num
    
    def iForest(self):
        clf = IsolationForest()
        df_nogroup = self.data.loc[:, ~(self.data.columns.isin(['group', 'Y']))]
        clf.fit(df_nogroup) 
        # scores = clf.decision_function(self.data) #score_samples: The lower, the more abnormal.
        scores = -clf.score_samples(df_nogroup)
        data_w_ypred = pd.concat([self.data, pd.DataFrame(scores, columns = ["outlier_score"])], axis = 1)
        data_w_ypred_sorted = data_w_ypred.sort_values(by = "outlier_score", ascending=False)
        data_w_ypred_sorted["y_pred"] = [1 if i < self.total_criminal else 0 for i in range(len(data_w_ypred_sorted))]
        self.data_pred = data_w_ypred_sorted
        return self.data_pred
    
    def sample_size_bias(self, beta_s):
        drop_total = round(self.num_b * beta_s)
        rows_to_drop = self.data[(self.data['group'] == 1)].sample(n=drop_total, random_state=42).index
        # print(rows_to_drop)
        new_data = self.data.drop(rows_to_drop).reset_index(drop=True)
        num_criminal_b = ((new_data["Y"] == 1) & (new_data["group"] == 1)).sum()
        self.data = new_data
        self.total_criminal =  round(self.num_a * self.base_ratea) + num_criminal_b
        self.num_b = self.num_b - drop_total
        
    def under_representation(self, beta_u):
        total_drop = round(self.num_b * self.base_rateb * beta_u)
        rows_to_drop_criminal = self.data[(self.data['group'] == 1) & (self.data['Y'] == 1)].sample(n=total_drop, random_state=42).index
        self.data = self.data.drop(rows_to_drop_criminal).reset_index(drop=True)
        self.total_criminal = round(self.num_a * self.base_ratea + self.num_b * self.base_rateb - total_drop)
        self.num_b = self.num_b - total_drop
        
    def mean_shift(self, meanShift):
        for i in range(len(self.data)):
            if(self.data.at[i,"group"]==1):
                self.data.iloc[i, self.dimension+1: self.dimension*2+1] += meanShift 
        
    def wrong_report(self, num):
        random.seed(42)
        # index of groupa
        groupa_index = self.data.index[self.data['group'] == 0].tolist()
        groupb_index = self.data.index[self.data['group'] == 1].tolist()
        replicate_index = random.sample(groupa_index, num)
        repTo_index = random.sample(groupb_index, num)
        
        #copy a's to b's
        for i in range(len(repTo_index)):
            self.data.iloc[repTo_index[i], self.dimension+1: self.dimension*2+1] = self.data.iloc[replicate_index[i], self.dimension+1: self.dimension*2+1]
            
   # 3 choice: Xg, Xc, Xc_anomaly
    def variance_shift(self, type, varShift):
        np.random.seed(42)
        random.seed(42)
        # data_b_criminal = 0
        # data_b_nonc = 0
        type_list = type.split("_")
        if type_list[0] == "multiple":
            # multiple
            # prepare for new Xg
            n = round(self.dimension/2)
            Xg_mean2 = np.array([20.0] * n + [5.0] * (self.dimension - n))
            Xg_cov2 = np.diag([1.0 * varShift] * self.dimension) 
            # Generate random data points for the two distributions
            data_new = np.random.multivariate_normal(Xg_mean2, Xg_cov2, self.num_b)
            
            # prepare for new Xc
            Xc_mean1 = np.array([0.0] * self.dimension) #normal
            Xc_cov1 = np.diag([1.0 * varShift] * self.dimension)
            Xc_mean2 = np.array([10.0] * self.dimension) #anoally
            Xc_cov2 = np.diag([1.0 * varShift] * self.dimension) 
            
            criminal_b = round(self.num_b * self.base_rateb)
                             
            # Generate random data points for the two distributions
            data_b_criminal = np.random.multivariate_normal(Xc_mean2, Xc_cov2, criminal_b)
            data_b_nonc = np.random.multivariate_normal(Xc_mean1, Xc_cov1, self.num_b - criminal_b)
        
        elif type_list[0] == "add":
            # add
            # prepare for new Xg
            n = round(self.dimension/2)
            Xg_mean2 = np.array([20.0] * n + [5.0] * (self.dimension - n))
            data_new = []
            for i in range(self.num_b):
                # choose the number of dimension to inflate
                inflate_num = random.randint(3, 5)
                # choose the specific dimension
                inflate_indice = random.sample(list(range(5)), inflate_num)
                # prepare cov
                cov = [1.0] * self.dimension
                new_cov = [cov[i] + varShift if i in inflate_indice else cov[i] for i in range(len(cov))]
                Xg_cov2 = np.diag(new_cov) 
                # generate points
                data_new.append(np.random.multivariate_normal(Xg_mean2, Xg_cov2,1))
            # # Generate random data points for the two distributions
            # data_new = np.random.multivariate_normal(Xg_mean2, Xg_cov2, self.num_b)
            
            
            # prepare for new Xc
            Xc_mean1 = np.array([0.0] * self.dimension) #normal
            # Xc_cov1 = np.diag([1.0 + varShift] * self.dimension)
            Xc_mean2 = np.array([10.0] * self.dimension) #anomally
            # Xc_cov2 = np.diag([1.0 + varShift] * self.dimension) 
            
            # anomally 
            criminal_b = round(self.num_b * self.base_rateb)
            data_b_criminal = []
            for i in range(criminal_b):
                # choose the number of dimension to inflate
                inflate_num = random.randint(3, 5)
                # choose the specific dimension
                inflate_indice = random.sample(list(range(5)), inflate_num)
                # prepare cov
                cov = [1.0] * self.dimension
                new_cov = [cov[i] + varShift if i in inflate_indice else cov[i] for i in range(len(cov))]
                Xc_cov2 = np.diag(new_cov) 
                data_b_criminal.append(np.random.multivariate_normal(Xc_mean2, Xc_cov2, 1))
            

            # normal
            normal_b = self.num_b - criminal_b  
            data_b_nonc = []
            for i in range(normal_b):
                # choose the number of dimension to inflate
                inflate_num = random.randint(3, 5)
                # choose the specific dimension
                inflate_indice = random.sample(list(range(5)), inflate_num)
                # prepare cov
                cov = [1.0] * self.dimension
                new_cov = [cov[i] + varShift if i in inflate_indice else cov[i] for i in range(len(cov))]
                Xc_cov1 = np.diag(new_cov) 
                data_b_nonc.append(np.random.multivariate_normal(Xc_mean1, Xc_cov1, 1))
            
            # # Generate random data points for the two distributions
            # data_b_criminal = np.random.multivariate_normal(Xc_mean2, Xc_cov2, criminal_b)
            # data_b_nonc = np.random.multivariate_normal(Xc_mean1, Xc_cov1, self.num_b - criminal_b)
        

        # inflate both Xc and Xg
        if type_list[1] == "both-m":
            index = 0 # assign new data for index row
            criminal = 0
            non_criminal = 0
            for i in range(len(self.data)):
                if(self.data.at[i,"group"]==1):
                    self.data.iloc[i, 1: self.dimension+1] = data_new[index]
                    if(self.data.at[i,"Y"]==1):
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_criminal[criminal]
                        criminal += 1
                    else:
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_nonc[non_criminal]
                        non_criminal += 1
                    index += 1
        # inflate only Xg
        elif type_list[1] =="Xg":
            index = 0 # assign new data for index row
            for i in range(len(self.data)):
                if(self.data.at[i,"group"]==1):
                    self.data.iloc[i, 1: self.dimension+1] = data_new[index][0]
                    index += 1
        # inflate only Xc for anomaly and normal
        elif type_list[1] == "Xc":
            criminal = 0
            non_criminal = 0
            for i in range(len(self.data)):
                if(self.data.at[i,"group"]==1):
                    if(self.data.at[i,"Y"]==1):
                        #criminal
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_criminal[criminal][0]
                        criminal += 1
                    else:
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_nonc[non_criminal][0]
                        non_criminal += 1
        elif type_list[1] == "Xc-anomaly":
            criminal = 0
            for i in range(len(self.data)):
                if(self.data.at[i,"group"]==1):
                    if(self.data.at[i,"Y"]==1):
                        # print(self.data.iloc[i, self.dimension+1: self.dimension*2+1])
                        # print(data_b_criminal[criminal])
                        # print()
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_criminal[criminal][0]
                        criminal += 1
        # Xg and Xc
        elif type_list[1] == "both-a":
            index = 0 # assign new data for index row
            criminal = 0
            non_criminal = 0
            for i in range(len(self.data)):
                if(self.data.at[i,"group"]==1):
                    self.data.iloc[i, 1: self.dimension+1] = data_new[index][0]
                    if(self.data.at[i,"Y"]==1):
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_criminal[criminal][0]
                        criminal += 1
                    else:
                        self.data.iloc[i, self.dimension+1: self.dimension*2+1] = data_b_nonc[non_criminal][0]
                        non_criminal += 1
                    index += 1
        
    def obfuscation_bias_single(self, lb):
        np.random.seed(42)
        random.seed(42)
        num = int(self.num_b * lb)
        # index of groupa
        groupa_index = self.data.index[self.data['group'] == 0].tolist()
        groupb_index = self.data.index[self.data['group'] == 1].tolist()
        replicate_index = random.sample(groupa_index, num)
        repTo_index = random.sample(groupb_index, num)
        
        #copy a's to b's
        for i in range(len(repTo_index)):
            # choose dimension that be copied:
            dim_num = random.randint(1, 5)
            dim_index = random.sample(range(1,6), dim_num)
            for j in dim_index:
                self.data.iloc[repTo_index[i], j] = self.data.iloc[replicate_index[i], j]
            

                
            
        
            
        