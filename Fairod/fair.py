import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(__file__)
od_bias_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(od_bias_dir)
print(sys.path)
import utils.cluster_data as cluster_data
import utils.scatter_data as scatter_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utils.utility import cor_sens_criterion, group_corr_criterion, group_ndcg_criterion, evaluate
from basic_model import *
from fair_od import *
import utils.matrix as matrix
import json
import utils.save as save
import itertools
import torch
torch.set_num_threads(12)

save.save_dot("a", "b", "c", "e", [12], [34], [123], (1, 2))
path = f'result/hparam_value/a/b/c' #result/hparam_value/sample_size_bias/scatter/FairOD/hparam.json
save.new_save_alpha_gamma(path, "a", [23], [234])

# input
bias_type_list = ["sample_size_bias"] # "sample_size_bias", "under_representation_bias", "obfuscation_bias", "variance_shift", "base_rate"
data_type_list = ["cluster", "scatter"]

for bias_type in bias_type_list:
    for data_type in data_type_list:
        print("bias_type:", bias_type, "data_type:", data_type)

        # import hparam
        method = "DeepAE" 
        hparam_list, AE_auroc_list = matrix.read_hparam(f'result/hparam_value/{bias_type}/{data_type}/{method}.json')

        # setting
        method = 'FairOD'
        # seed setting
        seed = [123, 124, 125, 126, 127]
        # alpha, gamma setting
        alpha_try = [ 0.01, 0.05, 0.2, 0.5, 0.8]
        gamma_try = [ 0.01, 0.2, 0.5, 0.8]
        # data setting
        # train X: load data directly, train Y: true Y
        # group a and group b have data respective 1000
        # change base rate ratio, while holding total number of criminals same
        num_a = 1000
        num_b = 1000
        base_rate_a = 0.1
        base_rate_b = 0.1
        base_rate_whole = 0.1
        total = 100
        dimension = 5
        inflate_list = list(range(10,10+5*3,3)) # use for create scatter data
        # bias setting
        if bias_type == "base_rate":
            #base rate
            # compute base_rate
            beta = list(range(1,5))
            base_rate_a = []
            base_rate_b = []
            for p in beta:
                base_rate_a += [(total * 1 /(1+p))/num_a]
                base_rate_b += [(total * p /(1+p))/num_b]
        # base_rate_a: [0.05, 0.03333333333333333, 0.025, 0.02]
        # base_rate_b: [0.05, 0.06666666666666667, 0.075, 0.08]
        elif bias_type == "sample_size_bias":
            beta = [ 0.01, 0.05, 0.10, 0.2, 0.4, 0.6, 0.8] # sample size change rate
        elif bias_type == "under_representation_bias":
            beta = [ 0.01, 0.05, 0.10, 0.2, 0.4, 0.6, 0.8] # under representation change rate
        elif bias_type == "mean_shift":
            beta = [0, 2, 4, 6, 8] # mean shift change
        elif bias_type == "variance_shift":
            if data_type == "scatter":
                beta =  [0, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 3] # variance shift add scatter
            else:
                beta = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 6] 
        else:
            beta = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4] # obfuscation bias rate

        # final optimal output
        # initialize result for plot: list of list
        # base rate
        final_base_rate_a = []
        final_base_rate_b = []
        # tpr
        final_tpr_whole = []
        final_recall_a = []
        final_recall_b = []
        final_true_positive_ratio = []
        # fpr
        final_fpr_whole = []
        final_fpr_a = []
        final_fpr_b = []
        final_fpr_ratio = []
        # ppr
        final_ppr_whole = []
        final_ppr_a = []
        final_ppr_b = []
        final_ppr_ratio = []
        # flag rate
        final_flag_rate_a = []
        final_flag_rate_b = []
        final_flag_ratio = [] 
        # auroc
        final_auroc = []
        auroc_matric = []
        # data
        final_violin_df = pd.DataFrame()
        final_violin_true = pd.DataFrame()
        # list record alpha, gamma
        final_alpha_optimal = []
        final_gamma_optimal = []
        # dictionary record each bias' parameter dis
        final_dis_bias = {} #{'0.01': {'0-1': 1, '0-0': 2}, '0.05': {'0-1': 1, '0-0': 2} ... }

        # begin
        for i in range(len(beta)): # loop for different beta
            print(f'now running bias {beta[i]}') 
            # use to select optimal
            auroc_max = 0
            hparam_max = ""

            # lists of best results for each bias
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
            flag_rate_a = []
            flag_rate_b = []
            flag_ratio = [] 
            # auroc
            auroc = []
            # violin plot data
            temp_violin_df = pd.DataFrame()
            temp_violin_true = pd.DataFrame()
            # list record alpha, gamma
            alpha_optimal = None
            gamma_optimal = None
            # dictionary record each bias' parameter dis
            dis_bias = {} #'0.01': {'0-1': 1, '0-0': 2}

            for exp_num in range(len(seed)):
                print("Current perturbation bias values %.3f, seed value %.3f" % (beta[i], seed[exp_num])) #
                # dictionary record distance to (1,1,1)
                dis_list = {} #{'0-1': 1, '0-0': 2}

                # create dataset
                if (data_type == "cluster"):
                # cluster data
                    if bias_type == "base_rate":
                        df = cluster_data.Data(num_a, num_b, base_rate_a[i], base_rate_b[i], dimension, seed[exp_num])  
                    else:
                        df = cluster_data.Data(num_a, num_b, base_rate_a, base_rate_b, dimension, seed[exp_num])  
                else:
                    # scatter data
                    if bias_type == "base_rate":
                        df = scatter_data.Data(num_a, num_b, base_rate_a[i], base_rate_b[i], dimension, inflate_list, seed[exp_num])
                    else:
                        df = scatter_data.Data(num_a, num_b, base_rate_a, base_rate_b, dimension, inflate_list, seed[exp_num])
                # add bias
                if bias_type == "sample_size_bias":
                    df.sample_size_bias(beta[i])
                elif bias_type == "under_representation_bias":
                    df.under_representation(beta[i])
                elif bias_type == "variance_shift":
                    df.variance_shift("add_both-a", beta[i])
                elif bias_type == "obfuscation_bias":
                    df.obfuscation_bias_single(beta[i])
                # print(df.data)
                
                # hparam read
                hparam_opt = hparam_list[i] #i
                AE_auroc_opt = AE_auroc_list[i] #i
                print(f'For bias {beta[i]}, hparam_opt is {hparam_opt}, AE_score is {AE_auroc_opt}')

                # AE: store weight and get score
                config = {"group_criterion": None, 
                        "data_class": df, 
                        "epochs": hparam_opt['epochs'], "lr": hparam_opt['lr'],
                        "base_model_weights":  None, 
                        "base_model_scores": None, "niters": 1, "n_splits": 2, "n_hidden": 2,
                        "model_type": "base", "recon_criterion": nn.MSELoss(),
                        "sens_criterion": cor_sens_criterion,  # it doesn't use it, but needs the function
                        "model_path": f'result/{method}_weights/{bias_type}/{data_type}/seed_{seed[exp_num]}/biasRate_{beta[i]}.pt', 
                        "params": [1.],
                        "hparam": hparam_opt,
                        "bias_rate": beta[i],
                        "data_type": data_type,
                        "bias_type": bias_type,
                        "find_weight_stage": True,
                        "auroc_opt": AE_auroc_opt,
                        "seed": seed[exp_num]}
                # print("fscore_opt", fscore_opt)
                base_trainer = BTrainer(config=config)
                base_trainer.train()
                base_scores = base_trainer.scores

                print("now run FairOD")
                # run fairod with the saved weight
                config = {"model_type": "fairOD", "recon_criterion": nn.MSELoss(), "sens_criterion": cor_sens_criterion,
                        "group_criterion": group_ndcg_criterion,
                        "data_class": df,
                        "epochs": hparam_opt['epochs'], "lr": hparam_opt['lr'],
                        "model_path": None,
                        "base_model_weights": f'result/{method}_weights/{bias_type}/{data_type}/seed_{seed[exp_num]}/biasRate_{beta[i]}.pt',
                        "base_model_scores": base_scores,
                        "niters": 1, "n_splits": 2, "n_hidden": 2,
                        "params": [(a, g) for a in alpha_try for g in gamma_try],
                        "hparam": hparam_opt,
                        "bias_rate": beta[i],
                        "data_type": data_type,
                        "bias_type": bias_type,
                        "seed": seed[exp_num]}
                # Train alpha, gamma parameters
                fairOD_trainer = FTrainer(config=config)
                # create empty json file
                folder_path = f'result/FairOD_hparam/{bias_type}/{data_type}/seed_{seed[exp_num]}'
                os.makedirs(folder_path,exist_ok=True)
                with open(os.path.join(folder_path, f'biasRate_{beta[i]}.json'), "w") as json_file:
                    json.dump({}, json_file)
                # train
                fairOD_trainer.train()
                print()
            print("begin find optimal")
            # # data
            dot_tpr_ratio = []
            dot_flag_ratio = []
            dot_auroc = []
            key_list = []
            
            # best point and temp optimal key
            point1 = np.array([1,1,1])
            dis_min = np.inf 
            auroc_max = 0
            
            # loop through each key pair
            for (alpha, gamma) in itertools.product(alpha_try, gamma_try):
                print(f'now get mean of bias {beta[i]}, alpha{alpha} - gamma{gamma}')
                # store each key pair's flag rate, tpr, auroc: flag_a = [seed123, seed124, seed125, seed126, seed127]
                temp_flag_a = []
                temp_flag_b = []
                temp_tpr_a = []
                temp_tpr_b = []
                temp_auroc = []
                for seed_num in seed: # get value in each seed 
                    print("seed", seed_num)
                    path = f'result/{method}_hparam/{bias_type}/{data_type}/seed_{seed_num}/biasRate_{beta[i]}.json'
                    flaga, flagb, tpra, tprb, roc = matrix.read_flagtprroc(path, alpha, gamma)
                    temp_flag_a.append(flaga)
                    temp_flag_b.append(flagb)
                    temp_tpr_a.append(tpra)
                    temp_tpr_b.append(tprb)
                    temp_auroc.append(roc)
                # get mean for each value of each key pair
                flag_a_mean = [np.mean(temp_flag_a)]
                flag_b_mean = [np.mean(temp_flag_b)]
                tpr_a_mean = [np.mean(temp_tpr_a)]
                tpr_b_mean = [np.mean(temp_tpr_b)]
                auroc_mean = [np.mean(temp_auroc)]
                # compute ratio and put into list for future save
                flag_ratio = [min(flag_a_mean[i]/flag_b_mean[i], flag_b_mean[i]/flag_a_mean[i]) if flag_a_mean[i] != 0 and flag_b_mean[i] != 0 else 0 for i in range(len(flag_a_mean))]
                tpr_ratio = [min(tpr_a_mean[i]/tpr_b_mean[i], tpr_b_mean[i]/tpr_a_mean[i]) if tpr_a_mean[i] != 0 and tpr_b_mean[i] != 0 else 0 for i in range(len(tpr_a_mean))]
                dot_flag_ratio = dot_flag_ratio + flag_ratio
                dot_tpr_ratio = dot_tpr_ratio + tpr_ratio
                dot_auroc = dot_auroc + auroc_mean
                print(f'flag_ratio: {flag_ratio}')
                print(f'tpr_ratio: {tpr_ratio}')
                print(f'auroc mean: {auroc_mean}')
                print(f'dot_plot: flag_ratio:{dot_flag_ratio}, tpr_ratio:{dot_tpr_ratio}, auroc:{dot_auroc}')
                key = str(alpha) + '-' + str(gamma)
                key_list = key_list + [key]
                print("key_list", key_list)
                # form point
                point = np.array([flag_ratio[0], tpr_ratio[0], auroc_mean[0]])
                # compute distance
                dis = np.linalg.norm(point1 - point)
                print(f'distance for {alpha}-{gamma} is {dis}')
                dis_bias[str(alpha) + '-' + str(gamma)] = dis
                # compare and update optimal and record optimal results
                if dis < dis_min:
                    print("small distance found")
                    # get optimal by euclidean distance
                    # temp optimal result: list
                    # tpr
                    y_tpr_whole = []
                    y_recall_a = []
                    y_recall_b = []
                    y_true_positive_ratio = []
                    # fpr
                    y_fpr_whole = []
                    y_fpr_a = []
                    y_fpr_b = []
                    y_fpr_ratio = []
                    # ppr
                    y_ppr_whole = []
                    y_ppr_a = []
                    y_ppr_b = []
                    y_ppr_ratio = []
                    # flag rate
                    y_flag_a = []
                    y_flag_b = []
                    y_flag_ratio = [] 
                    # y_criminal = []
                    # neighbor_list = []
                    # auroc
                    auroc_list = []
                    
                    # update optimal
                    dis_min = dis
                    alpha_optimal = alpha
                    gamma_optimal = gamma
                    auroc_max = auroc_mean[0]
                    print(f'now optimal alpha and gamma is {alpha_optimal}-{gamma_optimal}')
                    # get optimal result
                    temp_violin_data = matrix.new_read_violin(f'result/data/{bias_type}/{data_type}/{method}', key)
                    temp = temp_violin_data.copy()
                    temp['rate'] = str(beta[i])
                    temp_violin_df = pd.concat([temp_violin_df, temp], axis=0)
                    t = temp[temp['Y'] == 1]
                    #t['rate'] = str(beta[i])
                    temp_violin_true = pd.concat([temp_violin_true, t], axis=0)
                    print("update result")
                    for seed_num in seed:
                        print("seed", seed_num)
                        path = f'result/{method}_hparam/{bias_type}/{data_type}/seed_{seed_num}/biasRate_{beta[i]}.json'
                        temp_auroc_list, temp_flag_rate_a, temp_flag_rate_b, temp_flag_ratio, temp_recall_a,\
          temp_recall_b, temp_tpr_whole, temp_true_positive_ratio, temp_fpr_a, temp_fpr_b, temp_fpr_ratio, temp_fpr_whole,\
          temp_ppr_a, temp_ppr_b, temp_ppr_ratio, temp_ppr_whole = matrix.read_fairod(path, key)
                        auroc_list = auroc_list + temp_auroc_list
                        y_flag_a = y_flag_a + temp_flag_rate_a
                        y_flag_b = y_flag_b + temp_flag_rate_b
                        y_flag_ratio = y_flag_ratio + temp_flag_ratio
                        # tpr
                        y_tpr_whole = y_tpr_whole + temp_tpr_whole
                        y_recall_a = y_recall_a + temp_recall_a
                        y_recall_b = y_recall_b + temp_recall_b
                        y_true_positive_ratio = y_true_positive_ratio + temp_true_positive_ratio
                        # fpr
                        y_fpr_whole = y_fpr_whole + temp_fpr_whole
                        y_fpr_a = y_fpr_a + temp_fpr_a
                        y_fpr_b = y_fpr_b + temp_fpr_b
                        y_fpr_ratio = y_fpr_ratio + temp_fpr_ratio
                        # ppr
                        y_ppr_whole = y_ppr_whole + temp_ppr_whole
                        y_ppr_a = y_ppr_a + temp_ppr_a
                        y_ppr_b = y_ppr_b + temp_ppr_b 
                        y_ppr_ratio = y_ppr_ratio + temp_ppr_ratio
                        print("flag_rate_a list", y_flag_a)
                        print("flag_rate_b list", y_flag_a)

                tpr_whole = y_tpr_whole[:]
                recall_a = y_recall_a[:]
                recall_b = y_recall_b[:]
                true_positive_ratio = y_true_positive_ratio[:]
                # fpr
                fpr_whole = y_fpr_whole[:]
                fpr_a = y_fpr_a[:]
                fpr_b = y_fpr_b[:]
                fpr_ratio = y_fpr_ratio[:]
                # ppr
                ppr_whole = y_ppr_whole[:]
                ppr_a = y_ppr_a[:]
                ppr_b = y_ppr_b[:]
                ppr_ratio = y_ppr_ratio[:]
                # flag rate
                flag_rate_a = y_flag_a[:]
                flag_rate_b = y_flag_b[:]
                flag_ratio = y_flag_ratio[:]
                # auroc
                auroc = auroc_list[:]
                print("temp tpr a:", recall_a)
                print("temp tpr b:", recall_b)
                print()

            # dis
            final_dis_bias[beta[i]] = dis_bias
            print("final dis dictionary", final_dis_bias)
            
            # save dot plot data: '
            # result/{method}_dot/{bias_type}/{data_type}/bias_Rate_{bias}
            save.save_dot(method, bias_type, data_type, beta[i], dot_flag_ratio, dot_tpr_ratio, dot_auroc, key_list)

            # append to matrix
            final_alpha_optimal = final_alpha_optimal + [alpha_optimal]
            final_gamma_optimal = final_gamma_optimal + [gamma_optimal]
            print("temp alpha optimal list:", alpha_optimal)
            print("temp gamma optimal list:", gamma_optimal)
            # tpr
            final_tpr_whole = final_tpr_whole + [tpr_whole]
            final_recall_a = final_recall_a + [recall_a]
            final_recall_b = final_recall_b + [recall_b]
            final_true_positive_ratio = final_true_positive_ratio + [true_positive_ratio]
            # fpr
            final_fpr_whole = final_fpr_whole + [fpr_whole]
            final_fpr_a = final_fpr_a + [fpr_a]
            final_fpr_b = final_fpr_b + [fpr_b]
            final_fpr_ratio = final_fpr_ratio + [fpr_ratio]
            # ppr
            final_ppr_whole = final_ppr_whole + [ppr_whole]
            final_ppr_a = final_ppr_a + [ppr_a]
            final_ppr_b = final_ppr_b + [ppr_b]
            final_ppr_ratio = final_ppr_ratio + [ppr_ratio]
            # flag rate
            final_flag_rate_a = final_flag_rate_a + [flag_rate_a]
            final_flag_rate_b = final_flag_rate_b + [flag_rate_b]
            final_flag_ratio = final_flag_ratio + [flag_ratio]
            # auroc
            final_auroc = final_auroc + [auroc_max]
            auroc_matric = auroc_matric + [auroc]
            # violin df
            final_violin_df = pd.concat([final_violin_df, temp_violin_df], axis=0)
            final_violin_true = pd.concat([final_violin_true, temp_violin_true], axis=0)

            print("final auroc mean:", final_auroc)
            print("final ppr_a matric:", final_ppr_a)
        
        # save 
        # save optimal alpha gamma
        print("optimal alpha:", final_alpha_optimal)
        print("optimal gamma:", final_gamma_optimal)
        path = f'result/hparam_value/{bias_type}/{data_type}/{method}' #result/hparam_value/sample_size_bias/scatter/FairOD/hparam.json
        save.new_save_alpha_gamma(path, final_alpha_optimal, final_gamma_optimal, final_auroc)
        # print(final_dis_bias)
        # save each setting's distance
        path = f'result/FairOD_distance/{bias_type}/{data_type}' #result/FairOD_distance/sample_sie_bias/scatter/distance.json
        save.save_distance(path, final_dis_bias)
        # save plot data
        save.newsave(bias_type, data_type, method, beta, 
                base_rate_a, base_rate_b, base_rate_whole,
                final_flag_rate_a, final_flag_rate_b, final_flag_ratio, 
                final_recall_a, final_recall_b, final_tpr_whole, final_true_positive_ratio,
                final_fpr_a, final_fpr_b, final_fpr_whole, final_fpr_ratio, final_ppr_a, final_ppr_b, final_ppr_whole,
                final_ppr_ratio, auroc_matric)
        # save violin data
        save.save_violin_data(bias_type, data_type, method, final_violin_df, final_violin_true)

                    
                
            





