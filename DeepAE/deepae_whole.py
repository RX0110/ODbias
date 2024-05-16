import argparse
import numpy as np
import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(__file__)
od_bias_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(od_bias_dir)
print(sys.path)
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import torch
torch.set_num_threads(12)
from models.ae import AEModel, ConvAEModel

import utils.cluster_data as cluster_data
import utils.scatter_data as scatter_data
import utils.matrix as matrix
import pandas as pd
import utils.save as save
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import f1_score

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
#general settings
parser = HyperOptArgumentParser(strategy='grid_search') # Grid search as the strategy
# parser.add_argument('--data', default='lympho', help='currently support MNIST only')
parser.add_argument('--batch_size', type=int, default=300, help='batch size') #2000 - 200/100 fix no change
# parser.add_argument('--normal_class', type = int, default = 4)
parser.add_argument('--device', type = str, default = 'cuda')
# parser.add_argument('--cuda_device', type = int, default = 5) 
parser.add_argument('--exp_num', type = int, default = 0)
parser.add_argument('--model', type = str, default= "AE")
parser.add_argument('--transductive', type = str2bool, default= True) 
parser.add_argument('--gpu_num', type = int, default= -1) # which machine to run on

args=parser.parse_args()
    # Access parsed arguments
    # learning_rate = args.learning_rate
    # batch_size = args.batch_size


if args.gpu_num != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    print("GPU num: %d, GPU device: %d" %( torch.cuda.device_count(), args.gpu_num)) # print 1
    torch.set_num_threads(4)
else:
    print("Default GPU:5 used")
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    



# generat input dimenson list
def generate_input_dim_lst(num_layer, input_decay, input_dim, threshold = 3):
    input_dim_list = []
    for i in range(num_layer):
        if int(input_dim / (input_decay**i)) >= threshold:
            input_dim_list.append(int(input_dim / (input_decay**i)))
        else:
            input_dim_list.append(threshold)
#    print(input_dim_list)
    return input_dim_list


# generat input dimenson list
def generate_input_dim_lst(num_layer, input_decay, input_dim, threshold = 3):
    input_dim_list = []
    for i in range(num_layer):
        if int(input_dim / (input_decay**i)) >= threshold:
            input_dim_list.append(int(input_dim / (input_decay**i)))
        else:
            input_dim_list.append(threshold)
    # print(input_dim_list)
    return input_dim_list

# hparam setting
parser = HyperOptArgumentParser(strategy='grid_search')
parser.opt_list('--num_layer', default = 2, type = int, tunable = True, options = [2,4]) # 3,4,5 Default value if not specified during tuning. 
parser.opt_list('--weight_decay', default = 0, type = float, tunable = True, options = [0,1e-5])
parser.opt_list('--lr', default =0.0001, type = float, tunable = True, options = [1e-3, 1e-4])
parser.opt_list('--epochs', default = 250, type = int, tunable = True, options = [100,250]) #100,250,500
parser.opt_list('--threshold', default = 3, type = int, tunable = True, options = [1])
parser.opt_list('--input_decay', default = 2, type = float, tunable = True, options = [1.0,1.5,2,2.5])
parser.opt_list('--dropout', default = 0, type = float, tunable = True, options = [0,0.2])
model_hparams = parser.parse_args("")


# input
bias_type_list = ["base_rate"] # base_rate/ sample_size_bias/ under_representation_bias/ mean_shift / variance_shift/ obfuscation_bias
data_type_list = ["scatter"]#scatter / cluster
method = "DeepAE" #Isolation Forest/ LOF


for bias_type in bias_type_list: 
    for data_type in data_type_list:
        # seed setting
        seed = [123, 124, 125, 126, 127]

        # data setting
        # train X: load data directly, train Y: true Y
        # group a and group b have data respective 1000
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
        # model hyparam
        hparam_list = []
        # data
        final_violin_df = pd.DataFrame()
        final_violin_true = pd.DataFrame()

        # begin
        for i in range(len(beta)): # loop for different beta
            print(f'now running bias {beta[i]}')
            # use to select optimal
            auroc_max = 0
            hparam_max = ""

            # list to store: list
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
            flag_ratio = [] 
            # auroc
            auroc = []
            # data
            y_violin_df = pd.DataFrame()
            y_violin_true = pd.DataFrame()

            # hparam choose
            for hparam in model_hparams.trials(2000): # Iterates through 2000 trials generated via grid search, exploring different hyperparameter combinations.
                print(hparam) # print this trail's hparams
                
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
                y_flag_rate = []
                y_flag_a = []
                y_flag_b = []
                y_flag_ratio = [] 
                # y_criminal = []
                # neighbor_list = []
                # auroc
                auroc_list = []
                # data
                temp_violin_df = pd.DataFrame()
                temp_violin_true = pd.DataFrame()

                for exp_num in range(len(seed)): # repeat times with same hparams setting
                    print(f'now running bias: {beta[i]}, seed num: {seed[exp_num]}')
                    # create data
                    if data_type == "cluster":
                        if bias_type != "base_rate":
                            df = cluster_data.Data(num_a, num_b, base_rate_a, base_rate_b, dimension, seed[exp_num])  # other
                        else:   
                            df = cluster_data.Data(num_a, num_b, base_rate_a[i], base_rate_b[i], dimension, seed[exp_num])  # base rate
                    else: 
                        if bias_type != "base_rate":
                            df = scatter_data.Data(num_a, num_b, base_rate_a, base_rate_b, dimension, inflate_list, seed[exp_num])
                        else:
                            df = scatter_data.Data(num_a, num_b, base_rate_a[i], base_rate_b[i], dimension, inflate_list, seed[exp_num]) 

                    # case selection: base_rate/ sample_size_bias/ under_representation_bias/ mean_shift / variance_shift/ obfuscation_bias
                    if bias_type == "sample_size_bias":
                        df.sample_size_bias(beta[i])
                    elif bias_type == "under_representation_bias":
                        df.under_representation(beta[i])
                    elif bias_type == "mean_shift":
                        df.mean_shift(beta[i])
                    elif bias_type == "variance_shift":
                        df.variance_shift("add_both-a", beta[i])
                    elif bias_type == "obfuscation_bias":
                        df.obfuscation_bias_single(beta[i])
                    # else: base rate

                    # normalize
                    scaler = MinMaxScaler()
                    df_normalized = pd.DataFrame(scaler.fit_transform(df.data), columns=df.data.columns)
                    df.data = df_normalized

                    # prepare data
                    train_X = df.data.loc[:, ((df.data.columns != 'Y') & (df.data.columns != 'group'))]
                    X_cols = train_X.columns.tolist()
                    train_X = train_X.to_numpy()
                    train_y = df.data['Y'].to_numpy()
                    input_dim = train_X.shape[1] 
                    torch.cuda.empty_cache()  

                    # model
                    hp_name = {
                        'num_layer': hparam.num_layer,
                        'input_decay': hparam.input_decay,
                        'epochs': hparam.epochs,
                        'lr': hparam.lr,
                        'weight_decay': hparam.weight_decay,
                        'dropout': hparam.dropout,
                        'threshold': hparam.threshold
                    }
                    input_dim_list =  generate_input_dim_lst(hparam.num_layer, 
                                                            hparam.input_decay,
                                                            input_dim, 
                                                            threshold = hparam.threshold)
                    model = AEModel(
                            input_dim_list = input_dim_list, 
                            learning_rate = hparam.lr,
                            weight_decay = hparam.weight_decay, 
                            epochs = hparam.epochs, 
                            batch_size = args.batch_size, 
                            device = args.device, 
                            dropout = hparam.dropout)
                    # save weight path
                    # path = f'../../../..result/{method}_weights/{bias_type}/{data_type}/seed_{seed[exp_num]}/biasRate_{beta[i]}.pt'
                    
                    loss_lst, total_time, memory_allocated, memory_reserved = model.fit(train_X)
                    result = model.predict(train_X, train_y)
                    # print(result) 

                    # identify predict anomaly and rebuild dataframe          
                    roc_score = roc_auc_score(train_y, result) # compute auroc
                    print("roc_score: ", roc_score)
                    X = pd.DataFrame(train_X, columns = X_cols)
                    pair = zip(train_y, result)
                    pair_df = pd.DataFrame(pair, columns=['Y', 'outlier_score'])
                    group = df.data[['group']]
                    df_new = pd.concat([X, pair_df, group], axis = 1)
                    df_sort = df_new.sort_values(by = 'outlier_score', ascending = False)
                    df_sort["y_pred"] = [1 if i < df.total_criminal else 0 for i in range(len(pair_df))]
                    # print(df_sort)
                    df.data_pred = df_sort

                    # violin data
                    if exp_num == 0: 
                        temp = df.data_pred.copy()
                        temp['rate'] = str(beta[i])
                        temp_violin_df = pd.concat([temp_violin_df, temp], axis=0)
                        t = df.data_pred[df.data_pred['Y'] == 1].copy()
                        t['rate'] = str(beta[i])
                        temp_violin_true = pd.concat([temp_violin_true, t], axis=0)

                    # add temp data
                    # auroc
                    auroc_list.append(roc_score)
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
                    y_tpr_whole.append(matrix.true_positive_rate_whole(df.data_pred))
                    #false positive rate
                    y_fpr_ratio.append(matrix.false_positive_rate_ratio(df.data_pred))
                    a, b = matrix.false_positive_rate(df.data_pred)
                    y_fpr_a.append(a)
                    y_fpr_b.append(b)
                    y_fpr_whole.append(matrix.false_positive_rate_whole(df.data_pred))
                    #positive predictive value rate
                    y_ppr_ratio.append(matrix.positive_predict_ratio(df.data_pred))
                    a, b = matrix.positive_predict(df.data_pred)
                    y_ppr_a.append(a)
                    y_ppr_b.append(b)
                    y_ppr_whole.append(matrix.positive_predict_whole(df.data_pred))
                    # print("each recall a", y_recall_a)
                    # print("each recall b", y_recall_b)

                torch.cuda.empty_cache()

                # compute mean
                print("auroc list for this hparam", auroc_list)
                auroc_mean = np.mean(auroc_list)
                print(f'mean of auroc for 5 experiments is {auroc_mean}')
                # choose hyparam and store best reulst
                if auroc_mean > auroc_max:
                    hparam_max = hp_name
                    auroc_max = auroc_mean
                    # tpr
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
                    # flag 
                    flag_ratio = y_flag_ratio[:]
                    flag_rate_a = y_flag_a[:]
                    flag_rate_b = y_flag_b[:]
                    # auroc
                    auroc = auroc_list[:]
                    # violin
                    y_violin_df = temp_violin_df.copy()
                    y_violin_true = temp_violin_true.copy()

                    print("temp optimal flag rate a:", y_flag_a)
                    print("temp tpr_a:", recall_a)
                    print("temp tpr_b:", recall_b)

            # add each hparam setting results into matrix
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
            # flag 
            final_flag_ratio = final_flag_ratio + [flag_ratio]
            final_flag_rate_a = final_flag_rate_a + [flag_rate_a]
            final_flag_rate_b = final_flag_rate_b + [flag_rate_b]
            # hparam
            hparam_list.append(hparam_max)
            # auroc
            final_auroc.append(auroc_max)
            auroc_matric = auroc_matric + [auroc]
            # data
            final_violin_df = pd.concat([final_violin_df, y_violin_df], axis=0)
            final_violin_true = pd.concat([final_violin_true, y_violin_true], axis=0)

        print("auroc:", final_auroc)
        print("hparam list:", hparam_list)

        print("flag rate a:", final_flag_rate_a)
        print("flag rate b:", final_flag_rate_b)
        print("final flag_ratio:", final_flag_ratio)
        print("tpr a:", final_recall_a)
        print("tpr b:", final_recall_b)
        print("tpr whole:", final_tpr_whole)
        print("fpr a:", final_fpr_a)
        print("fpr b:", final_fpr_b)
        print("fpr whole:", final_fpr_whole)
        print("ppr a:", final_ppr_a)
        print("ppr b:", final_ppr_b)
        print("ppr whole:", final_ppr_whole)

        # save
        save.save_opt(bias_type, data_type, method, hparam_list, final_auroc)
        save.newsave(bias_type, data_type, method, beta, 
                    base_rate_a, base_rate_b, base_rate_whole,
                    final_flag_rate_a, final_flag_rate_b, final_flag_ratio, 
                    final_recall_a, final_recall_b, final_tpr_whole, final_true_positive_ratio,
                    final_fpr_a, final_fpr_b, final_fpr_whole, final_fpr_ratio, final_ppr_a, final_ppr_b, final_ppr_whole,
                    final_ppr_ratio, auroc_matric)
        save.save_violin_data(bias_type, data_type, method, final_violin_df, final_violin_true)