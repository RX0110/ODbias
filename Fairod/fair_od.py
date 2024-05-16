import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import numpy as np
import torch
torch.set_num_threads(12)
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from scipy.stats import rankdata

from utils.utility import column_wise_norm, load_data, group_train, group_test, cor_train, cor_test, evaluate

from test_tube import HyperOptArgumentParser
import utils.matrix as matrix
from ae import *

import utils.cluster_data as cluster_data
import utils.scatter_data as scatter
import utils.save as save


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
# parser = HyperOptArgumentParser(strategy='grid_search') # Grid search as the strategy
# # parser.add_argument('--data', default='lympho', help='currently support MNIST only')
# parser.add_argument('--batch_size', type=int, default=300, help='batch size') #2000 - 200/100 fix no change
# # parser.add_argument('--normal_class', type = int, default = 4)
# parser.add_argument('--device', type = str, default = 'cuda')
# # parser.add_argument('--cuda_device', type = int, default = 5) 
# parser.add_argument('--exp_num', type = int, default = 0)
# parser.add_argument('--model', type = str, default= "AE")
# parser.add_argument('--transductive', type = str2bool, default= True) 
# parser.add_argument('--gpu_num', type = int, default= -1) # which machine to run on

# args=parser.parse_args()

def get_model_object(model_type, input_dim_list, hparam_opt):
    # model type is not used. However, in future, we may intend to invoke different structures,
    # where the model type can help in creating respective network structures
    # return RepresentationLearner(layer_dims)
    return AEModel(
                    input_dim_list = input_dim_list, 
                    learning_rate = hparam_opt['lr'],
                    weight_decay = hparam_opt['weight_decay'], 
                    epochs = hparam_opt['epochs'], 
                    batch_size = 300, 
                    device = 'cuda', 
                    dropout = hparam_opt['dropout'])

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

class FTrainer:
    def __init__(self, config):
        '''
        Keys in the config:
        model_type: takes values in ["base", "fairL", "fairC", "fairOD"]
        recon_criterion: loss function for reconstruction error
        sens_criterion: loss function for correlation based statistical parity
        group_criterion: loss function for group fidelity. See Eq. 13 in the paper.
        data: directory path to dataset. Directory contains X.pkl, y.pkl, pv.pkl.
                The .pkl files should be saved from an input dataset.
        epochs: training epochs for the learner
        lr: learning rate used in adam optimizer
        model_path: path to save the best model
        base_model_weights: path to saved parameters from the base autoencoder model
        base_model_scores: anomaly scores from base model. ndarray.
        niters: number of times repeating the experiment to reduce variation over random initializations
        n_splits: cross-validation splits
        n_hidden: number of nodes in a hidden layer
        params: list of parameter tuples
        '''
        self.model_type = config["model_type"]
        self.recon_criterion = config["recon_criterion"]
        self.sens_criterion = config["sens_criterion"]
        self.group_criterion = config["group_criterion"]
        # self.data_path = config["data"]
        self.epochs = config["epochs"]
        #self.lr = config["lr"]
        self.model_path = config["model_path"]  # if not None, then use this path to save the model
        self.base_model_weights = config[
            "base_model_weights"]  # final network weights of base model_type (base_weights.pth)
        self.base_model_scores = config[
            "base_model_scores"]  # anomaly scores for each instance by base model

        # how many re-runs with random initializations of weights for AE
        # for fair AE load the final AE weights corresponding to each iteration
        self.niters = config["niters"]

        self.n_splits = config["n_splits"]  # for recording cross-validation
        self.n_hidden = config["n_hidden"]  # network hidden layer nodes

        # list of tuples (alpha, gamma). See paper for definition of alpha and gamma
        # first entry in the list is a string "base" to denote no regularization
        self.params = config["params"]

        # hparam
        self.hparam = config["hparam"] # hparam for AE model

        # data
        self.data_class = config["data_class"]

        # storing anomaly scores
        self.scores = None

        # store true y and S
        # self.y = None
        # self.s = None
        self.data_pred = None

        # bias
        self.bias = config["bias_rate"]
        self.bias_type = config["bias_type"]

         # data type
        self.data_type = config["data_type"]

        # seed
        self.seed = config["seed"]

    def train(self):
        """
        Full training logic
        """
        # # returns anomaly scores from best model
        # residuals = None

        # # load dataset -- stored as pickled files
        X_df = self.data_class.data.loc[:, ((self.data_class.data.columns != 'group') & (self.data_class.data.columns != 'Y'))]
        y = self.data_class.data['Y'].to_numpy()
        S = self.data_class.data['group'].to_numpy()
        # y_df = self.data.loc[:,'Y']
        # X_cols = X_df.columns.tolist()
        X_numpy = X_df.to_numpy()
        # print(X_numpy)
        # y = y_df.to_numpy()
        # S = S_df.to_numpy()
        # print(S.shape)
        # self.y = y # numpy
        # self.s = S # numpy

        X_torch = column_wise_norm(torch.FloatTensor(X_numpy))
        S_torch = torch.LongTensor(S).flatten()
        # print(S_torch.shape)
        input_dim = X_df.shape[1] 

        # load scores for each instance from base model as a Tensor
        AE_scores = torch.Tensor(self.base_model_scores).flatten()
        # print(AE_scores.shape)

        # create tensor dataset
        dataset = TensorDataset(X_torch, S_torch, AE_scores)
        dataloader = DataLoader(dataset, batch_size=300, shuffle=True)
        # n_samples = len(dataset)
        # train_size = int(0.8 * n_samples)  # using 80% split in each random cross-validation
        # test_size = n_samples - train_size

        # if model type is FairOD, the use ndcg normalizer
        if self.model_type == "fairOD":
            ndcg_norm_maj = np.sum(
                (np.power(2.0, AE_scores.numpy()[S == 0]) - 1.0) / np.log2(rankdata(-AE_scores.numpy()[S == 0]) + 1.0))
            ndcg_norm_min = np.sum(
                (np.power(2.0, AE_scores.numpy()[S == 1]) - 1.0) / np.log2(rankdata(-AE_scores.numpy()[S == 1]) + 1.0))

        else:
            ndcg_norm_maj, ndcg_norm_min = None, None

        # record errors for each parameter across iterations
        total_loss = {}
        construction_loss = {}
        protected_loss = {}
        ranking_loss = {}
        for param in self.params:
            total_loss[param] = np.zeros(self.epochs)
            construction_loss[param] = np.zeros(self.epochs)
            protected_loss[param] = np.zeros(self.epochs)
            ranking_loss[param] = np.zeros(self.epochs)

        for param in self.params:
            alpha, gamma = param
            beta = 1.0 - alpha
            print(f'now runing seed {self.seed}: {alpha}, {gamma}')

            # compute dimension list
            input_dim_list = generate_input_dim_lst(self.hparam['num_layer'], 
                                            self.hparam['input_decay'],
                                            input_dim, 
                                            threshold = self.hparam['threshold'])
            model = get_model_object(self.model_type, input_dim_list, self.hparam)
            model.AE.load_state_dict(torch.load(self.base_model_weights))
            model.AE.to(device)
            #for param in model.AE.parameters():
            #    print(param.data)
            
            # # setting up the optimizer that will be used to update the weights of the model with respect to its loss gradient.
            # optimizer = optim.Adam(model.parameters(), lr=self.lr) # Adam, short for "Adaptive Moment Estimation", is an algorithm that computes adaptive learning rates for each parameter. 
            optimizer = torch.optim.Adam(model.AE.parameters(), \
                                        lr=self.hparam['lr'], weight_decay = self.hparam['weight_decay'])
            
            for epoch in range(self.epochs):
                # train the model_type for given number of epochs
                recons_l, sens_l, train_l,group_l = group_train(model=model,
                                            dataloader=dataloader, # dataset = TensorDataset(X_torch, S_torch, AE_scores)
                                            device=device,
                                            recon_criterion=self.recon_criterion,
                                            sens_croterion=self.sens_criterion,
                                            group_criterion=self.group_criterion,
                                            optimizer=optimizer,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            ndcg_norm_maj=ndcg_norm_maj,
                                            ndcg_norm_min = ndcg_norm_min
                )
            # print("Recons Loss: %.3f, Sensitivity Loss : %.3f, Group Loss : %.3f" % (recons_l, sens_l, group_l))
            # record test loss for this split
            split_loss = group_test(model=model,
                                            dataloader=dataloader,
                                            device=device,
                                            recon_criterion=self.recon_criterion,
                                            sens_criterion=self.sens_criterion,
                                            group_criterion=self.group_criterion,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            ndcg_norm_maj=ndcg_norm_maj,
                                            ndcg_norm_min=ndcg_norm_min
                                            )

            # compute anomaly scores
            model.AE.eval()
            this_X_pred = model.AE(X_torch.to(device))
            # check nan
            X_torch_numpy = X_torch.cpu().detach().numpy()
            this_X_pred_numpy = this_X_pred.cpu().detach().numpy()
            X_torch_numpy = np.nan_to_num(X_torch_numpy, nan=0.0, posinf=0.0, neginf=0.0)
            this_X_pred_numpy = np.nan_to_num(this_X_pred_numpy, nan=0.0, posinf=0.0, neginf=0.0)
            residuals = np.linalg.norm(X_torch_numpy - this_X_pred_numpy, axis=1)
            # residuals = np.linalg.norm(X_torch.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),
                                        # axis=1)

            self.scores = residuals

            if len(self.params) > 1: # in the choose gamma and alpha process
                fairOD_metrics = evaluate(y_true=y, S=S, scores=self.scores)
                method = "FairOD" 
                save.save_fairod(self.bias, self.bias_type, self.data_type, method, self.seed, alpha, gamma, fairOD_metrics)
                if self.seed == 123:
                    score = pd.DataFrame(self.scores, columns=['outlier_score'])
                    df_new = pd.concat([self.data_class.data, score],axis=1)
                    df_sort = df_new.sort_values(by = 'outlier_score', ascending = False)
                    df_sort["y_pred"] = [1 if i < self.data_class.total_criminal else 0 for i in range(len(self.scores))]
                    key = str(alpha) + '-' + str(gamma)
                    save.save_data(self.bias_type, self.data_type, "FairOD", df_sort, key)
                    
            else: # compute flag rate, TPR, FPR, PPR
                # reconstruct 
                # dataset = pd.concat([X_df, S_df, y_df], axis= 1)
                # dataset.columns = X_cols + ['group'] + ['Y']
                score = pd.DataFrame(self.scores, columns=['outlier_score'])
                df_new = pd.concat([self.data_class.data, score],axis=1)
                df_sort = df_new.sort_values(by = 'outlier_score', ascending = False)
                df_sort["y_pred"] = [1 if i < self.data_class.total_criminal else 0 for i in range(len(self.scores))]
                self.data_pred = df_sort
                # print(self.data_pred)
                


