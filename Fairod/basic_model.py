import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import numpy as np
import torch
torch.set_num_threads(12)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from scipy.stats import rankdata


from utils.utility import column_wise_norm, load_data, group_train, group_test, cor_train, cor_test, evaluate

from test_tube import HyperOptArgumentParser
import utils.matrix as matrix
from ae import AEModel


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import utils.save as save



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
                    device = "cuda", 
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

class BTrainer:
    def __init__(self, config):
        '''
        Keys in the config:
        model_type: takes values in ["base", "fairL"]
        recon_criterion: loss function for reconstruction error
        sens_criterion: loss function for correlation based statistical parity
        data: directory path to dataset. Directory contains X.pkl, y.pkl, pv.pkl.
                The .pkl files should be saved from an input dataset.
        epochs: training epochs for the learner
        lr: learning rate used in adam optimizer
        model_path: path to save the best model
        base_model_weights: path to saved parameters from the base autoencoder model if model_type="fairL"
        niters: number of times repeating the experiment to reduce variation over random initializations
        n_splits: cross-validation splits
        n_hidden: number of nodes in a hidden layer
        params: list of parameter values. parameter = alpha
        '''
        self.model_type = config["model_type"] # base
        self.recon_criterion = config["recon_criterion"] # nn.MSELoss()
        self.sens_criterion = config["sens_criterion"] # cor_sens_criterion

        # self.data_path = config["data"] # data path
        self.epochs = config["epochs"] 
        #self.lr = config["lr"] # 0.001
        self.model_path = config["model_path"]  # if not None, then use this path to save the model 
        self.base_model_weights = config[
            "base_model_weights"]  # final network weights of base model_type (base_weights.pth) # None

        # how many re-runs with random initializations of weights for AE
        self.niters = config["niters"] # 1

        self.n_splits = config["n_splits"]  # for recording cross-validation #2
        self.n_hidden = config["n_hidden"]  # network hidden layer nodes # 2

        # list of tuples (alpha, gamma). See paper for definition of alpha and gamma
        # first entry in the list is a string "base" to denote no regularization
        self.params = config["params"] # 1.

        # storing anomaly scores
        self.scores = None

        # hparam
        self.hparam = config["hparam"] # hparam for AE model

        # data setting
        self.data_class = config["data_class"]

        # bias
        self.bias = config["bias_rate"]
        self.bias_type = config["bias_type"]

        # data type
        self.data_type = config["data_type"]

        # stage
        self.find_weight_stage = config["find_weight_stage"]

        # auroc opt
        self.auroc_opt = config["auroc_opt"]

        # seed 
        self.seed = config["seed"]

    def train(self):
        """
        Full training logic
        """
        # returns anomaly scores from best model
        residuals = None

        # load dataset -- stored as pickled files
        # X, S, y = load_data(self.data_path)
        
        X = self.data_class.data.loc[:, ((self.data_class.data.columns != 'group') & (self.data_class.data.columns != 'Y'))].to_numpy()
        # print(X)
        S_numpy = self.data_class.data.loc[:, self.data_class.data.columns == 'group'].to_numpy()
        y_numpy = self.data_class.data['Y'].to_numpy()
        # X = X.to_numpy()
        # y = y.to_numpy()
        # S = S.to_numpy()

        X = column_wise_norm(torch.FloatTensor(X))
        S = torch.LongTensor(S_numpy).flatten()
        input_dim = X.shape[1] 

        # create tensor dataset
        dataset = TensorDataset(X, S)
        dataloader = DataLoader(dataset, batch_size=300, shuffle=True) # get baches of data
        # n_samples = len(dataset)
        # train_size = int(0.8 * n_samples)  # using 80% split in each random cross-validation
        # test_size = n_samples - train_size

        # record errors for each parameter across iterations
        total_loss = {}
        construction_loss = {}
        protected_loss = {}
        for param in self.params:
            total_loss[param] = np.zeros(self.epochs)
            construction_loss[param] = np.zeros(self.epochs)
            protected_loss[param] = np.zeros(self.epochs)


            # here params will only have alpha parameter
            alpha = 1.0
            beta = 0.0
            
            input_dim_list = generate_input_dim_lst(self.hparam['num_layer'], 
                                                self.hparam['input_decay'],
                                                input_dim, 
                                                threshold = self.hparam['threshold'])
            model = get_model_object(self.model_type, input_dim_list, self.hparam)
            

            optimizer = torch.optim.Adam(model.AE.parameters(), \
                                           lr=self.hparam['lr'], weight_decay = self.hparam['weight_decay'])
            
            print("Total running epochs %d" % self.epochs)
            print("Learning Rate: %5f" % self.hparam['lr'])
            print("Weight Decay: %.5f" % self.hparam['weight_decay'])
           
            
            while self.find_weight_stage == True:
                model.AE.to(device) # move to cuda
                #training stage for Base AE

                for epoch in range(self.epochs):
                    # train on full dataset
                    # for name, param in model.AE.named_parameters():
                    #     if param.requires_grad:
                    #         print(name, param.data) 

                    _, _, _ = cor_train(model=model,
                                                    dataloader=dataloader, # full dataset
                                                    device=device,
                                                    recon_criterion=self.recon_criterion,
                                                    sens_criterion=self.sens_criterion,
                                                    optimizer=optimizer,
                                                    alpha=1,
                                                    beta=0)
                
                #print(model.AE)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(model.AE.state_dict(), self.model_path)
                self.find_weight_stage = False
                # compute anomaly scores
                model.AE.eval()
                this_X_pred = model.AE(X.to(device))
                print("===========This is prediction================================================")
                # print(this_X_pred)
                residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),axis=1)
                fairOD_metrics = evaluate(y_true=y_numpy, S=S, scores=residuals)
                print(fairOD_metrics)
                if abs(fairOD_metrics["roc"] - self.auroc_opt) <= 0.04:
                    self.find_weight_stage == False

                # score = pd.DataFrame(residuals, columns=['outlier_score'])
                # df_new = pd.concat([self.data_class.data, score],axis=1)
                # df_sort = df_new.sort_values(by = 'outlier_score', ascending = False)
                # df_sort["y_pred"] = [1 if i < self.data_class.total_criminal else 0 for i in range(len(residuals))]
                # self.data_pred = df_sort
                # f1 = f1_score(self.data_pred['Y'], self.data_pred['y_pred'])
                # fairOD_metrics = evaluate(y_true=y_numpy, S=S, scores=residuals)
                # print("=============Here is the base AE evaluation metrics ============================")
                # print(fairOD_metrics)
            # else:
            #     # load weight
            #     #print(model.AE)
            #     model.AE.load_state_dict(torch.load(self.base_model_weights))
            #     model.AE.to(device) # move to cuda
            #     optimizer = torch.optim.Adam(model.AE.parameters(), \
            #                                lr=self.hparam['lr'], weight_decay = self.hparam['weight_decay'])
            #     for epoch in range(self.epochs):
            #         # train on full dataset
            #         # for name, param in model.AE.named_parameters():
            #         #     if param.requires_grad:
            #         #         print(name, param.data) 

            #         _, _, _ = cor_train(model=model,
            #                                         dataloader=dataloader, # full dataset
            #                                         device=device,
            #                                         recon_criterion=self.recon_criterion,
            #                                         sens_criterion=self.sens_criterion,
            #                                         optimizer=optimizer,
            #                                         alpha=1,
            #                                         beta=0)
                
            
            # # compute anomaly scores
            # model.AE.eval()
            # this_X_pred = model.AE(X.to(device))
            # residuals = np.linalg.norm(X.cpu().detach().numpy() - this_X_pred.cpu().detach().numpy(),
            #                                 axis=1)
            # print("=============Here is the base AE evaluation metrics ============================")
            # fairOD_metrics = evaluate(y_true=y_numpy, S=S, scores=residuals)
            # print(fairOD_metrics)
            save.save_deepAE_result("FairOD", self.bias_type, self.data_type, self.seed, self.bias, fairOD_metrics)
        self.scores = residuals
