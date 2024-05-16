import json
import os

def save(bias_type, data_type, method, parameter, 
          base_rate_a, base_rate_b, base_rate_whole, 
          flag_rate_a, flag_rate_b, flag_ratio, 
          recall_a, recall_b, tpr_whole, true_positive_ratio, fpr_a, fpr_b, fpr_whole, fpr_ratio,
          ppr_a, ppr_b, ppr_whole, ppr_ratio, auroc):
    
    # initialize the dictionary
    nested_data = {}
    nested_data["Bias Type"] = bias_type
    nested_data["Data Type"] = data_type
    nested_data["Algorithm method"] = method
    nested_data["Base Rate A"] = base_rate_a
    nested_data["Base Rate B"] = base_rate_b
    nested_data["Base Rate whole"] = base_rate_whole
    nested_data["Group A"] = "Type: Gaussian, Xc mean: 0 variance: 1"
    nested_data["Group B"] = "Type: Gaussian, Xc mean: 0 variance: 1"
    nested_data["Group A_anomaly"] = "Type: Gaussian, Xc mean: 3 variance: 1"
    nested_data["Group B_anomaly"] = "Type: Gaussian, Xc mean: 3 variance: 1"
    
    
    # each parameter is a key
    for i in range(len(parameter)):
        config = {}
        config["parameter"] = parameter[i]
        config["Flag_rate_A"] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_rate_a))[i])}
        config["Recall_A"] = {f'e{index}': value for index, value in enumerate(list(zip(*recall_a))[i])}
        config["FPR_A"] = {f'e{index}': value for index, value in enumerate(list(zip(*fpr_a))[i])}
        config["Precision_A"] = {f'e{index}': value for index, value in enumerate(list(zip(*ppr_a))[i])}
        config["Flag_rate_B"] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_rate_b))[i])}
        config["Recall_B"] = {f'e{index}': value for index, value in enumerate(list(zip(*recall_b))[i])}
        config["FPR_B"] = {f'e{index}': value for index, value in enumerate(list(zip(*fpr_b))[i])}
        config["Precision_B"] = {f'e{index}': value for index, value in enumerate(list(zip(*ppr_b))[i])}
        # config["Flag_rate_overall"] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_rate_whole))[i])}
        config["Recall_overall"] = {f'e{index}': value for index, value in enumerate(list(zip(*tpr_whole))[i])}
        config["FPR_ovrall"] = {f'e{index}': value for index, value in enumerate(list(zip(*fpr_whole))[i])}
        config["Precision_overall"] = {f'e{index}': value for index, value in enumerate(list(zip(*ppr_whole))[i])}
        config["Flag_rate_ratio"] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_ratio))[i])}
        config["TPR_ratio"] = {f'e{index}': value for index, value in enumerate(list(zip(*true_positive_ratio))[i])}
        config["FPR_ratio"] = {f'e{index}': value for index, value in enumerate(list(zip(*fpr_ratio))[i])}
        config["PPR_ratio"] = {f'e{index}': value for index, value in enumerate(list(zip(*ppr_ratio))[i])}
        config["auroc"] =  {f'e{index}': value for index, value in enumerate(list(zip(*auroc))[i])}
        
        # add the dic into whole dictionary
        nested_data[f'parameter_{parameter[i]}'] = config
    
    
    # flag_A = {}
    # flag_B = {}
    # recall_A = {}
    # recall_B = {}
    # fpr_A = {}
    # fpr_B = {}
    # ppr_A = {}
    # ppr_B = {}
    
    # for i in range(len(flag_rate_a[0])):
        
    #     flag_A[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_rate_a))[i])}
    #     flag_B[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_rate_b))[i])}
    #     recall_A[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*recall_a))[i])}
    #     recall_B[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*recall_b))[i])}
    #     fpr_A[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*fpr_a))[i])}
    #     fpr_B[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*fpr_b))[i])}
    #     ppr_A[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*ppr_a))[i])}
    #     ppr_B[f'beta_{i}'] = {f'e{index}': value for index, value in enumerate(list(zip(*ppr_b))[i])}

   
    # nested_data = {"base_rateA": base_ratea, "base_rateB": base_rateb,
    #                "flag_A": flag_A, "flag_B": flag_B,
    #                "recall_A": recall_A, "recall_B": recall_B,
    #                "fpr_A": fpr_A, "fpr_B": fpr_B,
    #                "ppr_A": ppr_A, "ppr_B": ppr_B}
    
    folder_path = f'../../result/results/{bias_type}/{data_type}' #/result/results/sample_size_bias/cluster/FairOD.json
    
    # Create the folder if it doesn't exist
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("Folder already exists")
    
    
    # Open a file for writing
    with open(os.path.join(folder_path, f'{method}.json'), "w") as json_file:
        json.dump(nested_data, json_file)

    
    # with open(file_path, 'w') as json_file:
    #     json.dump(nested_data, json_file)

    
# # Sample data - replace this list of lists with your actual data
# list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# # Create a dictionary where each key is a row index and the value is the corresponding row
# data_dict = {f'row_{index}': row for index, row in enumerate(list_of_lists)}

# # Specify the file path where you want to save the JSON file
# file_path = '/Users/hahaha/Desktop/research-cmu/result/output_data.json'

# # Save the dictionary to a JSON file
# with open(file_path, 'w') as json_file:
#     json.dump(data_dict, json_file)

# print(f"Data has been saved to {file_path}")



# # Sample data - replace these list of lists with your actual data
# data_list_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# data_list_2 = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']]

# # Create a dictionary to hold multiple layers
# nested_data = {}

# # Add the first list of lists as a big key-value pair
# nested_data['data_1'] = {f'row_{index}': row for index, row in enumerate(data_list_1)}

# # Add the second list of lists as another big key-value pair
# nested_data['data_2'] = {f'row_{index}': row for index, row in enumerate(data_list_2)}

# # Specify the file path where you want to save the JSON file
# file_path = '/Users/hahaha/Desktop/research-cmu/result/output_data.json'

# # Save the nested dictionary to a JSON file
# with open(file_path, 'w') as json_file:
#     json.dump(nested_data, json_file)

# print(f"Data has been saved to {file_path}")

def save_opt(bias_type,data_type, method, hparams, auroc): 
    print(os.getcwd())
    folder_path = f'result/hparam_value/{bias_type}/{data_type}'
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("Folder already exists")
    
    combine = {"auroc": auroc, "hparams": hparams}
    # Open a file for writing
    with open(os.path.join(folder_path, f'{method}.json'), "w") as json_file:
        json.dump(combine, json_file)

                    # sample_size, scatter, FairOD, violin_df, violin_true
def save_violin_data(bias_type,data_type, method, violin_df, violin_true):
    folder_path = f'result/data/{bias_type}/{data_type}' # /result/data/sample_size_bias/sctter/FairOD.json
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("Folder already exists")
    
    json_true = violin_true.to_json(orient='records')
    json_df = violin_df.to_json(orient='records')  
    combined_json = {
    'violin_true': json_true,
    'violin_df': json_df
        }  
    # Open a file for writing
    with open(os.path.join(folder_path, f'{method}.json'), "w") as json_file:
        json.dump(combined_json, json_file)

#                  0.01 ,sample_size_bias, cluster, fairOD, alpha, gamma, matrix
def save_fairod(bias_rate, bias_type,data_type, method, seed, alpha, gamma, matrix):
    folder_path = f'result/{method}_hparam/{bias_type}/{data_type}/seed_{seed}' # result/FairOD_hparam/sample_size_bias/cluster/seed_?/biasRate_?.json

    os.makedirs(folder_path, exist_ok=True)

    # get original data in the file
    try:
        with open(os.path.join(folder_path, f'biasRate_{bias_rate}.json'), 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}  # If the file does not exist, start with an empty dictionary

    # add new information to the data
    key = f'{alpha}-{gamma}'
    data[key] = matrix

    # rewrite the new data into the file
    with open(os.path.join(folder_path, f'biasRate_{bias_rate}.json'), 'w') as file:
        json.dump(data, file)

    # if os.path.isdir(folder_path):
    #     print(f"The directory '{folder_path}' exists.")
    #     # Write the updated data back to the file
    #     with open(os.path.join(folder_path, f'{bias_rate}.json'), 'w') as file:
    #         json.dump(data, file) 
    # else:
    #     print(f"The directory '{folder_path}' does not exist.")
    #     os.makedirs(folder_path)
        
            #sample_size_bias, scatter, FairOD,  
def save_alpha_gamma(path, seed, alpha, gamma):
    #folder_path = f'../result/hparam_value/{bias_type}/{data_type}' # result/hparam_value/sample_size_bias/scatter
    folder_path = path # result/hparam_value/sample_size_bias/scatter/FairOD
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print("Folder already exists")
    
    data = {'alpha_list': alpha, 'gamma_list': gamma}
     # Open a file for writing
    with open(os.path.join(folder_path, f'seed_{seed}.json'), "w") as json_file:
        json.dump(data, json_file)

def save_distance(path, dis):
    folder_path = path # result/FairOD_distance/sample_sie_bias/scatter/distance.json
    # create dictory
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, f'distance.json'), "w") as json_file:
        json.dump(dis, json_file)


def save_AE(bias_rate, bias_type,data_type, method, seed, matrix):
    folder_path = f'result/{method}_result/{bias_type}/{data_type}/seed_{seed}' # result/AE_result/sample_size_bias/cluster/seed_?/biasRate_?.json

    os.makedirs(folder_path, exist_ok=True)

    # get original data in the file
    try:
        with open(os.path.join(folder_path, f'biasRate_{bias_rate}.json'), 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}  # If the file does not exist, start with an empty dictionary


    # rewrite the new data into the file
    with open(os.path.join(folder_path, f'biasRate_{bias_rate}.json'), 'w') as file:
        json.dump(matrix, file)


def save_dot(method, bias_type, data_type, bias, flag_ratio, tpr_ratio, auroc, key):
    folder_path = f'result/{method}_dot/{bias_type}/{data_type}' 
    dic = {}
    dic["flag_ratio"] = flag_ratio
    dic["tpr_ratio"] = tpr_ratio
    dic["auroc"] = auroc
    dic["key"] = key
    # create dictory
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, f'biasRate_{bias}.json'), "w") as json_file:
        json.dump(dic, json_file)


def new_save_alpha_gamma(path, alpha, gamma, auroc):
    #folder_path = f'../result/hparam_value/{bias_type}/{data_type}' # result/hparam_value/sample_size_bias/scatter
    folder_path = path # result/hparam_value/sample_size_bias/scatter/FairOD
    
    os.makedirs(folder_path, exist_ok=True)
    data = {'alpha_list': alpha, 'gamma_list': gamma, 'auroc': auroc}
     # Open a file for writing
    with open(os.path.join(folder_path, f'hparam.json'), "w") as json_file:
        json.dump(data, json_file)

def newsave(bias_type, data_type, method, parameter, 
          base_rate_a, base_rate_b, base_rate_whole,
          flag_rate_a, flag_rate_b, flag_ratio, 
          recall_a, recall_b, tpr_whole, true_positive_ratio, fpr_a, fpr_b, fpr_whole, fpr_ratio,
          ppr_a, ppr_b, ppr_whole, ppr_ratio, auroc):
    
    
    # initialize the dictionary
    nested_data = {}
    nested_data["Bias Type"] = bias_type
    nested_data["Data Type"] = data_type
    nested_data["Algorithm method"] = method
    nested_data["Base Rate A"] = base_rate_a
    nested_data["Base Rate B"] = base_rate_b
    nested_data["Base Rate whole"] = base_rate_whole
    nested_data["Group A"] = "Type: Gaussian, Xc mean: 0 variance: 1"
    nested_data["Group B"] = "Type: Gaussian, Xc mean: 0 variance: 1"
    nested_data["Group A_anomaly"] = "Type: Gaussian, Xc mean: 3 variance: 1"
    nested_data["Group B_anomaly"] = "Type: Gaussian, Xc mean: 3 variance: 1"
    
    
    # each parameter is a key
    for i in range(len(parameter)):
        config = {}
        config["parameter"] = parameter[i]
        config["Flag_rate_A"] = {f'e{index}': value for index, value in enumerate(flag_rate_a[i])}
        config["Recall_A"] = {f'e{index}': value for index, value in enumerate(recall_a[i])}
        config["FPR_A"] = {f'e{index}': value for index, value in enumerate(fpr_a[i])}
        config["Precision_A"] = {f'e{index}': value for index, value in enumerate(ppr_a[i])}
        config["Flag_rate_B"] = {f'e{index}': value for index, value in enumerate(flag_rate_b[i])}
        config["Recall_B"] = {f'e{index}': value for index, value in enumerate(recall_b[i])}
        config["FPR_B"] = {f'e{index}': value for index, value in enumerate(fpr_b[i])}
        config["Precision_B"] = {f'e{index}': value for index, value in enumerate(ppr_b[i])}
        # config["Flag_rate_overall"] = {f'e{index}': value for index, value in enumerate(list(zip(*flag_rate_whole))[i])}
        config["Recall_overall"] = {f'e{index}': value for index, value in enumerate(tpr_whole[i])}
        config["FPR_ovrall"] = {f'e{index}': value for index, value in enumerate(fpr_whole[i])}
        config["Precision_overall"] = {f'e{index}': value for index, value in enumerate(ppr_whole[i])}
        config["Flag_rate_ratio"] = {f'e{index}': value for index, value in enumerate(flag_ratio[i])}
        config["TPR_ratio"] = {f'e{index}': value for index, value in enumerate(true_positive_ratio[i])}
        config["FPR_ratio"] = {f'e{index}': value for index, value in enumerate(fpr_ratio[i])}
        config["PPR_ratio"] = {f'e{index}': value for index, value in enumerate(ppr_ratio[i])}
        config["auroc"] =  {f'e{index}': value for index, value in enumerate(auroc[i])}
        
        # add the dic into whole dictionary
        nested_data[f'parameter_{parameter[i]}'] = config

    
    folder_path = f'result/new_results/{bias_type}/{data_type}'
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Open a file for writing
    with open(os.path.join(folder_path, f'{method}.json'), "w") as json_file:
        json.dump(nested_data, json_file)

    

            # sample_size, scatter, FairOD, df, key
def save_data(bias_type,data_type, method, df, key):
    folder_path = f'result/data/{bias_type}/{data_type}/{method}' # /result/new_data/sample_size_bias/scatter/FairOD/0.001-0.04.json
    os.makedirs(folder_path, exist_ok=True)

    json_df = df.to_json(orient='records')
    combined_json = {
    "data": json_df
        }  
    # Open a file for writing
    with open(os.path.join(folder_path, f'{key}.json'), "w") as json_file:
        json.dump(combined_json, json_file)


def save_deepAE_result(method, bias_type, data_type, seed, bias_rate, metric):
    folder_path = f'result/{method}_deep_result/{bias_type}/{data_type}/seed_{seed}' # result/FairOD_hparam/sample_size_bias/cluster/seed_?/biasRate_?.json

    os.makedirs(folder_path, exist_ok=True)

    # get original data in the file
    try:
        with open(os.path.join(folder_path, f'biasRate_{bias_rate}.json'), 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}  # If the file does not exist, start with an empty dictionary

    data['matrix'] = metric

    # rewrite the new data into the file
    with open(os.path.join(folder_path, f'biasRate_{bias_rate}.json'), 'w') as file:
        json.dump(data, file)
