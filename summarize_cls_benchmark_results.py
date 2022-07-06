import os
import numpy as np
import pandas as pd

results_path = '/home/sungmanc/scripts/training_extensions/outputs/classification_sc_benchmark/hpo'

df = pd.DataFrame()
results = {'lineareval':{}, 'finetune':{}}
for exp_folder in os.listdir(results_path):
    splt_exp_folder = exp_folder.split('_')
    data_name, subset, model, mode = splt_exp_folder[0], splt_exp_folder[4], '_'.join(splt_exp_folder[5:7]), splt_exp_folder[-2]

    if int(subset) == 6:
        performance_file_path = os.path.join(results_path, exp_folder, 'results/performance_result.txt')
        try:
            performance_file = open(performance_file_path, 'r')
            
            if model not in results[mode].keys():
                results[mode][model] = {}

            if data_name not in results[mode][model].keys():
                results[mode][model][data_name] = []    

            results[mode][model][data_name].append(float(performance_file.readlines()[-1][8:14])*100)
        except:
            pass


print(results)
    