import os
import glob
import csv
import pandas as pd
import autograd.numpy as np



def check_exists(arguments, results_folder = 'results/'):
    if os.path.exists(os.path.join(results_folder, arguments+'.csv')):
        return True
    return False


def save(result_dict, file_name, results_folder = 'results/'):
    # make results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    #create file path and dataframe
    csv_path = os.path.join(results_folder, file_name+'.csv')
    df = pd.DataFrame(result_dict)
    # append to the file if csv exists
    if not os.path.exists(csv_path):
        df.to_csv(csv_path,index=False)
    else:
        df.to_csv(csv_path,mode = 'a', header = False,index=False)
  

# generate dataframe that concat all csv in the same folder
def results_concat(results_folder):
    if not os.path.exists(results_folder):
        print('Results folder does not exist')
    all_csv = glob.glob(results_folder + '*.csv')

    name = []
    for file in all_csv:
        df = pd.read_csv(file, index_col=None)
        name.append(df)
    Frame = pd.concat(name, ignore_index= True)

    return Frame


