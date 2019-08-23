import numpy as np 
import pandas as pd 

import os 
import json

from scipy import stats
from pathlib import Path 

# simulation configurations path
configs_path = "sim_configs/std_simulation_configs.json"

# load the configurations
with open(configs_path) as json_file:
    configs = json.load(json_file)

# create the simulation data destination path
destination_path = Path(configs["destination_path"])
if destination_path.exists():
    raise RuntimeError("Destination directory already exists")
else:
    os.mkdir(destination_path)


# sample sizes to explore
sample_sizes = configs["sample_sizes"]

# minimum and maximum values for the standard deviation
min_scale = configs["min_scale"]
max_scale = configs["max_scale"]

# number of points between the minimum and maximum standard deviation
granularity = configs["granularity"]

# number of iterations
num_iterations = configs["num_iterations"]


scales = np.linspace(min_scale, max_scale, granularity)
df_list = list()


# simulation loop
for sample_size in sample_sizes:
    df = pd.DataFrame(data=scales, columns=["std_data"])
    
    print("Simulating data for sample size {}".format(sample_size))

    for i in range(num_iterations):
        ks_results = list()
        
        # calculate the KS test p-value
        for scale in scales:
            s0 = np.random.normal(loc=0, scale=1, size=sample_size)
            s_test = np.random.normal(loc=0, scale=scale, size=sample_size)
            ks_results.append(stats.ks_2samp(s0, s_test)[1])
            
        # add results to dataframe
        df["iter_"+str(i)] = ks_results
        
    # calculate mean values across simulation
    df["mean_res"] = df[df.columns[1:]].mean(axis=1)
    df["std_res"] = df[df.columns[1:]].std(axis=1)

    # append results to dataframe list
    df_list.append(df)

# saving data to files
for df, sample_size in zip(df_list, sample_sizes):
    df.to_csv(str(destination_path) + "/size_"+str(sample_size), header=True, index=False)