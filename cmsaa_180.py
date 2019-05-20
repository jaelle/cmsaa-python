from cmsaa import attentional_bias, optimize_prioritymap, rmse,plot_results,plot_results_w_test
from cmsaa import GoalMap, SaliencyMap, PriorityMap
import matplotlib.pyplot as plt
import csv
from scipy.integrate import simps

import numpy as np


data_180 = {}
test_180 = {}
stimuli_locations_180 = [-90,-45,0,45,90]


np_data = np.loadtxt('data/180degree.csv',delimiter=',',skiprows=1)
partitions = np.loadtxt('data/bootstrap_partitions.csv',dtype=int,delimiter=',')

x = np.array(stimuli_locations_180)

# Standard model:
# init_vals = [0.7662, 50, 0.760506149, 50]
# min_bounds = 0
# max_bounds = [0.79,1000000,5,1000000]

# GM Only model:
init_vals = [0.7662, 50]
min_bounds = 0
max_bounds = [0.79,1000000]

save_rows = []

for i in range(len(partitions)):
    
    training_set = []

    for col in partitions[i]:
        training_set += [np_data[col]]

    training_set = np.array(training_set)

    bootstrap_means = np.mean(training_set,axis=0)
    alldata_means = np.mean(np_data,axis=0)

    data_180['-90'] = bootstrap_means[0:5]
    data_180['0'] = bootstrap_means[5:10]
    data_180['90'] = bootstrap_means[10:15]

    test_180['-90'] = alldata_means[0:5]
    test_180['0'] = alldata_means[5:10]
    test_180['90'] = alldata_means[10:15]
    
    for attended_location in [-90,0,90]:

        # attentional bias derived from the mean reaction times at the attended location
        y = np.array(attentional_bias(data_180[str(attended_location)]))
        print(y)
        best_vals = optimize_prioritymap(attended_location, x, y, init_vals, min_bounds, max_bounds)

        degrees = np.arange(x[0],x[4],1)
        pm = PriorityMap(attended_location)
        pm.gmonly(degrees,*best_vals)

        train_error = rmse(x,pm.prioritymap,y)

        auc = simps(np.append(pm.prioritymap,[0.65,0.65,0.65]))

        test_y = np.array(attentional_bias(test_180[str(attended_location)]))
        test_error = rmse(x,pm.prioritymap,test_y)

        plot_results_w_test(x, y, test_y, pm, 'results_gmonly/images/' + str(attended_location) + '/180_' + str(attended_location) + '_bootstrap_' + str(i) + '.png')
        save_cols = [180,attended_location,i]
        save_cols = np.append(save_cols,best_vals)
        save_cols = np.append(save_cols,[train_error,test_error,auc])

        save_cols = np.array(save_cols,dtype=np.str)

        save_rows += [save_cols]

save_rows = np.array(save_rows).tolist()
# Standard Model:
# save_rows = [['standard location','stimuli location','bootstrap row','gm mag','gm stdev','sm mag','sm stdev','train error','test error','auc']] + save_rows
# GM Only Model:
save_rows = [['standard location','stimuli location','bootstrap row','gm mag','gm stdev','train error','test error','auc']] + save_rows
with open('results/180_params_gmonly.csv','w') as fp:
    writer = csv.writer(fp,lineterminator='\n')
    writer.writerows(save_rows)

print('Done!')
