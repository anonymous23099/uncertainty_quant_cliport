import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

root_dir = '/home/bobwu/cliport/exps/multi-language-conditioned-cliport-n1000-calib-test'
bin_num = 10
number_of_steps = 1000
type_of_calib = 'calib'

def calculate_ece(dist_values, place_conf_values, bin_boundaries):
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions that fall into the current bin
        in_bin = np.logical_and(bin_lower <= place_conf_values, place_conf_values < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(np.array(dist_values)[in_bin] == 0)
            confidence_in_bin = np.mean(np.array(place_conf_values)[in_bin])
            delta = confidence_in_bin - accuracy_in_bin
            ece += np.abs(delta) * prop_in_bin
    return ece

# Step 1: Load the pickle file and retrieve the list of dictionaries
with open(os.path.join(root_dir, f'test_results_{number_of_steps}_{type_of_calib}.pkl'), 'rb') as file:
    data = pickle.load(file)

# Step 2: Extract the values of 'err1']['dist'] and 'err1']['place_conf_max'] into separate lists
dist_values = []
place_conf_values = []
for item in data:
    dist_values.append(item['err1']['dist'])
    place_conf_values.append(item['err1']['place_conf_max'])

# Step 3: Determine the maximum value of place_conf_max in the new list
max_place_conf = max(place_conf_values)

# Step 4: Generate 10 equally spaced bins in the range [0, max_place_conf]
bins = np.linspace(0, max_place_conf, num=bin_num)

# Step 5: Calculate the success rate for each bin
success_rates = []
for i in range(len(bins) - 1):
    bin_values = [dist_values[j] for j in range(len(dist_values)) if bins[i] <= place_conf_values[j] < bins[i+1]]
    success_rate = bin_values.count(0) / len(bin_values) if len(bin_values)!=0 else 0
    success_rates.append(success_rate)
# import pdb; pdb.set_trace() 

ece = calculate_ece(dist_values, place_conf_values, bins)
print(f'ece: {ece}')

# Step 6: Create a plot using the bins and success rates
plt.bar(bins[:-1], success_rates, width=(max_place_conf/bin_num), align='edge')

plt.xticks(bins, rotation=45)
plt.title(f'Reliability Diagram {number_of_steps}_{type_of_calib}')
plt.text(0.95, 0.95, f'ECE: {ece:.3f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

plt.tight_layout()
# import pdb; pdb.set_trace()
# Step 7: Save the plot
save_path = os.path.join(root_dir, f'reliability_diagram_{number_of_steps}_{type_of_calib}.png')
print(save_path)
plt.savefig(save_path)
