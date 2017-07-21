import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.spatial import distance
import math as math
import data_generation as DG
import function_file as FF




##################################################
######## Generate sample data ####################
##################################################


# Self-generated data
# data_left = DG.generate_normal_time_series(1)
# data_right = DG.generate_normal_time_series(1)
# data_outlier = np.random.uniform(low=60, high=60, size=(10, 1))
# data = np.r_[data_left, np.concatenate(data_outlier), data_right]

# Common dataset
data, data_error = DG.generate_random_data(100,100,10)
raw_data = data

#data = np.array(raw_data, dtype=np.float64)
data = (data-min(data))/(max(data)-min(data))
temp_data = data
num_neighbor=2
my_list = list()
full_neigbour = list()
distance_list = list()
index = 0
while index < len(temp_data):
    if index < num_neighbor:
        left = temp_data[0:index]
        right = temp_data[index + 1:num_neighbor + index + 1 + num_neighbor - index]
    elif len(temp_data) - 1 - index < num_neighbor:
        left = temp_data[:index][-(num_neighbor+index - len(temp_data) + 3):]
        right = temp_data[index + 1: len(temp_data)]
    else:
        left = temp_data[:index][-num_neighbor:]
        right= temp_data[index+1:num_neighbor+index+1]
    result = np.append(left, right)
    my_list.append(result)
    full_neigbour.append(np.append(np.append(left,temp_data[index]),right))
    temp_value = distance.euclidean(result, temp_data[index])
    if temp_value> 0.5:
        temp_data = np.delete(temp_data, index, 0)
        index = index -1
    distance_list.append(temp_value)
    index = index + 1

print("---------")



distance_list_entropy = list();
for index, value in enumerate(my_list):
    distance_list_entropy.append(list(map(lambda x: distance.euclidean(data[index], x),value)))

#print(distance_list_entropy)
new_distance_list_entropy = list()
for i in full_neigbour:
    new_distance_list_entropy.append(FF.entropy(list(i)))

DEntropy_detected_outliners = (sorted(range(len(new_distance_list_entropy)), key=lambda i: new_distance_list_entropy[i])[-len(data_error):])
print("Distance Entropy percentge ", FF.correct_percentate(raw_data[DEntropy_detected_outliners],data_error))

# pattern_data = [1,2,3,4,5,6,7,1,2]
# pattern_data1 = [2,1,2,3,4,5,6,7,1]

#sample_corr = signal.correlate(pattern_data, pattern_data1)
# sample_corr =  np.correlate(pattern_data, pattern_data1,'full')
# max_post = np.argmax(sample_corr) - len(sample_corr)/2
# print("Cross Correlation: ", max_post)
#
#plt.plot(data)
#array_distance_list_entropy = np.fromiter(new_distance_list_entropy, dtype=np.int)
#plt.plot(new_distance_list_entropy)
#plt.plot(pattern_data)

# dao ham bac 1
der = FF.change_after_k_seconds(data, k=1)
# dao ham bac 2
sec_der = FF.change_after_k_seconds(der, k=1)

median_sec_der = np.median(sec_der)
std_sec_der = np.std(sec_der)

breakpoint_candidates = list(map(lambda x: np.abs(x[1] - median_sec_der) - np.abs(std_sec_der), enumerate(sec_der)))


plt.subplot(4, 1, 1)
plt.plot(data,'b.-')
plt.title('Original Data')

distance_list = np.array(distance_list,dtype=np.float64)
distance_list= (distance_list-min(distance_list))/(max(distance_list)-min(distance_list))
DE_detected_outliners = (sorted(range(len(distance_list)), key=lambda i: distance_list[i])[-len(data_error):])

print("Distance Euclidean percentage ", FF.correct_percentate(raw_data[DE_detected_outliners],data_error));
plt.subplot(4, 1, 2)
plt.plot(distance_list, 'b.-')
plt.title('Euclidean Distance weight')

plt.subplot(4, 1, 3)
plt.plot(new_distance_list_entropy, 'r.-')
plt.title('Entropy weight')

plt.subplot(4, 1, 4)
plt.plot(breakpoint_candidates,'b.-')
plt.title('Edge weight')
plt.show()