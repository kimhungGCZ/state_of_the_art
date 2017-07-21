
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import data_generation as DG


data, data_error = DG.generate_random_data(100,100,10)
raw_data = data
# Add X to real data
# data = np.reshape(raw_data, (-1,1))
data = (data-min(data))/(max(data)-min(data))

# # fit the model
clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
y_pred = clf.fit_predict(data)
raw_y_pred = clf.negative_outlier_factor_
#y_pred_outliers = y_pred[200:]

plt.plot(data)
plt.scatter(np.arange(len(raw_y_pred)),raw_y_pred,c='red')
#plt.scatter(np.arange(len(y_pred)),y_pred,c='red')

detected_outliners = (sorted(range(len(raw_y_pred)), key=lambda i: raw_y_pred[i])[:len(data_error)])
correct_percentage = np.mean( data_error != raw_data[sorted(detected_outliners)] ) * 100
print("The error percentage: ", correct_percentage , "%")
plt.show()
