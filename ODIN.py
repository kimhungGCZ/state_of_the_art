import lib.odin.odin as ODIN
import numpy as np
import data_generation as DG
import matplotlib.pyplot as plt
import function_file as FF

K = 20

## Fake data
data, data_error = DG.generate_random_data(100,100,10,3)
raw_data = data
## Real data
# data, data_error = DG.realData()
# raw_data = data
# data = np.array(raw_data, dtype=np.float64)

data = (data-min(data))/(max(data)-min(data))
data = np.reshape(data,(-1,1))

odin = ODIN.Meandist(K, len(data_error))
result,weight = odin.fit(data)
print("ODIN error percentge ", FF.correct_percentate(raw_data[sorted(result)],data_error))

plt.plot(data)
plt.plot(weight)
#plt.scatter(result,weight,color='r')
plt.show();