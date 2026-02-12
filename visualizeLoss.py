import pickle
import matplotlib.pyplot as plt

result = pickle.load(open('result_large3(3)_128.pkl', 'rb'))
x = [40 * v[0] for v in result]
y = [v[1] for v in result]

plt.xlabel('batch')
plt.ylabel('Loss')
plt.plot(x,y)
plt.show()

