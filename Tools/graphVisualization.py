import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data1 = pd.read_csv('1.csv', header=None).to_numpy()
data2 = pd.read_csv('2.csv', header=None).to_numpy()
data3 = pd.read_csv('3.csv', header=None).to_numpy()
data4 = pd.read_csv('4.csv', header=None).to_numpy()

averageArray = np.zeros((1,6000))


#CREATING ENTIRE ARRAY AVERAGE
# x = 0
# for i in data3[0]:
# 	#print(x)
# 	if(x==0):
# 		averageArray[x] = i
# 	else:
# 		averageArray[0][x] = ((averageArray[0][x-1]*x)+i)/(x+1)
# 	x=x+1

averageNumber = 7

#CREATING LAST 7 INDEX AVERAGE
x = 0
for i in data3[0]:
	if(x<averageNumber):
		sumVar = 0
		for j in range(x+1):
			sumVar += data3[0][x-j]
		averageArray[0][x] = sumVar/(x+1)
	else:
		sumVar = 0
		for j in range(averageNumber):
			sumVar += data3[0][x-j]
		averageArray[0][x] = sumVar/averageNumber
	x=x+1



fig = plt.figure(figsize=(100,5))
#plt.plot(results)

#plt.plot(data1[0])
#plt.plot(data2[0])
#plt.plot(data3[0])
#plt.plot(data4[0])
plt.plot(averageArray[0])

plt.legend()
plt.yticks(np.arange(1, 16,1))
plt.xticks(np.arange(1, 6000,100))
plt.grid()
#plt.axhline(linewidth=1, color='r')
plt.xlabel("5 ms")
plt.ylabel("PnP timing")
#figure(figsize=(8, 6), dpi=80)
fig.savefig('vis_test.png',dpi=200)
