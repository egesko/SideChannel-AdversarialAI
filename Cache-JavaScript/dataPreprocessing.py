from csv import writer
import csv
import numpy as np
import os




def append_list_as_row(target_file_name, source_file_name):
	with open(target_file_name,'a',newline = '\n') as fd:
		with open(source_file_name ,newline='') as sourceFile:

			data_list =  list(np.loadtxt(sourceFile, delimiter = ','))
			data_array = np.array(data_list)

			
			fd.write('\n')
			i = 0
			for data in data_list:
				if(i == 0):
					fd.write(str(data))
				else:
					fd.write(','+str(data))
				i += 1

def change_extremes_in_csv(target_file_name):
	with open(target_file_name,newline = '') as sourceFile:
		data_list = list(np.loadtxt(sourceFile, delimiter = ','))

		for row_index, row in enumerate(data_list):
			for col_index, item in enumerate(row):
				if(item > 1):
					data_list[row_index][col_index] = 1

		with open("train_normalized.csv","w",newline='') as my_csv:
		    csvWriter = csv.writer(my_csv,delimiter=',')
		    csvWriter.writerows(data_list)


def normalize(target_file_name,max,min):
	with open(target_file_name,newline = '') as sourceFile:
		data_list = list(np.loadtxt(sourceFile, delimiter = ','))

		for row_index, row in enumerate(data_list):
			for col_index, item in enumerate(row):
					data_list[row_index][col_index] = (item-min)/(max-min) #normalizing function, change if necessary

		with open("train_val_normalized.csv","w",newline='') as my_csv:
		    csvWriter = csv.writer(my_csv,delimiter=',')
		    csvWriter.writerows(data_list)



def create_DataSet_From_Folder(folder,valAmt):

	websiteAmount = len(os.listdir(folder))
	print('websiteAmount: '+ str(websiteAmount))


	listForAmount = os.listdir(folder)
	dataPerWebsite = len(os.listdir(folder + '/' + listForAmount[0]))

	for i in range(dataPerWebsite):
		for websiteName in os.listdir(folder):
			
			mylist = os.listdir(folder + '/' + websiteName)

			print(len(mylist))
			

			
		# for filename in os.listdir(folder + '/' + folderName):
		# 	append_list_as_row(target,folder+ '/' + folderName + '/' +filename)




#create_DataSet_From_Folder('try3')
#append_list_as_row('ssdec21-13_1629917674731.csv','ssdec21-13_1629917644600.csv')

#normalize('train_val.csv',29.7,1.3)



