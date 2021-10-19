from csv import writer
import csv
import numpy as np
import os




def append_list_as_row(target_file_name, source_file_name): #Helper function. Purpose: appending one csv file to another
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

def change_extremes_in_csv(target_file_name,num): #If you want to change datapoints above a predefined number, use this function
	with open(target_file_name,newline = '') as sourceFile:
		data_list = list(np.loadtxt(sourceFile, delimiter = ','))

		for row_index, row in enumerate(data_list):
			for col_index, item in enumerate(row):
				if(item > num):
					data_list[row_index][col_index] = 1

		with open("train_normalized.csv","w",newline='') as my_csv:
		    csvWriter = csv.writer(my_csv,delimiter=',')
		    csvWriter.writerows(data_list)


def normalize(target_file_name,source_file_name,max,min): # Normalizing all values in a csv file 
	with open(target_file_name,newline = '') as sourceFile:
		data_list = list(np.loadtxt(sourceFile, delimiter = ','))

		for row_index, row in enumerate(data_list):
			for col_index, item in enumerate(row):
					data_list[row_index][col_index] = (item-min)/(max-min) #normalizing function, change if necessary

		with open(source_file_name,"w",newline='') as my_csv:
		    csvWriter = csv.writer(my_csv,delimiter=',')
		    csvWriter.writerows(data_list)


def append_Label(target,number): # Helper function for appending labels (element of positive integers) to a target csv file
	with open(target,'a',newline = '\n') as fd:
		fd.write('\n')
		fd.write(str(number))


def create_DataSet_From_Folder(folder,finalTrainX,finalTrainY): #Creates DataSet from a directory. Also shuffles the samples.

	websiteAmount = len(os.listdir(folder))

	listForAmount = os.listdir(folder)
	dataPerWebsite = len(os.listdir(folder + '/' + listForAmount[0]))

	for i in range(dataPerWebsite):
		num = 0
		for websiteName in os.listdir(folder):
			
			dataListForWebsite = os.listdir(folder + '/' + websiteName)

			append_list_as_row(finalTrainX,folder+'/'+websiteName+'/'+dataListForWebsite[i])
			append_Label(finalTrainY,num)
			num += 1
			
		


create_DataSet_From_Folder('try3','trainX.csv','trainY.csv')
#append_list_as_row('ssdec21-13_1629917674731.csv','ssdec21-13_1629917644600.csv')
#normalize('train_val.csv',29.7,1.3)



