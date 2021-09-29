from csv import writer
import csv
import numpy as np




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



append_list_as_row('ssdec21-13_1629917674731.csv','ssdec21-13_1629917644600.csv')



