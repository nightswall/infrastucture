from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from myapp.gru_lstm_model_data_processing import getTrainLoaderFirstTime
from myapp.gru_lstm_model_data_processing import getTrainLoaderLater

from io import StringIO
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import csv

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from django.views.decorators.csrf import csrf_exempt

flag = False
datasetFileName = "myapp/Occupancy_source.csv"

class LSTMNet(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
		super(LSTMNet, self).__init__()
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers

		self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
		self.fc = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
		
	def forward(self, x, h):
		out, h = self.lstm(x, h)
		out = self.fc(self.relu(out[:,-1]))
		return out, h
	
	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
				  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
		return hidden
	
def evaluate(model, test_x,label_scaler):
	global flag
	model.eval()
	outputs = []
	#print(len(test_x))
	start_time = time.time()
	#print(test_x,test_y)
	inp = torch.from_numpy(np.array(test_x))
	if(flag==False):
		h = model.init_hidden(inp.shape[0])
		out, h = model(inp.to(device).float(), h)
		flag = True
		torch.save(h,'h_tensor.pt')
	else:
		h=torch.load('h_tensor.pt', map_location = device)
		out, h = model(inp.to(device).float(), h)
		torch.save(h,'h_tensor.pt')
	outputs.append(label_scaler.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
	#print(outputs)
	#print(labs)
	#targets.append(label_scaler.inverse_transform(labs.numpy().reshape(-1,1)))
	#print("Evaluation Time: {}".format(str(time.time()-start_time)))
	#MSE = 0
	#for i in range(len(outputs)):
	#	MSE += np.square(np.subtract(targets[i],outputs[i])).mean()
	#print(outputs[i][0],targets[i][0])
	#print("MSE: {}%".format(MSE*100))
	return outputs		
lookback = 3
device =torch.device('cpu')
temperature_model = LSTMNet(10, 256, 1, 2)

temperature_train_loader , sc, label_scaler , s_data= getTrainLoaderFirstTime(datasetFileName,0)
#inputs = np.zeros((1,lookback,10))
temperature_model.load_state_dict(torch.load('myapp/lstm_model_temperature_10.pt',map_location=device))
'''inputs = np.zeros((1,lookback,10))
temperature_model.load_state_dict(torch.load('myapp/lstm_model_temperature_10.pt',map_location=device))
s_data=pd.read_csv(datasetFileName,parse_dates=[0])
s_data['hour'] = s_data.apply(lambda x: x['date'].hour,axis=1)
s_data['dayofweek'] = s_data.apply(lambda x: x['date'].dayofweek,axis=1)
s_data['month'] = s_data.apply(lambda x: x['date'].month,axis=1)
s_data['dayofyear'] = s_data.apply(lambda x: x['date'].dayofyear,axis=1)
s_data = s_data.sort_values('date').drop('date',axis=1)
#s_data = s_data.drop('Light',axis=1)
#s_data = s_data.drop('Humidity',axis=1)
#s_data = s_data.drop('CO2',axis=1)
#s_data = s_data.drop('HumidityRatio',axis=1)
sc = MinMaxScaler()
sc.fit(s_data.values) #scaler for temperature
label_scaler = MinMaxScaler()
label_scaler.fit(s_data.iloc[:,0].values.reshape(-1,1))'''
@csrf_exempt
def predict_temperature(request):
	temp_data = request.POST.get("data")
	#print(temp_data)
	csv_data = StringIO("{}".format(temp_data))
	#csv_data = "/home/mrt/Desktop/diona/myproject/myapp/Occupancy.csv"
	# The scaler objects will be stored in this dictionary so that our output test data from the model can be re-scaled during evaluation
	# Store json file in a Pandas DataFrame
	columns=['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
	df = pd.read_csv(csv_data,header=None,names=columns,parse_dates=[0])

	#Apppend to csv file.

	# Check if the file exists
	file_exists = os.path.isfile('tempNewData.csv')

	# Append the DataFrame to the CSV file
	with open('tempNewData.csv', 'a') as f:
		df.to_csv(f, header=not file_exists, index=False)

	# Check csv file size if greater than 1000 call getTrainLoadeLater()

	csvFileName = "tempNewData.csv"
	line_count = 0

	with open(csvFileName, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			line_count += 1

	if line_count >= 101:
		train_loader , scaler_later , label_scaler_later, data = getTrainLoaderLater(csvFileName,datasetFileName,0)
		# delete new_temp
		## Will call train function
		os.remove(csvFileName)

	#df = df.sort_values('Humidity').drop('Humidity',axis=1)
	# Processing the time data into suitable input formats
	df['hour'] = df.apply(lambda x: x['date'].hour,axis=1)
	df['dayofweek'] = df.apply(lambda x: x['date'].dayofweek,axis=1)
	df['month'] = df.apply(lambda x: x['date'].month,axis=1)
	df['dayofyear'] = df.apply(lambda x: x['date'].dayofyear,axis=1)
	df = df.sort_values('date').drop('date',axis=1)

	#df = df.drop('Light',axis=1)
	#df = df.drop('Humidity',axis=1)
	#df = df.drop('CO2',axis=1)
	#df = df.drop('HumidityRatio',axis=1)
	data = sc.transform(df.values)
	if 'temperature_data.npz' in os.listdir('.'):
	# Load the existing file
		with np.load('temperature_data.npz',allow_pickle=True) as f:
	#		# Get the existing data
			existing_data = f['data']

			# TODO: TRASH OTHER DATA RECORDS BEFORE LOOKBACK.


	#		# Concatenate the existing data and the new data
			data = np.concatenate((existing_data, data))
			#print(data)
			# Save the updated data to the file
	np.savez('temperature_data.npz', data=data)
	with np.load('temperature_data.npz',allow_pickle=True) as data:
		# Get the data from the 'data' key
		all_data_temperature = data['data']
		#print(all_data_temperature)
	#print(lookback)
	count = len(all_data_temperature)
	if(count>lookback): #len(all_data_temperature)
		#print(all_data_temperature)
		inputs = np.array(all_data_temperature[count-lookback:count])## [count-lookback:count]
		inputs = np.expand_dims(inputs, axis=1)
		#print(inputs.shape)
		#print(label_sc.n_samples_seen_)
		prediction = evaluate(temperature_model,inputs,label_scaler)
		#print(prediction)
		json_prediction = str(prediction[0][0])
		#print(prediction[0][0].value())
		#print(json_prediction)
		#print((df['Temperature'].values)[0])
		if abs(float(json_prediction)-float((df['Temperature'].values)[0])) > 1.5:
			anomaly="Yes"
			response = HttpResponse(json.dumps({"prediction":json_prediction,"actual":str(float((df['Temperature'].values)[0])),"is_anomaly":str("WARNING AN ANOMALY DETECTED !!!!!")}) + "\n")
		else:
			anomaly="No"
			response = HttpResponse(json.dumps({"prediction":json_prediction,"actual":str(float((df['Temperature'].values)[0])),"is_anomaly":str(anomaly)}) + "\n")
		return response
	else:
		return JsonResponse({"available_after":(lookback-len(all_data_temperature))})#(lookback-len(all_data))
# Create your views here.


all_data_humidity=list()
humidity_model = LSTMNet(6, 256, 1, 2)
inputs = np.zeros((1,lookback,6))
humidity_model.load_state_dict(torch.load('myapp/lstm_model_humidity.pt',map_location=device))
r_data=pd.read_csv(datasetFileName,parse_dates=[0])
r_data['hour'] = r_data.apply(lambda x: x['date'].hour,axis=1)
r_data['dayofweek'] = r_data.apply(lambda x: x['date'].dayofweek,axis=1)
r_data['month'] = r_data.apply(lambda x: x['date'].month,axis=1)
r_data['dayofyear'] = r_data.apply(lambda x: x['date'].dayofyear,axis=1)
r_data = r_data.sort_values('date').drop('date',axis=1)
r_data = r_data.drop('Temperature',axis=1)
r_data = r_data.drop('Light',axis=1)
r_data = r_data.drop('CO2',axis=1)
r_data = r_data.drop('HumidityRatio',axis=1)
r_sc = MinMaxScaler() #scaler for humidity 
r_sc.fit(r_data.values)
r_label_sc = MinMaxScaler()
r_label_sc.fit(r_data.iloc[:,0].values.reshape(-1,1))

all_data_light=list()
light_model = LSTMNet(6, 256, 1, 2)
inputs = np.zeros((1,lookback,6))
light_model.load_state_dict(torch.load('myapp/lstm_model_light.pt',map_location=device))
l_data=pd.read_csv(datasetFileName,parse_dates=[0])
l_data['hour'] = l_data.apply(lambda x: x['date'].hour,axis=1)
l_data['dayofweek'] = l_data.apply(lambda x: x['date'].dayofweek,axis=1)
l_data['month'] = l_data.apply(lambda x: x['date'].month,axis=1)
l_data['dayofyear'] = l_data.apply(lambda x: x['date'].dayofyear,axis=1)
l_data = l_data.sort_values('date').drop('date',axis=1)
l_data = l_data.drop('Temperature',axis=1)
l_data = l_data.drop('Humidity',axis=1)
l_data = l_data.drop('CO2',axis=1)
l_data = l_data.drop('HumidityRatio',axis=1)
l_sc = MinMaxScaler() #scaler for light
l_sc.fit(l_data.values)
l_label_sc = MinMaxScaler()
l_label_sc.fit(l_data.iloc[:,0].values.reshape(-1,1))




all_data_co2=list()
co2_model = LSTMNet(6, 256, 1, 2)
inputs = np.zeros((1,lookback,6))
co2_model.load_state_dict(torch.load('myapp/lstm_model_co2.pt',map_location=device))
c_data=pd.read_csv(datasetFileName,parse_dates=[0])
c_data['hour'] = c_data.apply(lambda x: x['date'].hour,axis=1)
c_data['dayofweek'] = c_data.apply(lambda x: x['date'].dayofweek,axis=1)
c_data['month'] = c_data.apply(lambda x: x['date'].month,axis=1)
c_data['dayofyear'] = c_data.apply(lambda x: x['date'].dayofyear,axis=1)
c_data = c_data.sort_values('date').drop('date',axis=1)
c_data = c_data.drop('Temperature',axis=1)
c_data = c_data.drop('Humidity',axis=1)
c_data = c_data.drop('Light',axis=1)
c_data = c_data.drop('HumidityRatio',axis=1)
c_sc = MinMaxScaler() #scaler for co2
c_sc.fit(c_data.values)
c_label_sc = MinMaxScaler()
c_label_sc.fit(c_data.iloc[:,0].values.reshape(-1,1))

all_data_occupancy=list()
occupancy_model = LSTMNet(5, 256, 1, 2)
inputs = np.zeros((1,lookback,5))
occupancy_model.load_state_dict(torch.load('myapp/lstm_model_occupancy.pt',map_location=device))
h_data=pd.read_csv(datasetFileName,parse_dates=[0])

h_data['hour'] = h_data.apply(lambda x: x['date'].hour,axis=1)
h_data['dayofweek'] = h_data.apply(lambda x: x['date'].dayofweek,axis=1)
h_data['month'] = h_data.apply(lambda x: x['date'].month,axis=1)
h_data['dayofyear'] = h_data.apply(lambda x: x['date'].dayofyear,axis=1)
h_data = h_data.sort_values('date').drop('date',axis=1)
h_data = h_data.drop('Temperature',axis=1)
h_data = h_data.drop('Humidity',axis=1)
h_data = h_data.drop('CO2',axis=1)
h_data = h_data.drop('Light',axis=1)
h_data = h_data.drop('HumidityRatio',axis=1)
h_sc = MinMaxScaler() #scaler for occupancy
h_sc.fit(h_data.values)
h_label_sc = MinMaxScaler()
h_label_sc.fit(h_data.iloc[:,0].values.reshape(-1,1))



@csrf_exempt
def predict_occupancy(request):
	#model.eval()
	temp_data = request.POST.get("data")
	#print(temp_data)
	csv_data = StringIO("{}".format(temp_data))
	test_x = {}
	test_y = {}
	# Store json file in a Pandas DataFrame
	columns=['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
	df = pd.read_csv(csv_data,header=None,names=columns,parse_dates=[0])
	#df = df.sort_values('Humidity').drop('Humidity',axis=1)
	# Processing the time data into suitable input formats
	df['hour'] = df.apply(lambda x: x['date'].hour,axis=1)
	df['dayofweek'] = df.apply(lambda x: x['date'].dayofweek,axis=1)
	df['month'] = df.apply(lambda x: x['date'].month,axis=1)
	df['dayofyear'] = df.apply(lambda x: x['date'].dayofyear,axis=1)
	df = df.sort_values('date').drop('date',axis=1)
	df = df.drop('Humidity',axis=1)
	df = df.drop('Temperature',axis=1)
	df = df.drop('Light',axis=1)
	df = df.drop('CO2',axis=1)
	df = df.drop('HumidityRatio',axis=1)
	data = h_sc.transform(df.values)
	all_data_occupancy.append(data)
	#print(all_data_humidity)
	count = len(all_data_occupancy)-1
	#print(count)
	if(len(all_data_occupancy)>lookback):
		inputs = np.array(all_data_occupancy[count-lookback:count])
		prediction = evaluate(occupancy_model,inputs,h_label_sc)
		json_prediction = str(prediction[0][0])
		if abs(float(json_prediction)-float((df['Occupancy'].values)[0]))> 0.5:
			anomaly="Yes"
		else:
			anomaly="No"
		return JsonResponse({"prediction":json_prediction,"actual":str(float((df['Occupancy'].values)[0])),"is_anomaly":anomaly})
	else:
		return JsonResponse({"available_after":(lookback-len(all_data_occupancy))})#(lookback-len(all_data))

@csrf_exempt
def predict_co2(request):
	#model.eval()
	temp_data = request.POST.get("data")
	#print(temp_data)
	csv_data = StringIO("{}".format(temp_data))
	test_x = {}
	test_y = {}
	# Store json file in a Pandas DataFrame
	columns=['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
	df = pd.read_csv(csv_data,header=None,names=columns,parse_dates=[0])
	#df = df.sort_values('Humidity').drop('Humidity',axis=1)
	# Processing the time data into suitable input formats
	df['hour'] = df.apply(lambda x: x['date'].hour,axis=1)
	df['dayofweek'] = df.apply(lambda x: x['date'].dayofweek,axis=1)
	df['month'] = df.apply(lambda x: x['date'].month,axis=1)
	df['dayofyear'] = df.apply(lambda x: x['date'].dayofyear,axis=1)
	df = df.sort_values('date').drop('date',axis=1)
	df = df.drop('Humidity',axis=1)
	df = df.drop('Temperature',axis=1)
	df = df.drop('Light',axis=1)
	df = df.drop('HumidityRatio',axis=1)
	# Scaling the input data
	#print(df.values)
	data = c_sc.transform(df.values)
	all_data_co2.append(data)
	#print(all_data_humidity)
	count = len(all_data_co2)-1
	#print(count)
	if(len(all_data_co2)>lookback):
		inputs = np.array(all_data_co2[count-lookback:count])
		prediction = evaluate(co2_model,inputs,c_label_sc)
		json_prediction = str(prediction[0][0])
		if abs(float(json_prediction)-float((df['CO2'].values)[0])) > 10.0:
			anomaly="Yes"
		else:
			anomaly="No"
		return JsonResponse({"prediction":json_prediction,"actual":str(float((df['CO2'].values)[0])),"is_anomaly":anomaly})
	else:
		return JsonResponse({"available_after":(lookback-len(all_data_co2))})#(lookback-len(all_data))

@csrf_exempt
def predict_light(request):
	#model.eval()
	temp_data = request.POST.get("data")
	#print(temp_data)
	csv_data = StringIO("{}".format(temp_data))
	test_x = {}
	test_y = {}
	# Store json file in a Pandas DataFrame
	columns=['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
	df = pd.read_csv(csv_data,header=None,names=columns,parse_dates=[0])
	#df = df.sort_values('Humidity').drop('Humidity',axis=1)
	# Processing the time data into suitable input formats
	df['hour'] = df.apply(lambda x: x['date'].hour,axis=1)
	df['dayofweek'] = df.apply(lambda x: x['date'].dayofweek,axis=1)
	df['month'] = df.apply(lambda x: x['date'].month,axis=1)
	df['dayofyear'] = df.apply(lambda x: x['date'].dayofyear,axis=1)
	df = df.sort_values('date').drop('date',axis=1)
	df = df.drop('Humidity',axis=1)
	df = df.drop('Temperature',axis=1)
	df = df.drop('CO2',axis=1)
	df = df.drop('HumidityRatio',axis=1)
	data = l_sc.transform(df.values)
	all_data_light.append(data)
	#print(all_data_humidity)
	count = len(all_data_light)-1
	#print(count)
	if(len(all_data_light)>lookback):
		inputs = np.array(all_data_light[count-lookback:count])
		prediction = evaluate(light_model,inputs,l_label_sc)
		json_prediction = str(prediction[0][0])
		if abs(float(json_prediction)-float((df['Light'].values)[0])) > 50.0:
			anomaly="Yes"
		else:
			anomaly="No"
		return JsonResponse({"prediction":json_prediction,"actual":str(float((df['Light'].values)[0])),"is_anomaly":anomaly})
	else:
		return JsonResponse({"available_after":(lookback-len(all_data_light))})#(lookback-len(all_data))

@csrf_exempt
def predict_humidity(request):
	temp_data = request.POST.get("data")
	#print(temp_data)
	csv_data = StringIO("{}".format(temp_data))
	test_x = {}
	test_y = {}
	# Store json file in a Pandas DataFrame
	columns=['date','Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy']
	df = pd.read_csv(csv_data,header=None,names=columns,parse_dates=[0])
	#df = df.sort_values('Humidity').drop('Humidity',axis=1)
	# Processing the time data into suitable input formats
	df['hour'] = df.apply(lambda x: x['date'].hour,axis=1)
	df['dayofweek'] = df.apply(lambda x: x['date'].dayofweek,axis=1)
	df['month'] = df.apply(lambda x: x['date'].month,axis=1)
	df['dayofyear'] = df.apply(lambda x: x['date'].dayofyear,axis=1)
	df = df.sort_values('date').drop('date',axis=1)
	df = df.drop('Light',axis=1)
	df = df.drop('Temperature',axis=1)
	df = df.drop('CO2',axis=1)
	df = df.drop('HumidityRatio',axis=1)
	# Scaling the input data
	#print(df.values)
	data = r_sc.transform(df.values)
	all_data_humidity.append(data)
	count = len(all_data_humidity)-1
	if(len(all_data_humidity)>lookback):
		inputs = np.array(all_data_humidity[count-lookback:count])
		prediction = evaluate(humidity_model,inputs,r_label_sc)
		json_prediction = str(prediction[0][0])
		#print(prediction[0][0].value())
		#print(json_prediction)
		#print((df['Temperature'].values)[0])
		if abs(float(json_prediction)-float((df['Humidity'].values)[0])) > 2.5:
			anomaly="Yes"
		else:
			anomaly="No"
		return JsonResponse({"prediction":json_prediction,"actual":str(float((df['Humidity'].values)[0])),"is_anomaly":anomaly})
	else:
		return JsonResponse({"available_after":(lookback-len(all_data_humidity))})#(lookback-len(all_data))



