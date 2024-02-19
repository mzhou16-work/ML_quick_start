import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout, LeakyReLU
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MeanSquaredError, R2Score, MeanAbsoluteError
from torchsummary import summary
import numpy as np

import pandas as pd
import scipy.stats   as stats
import scipy.special as special

from netCDF4 import Dataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def cal_sca_ang(sza, saz, vza, vaz):

	# calculate the the scattering geometry of the each pixel
	idx = np.where(saz < 0)
	saz[idx] = 360.0 + saz[idx]

	idx = np.where(vaz < 0)
	vaz[idx] = 360.0 + vaz[idx]
	
	raz = abs(saz - vaz)
	idx = np.where(raz > 180.0)
	raz[idx] = 360.0 - raz[idx]

	raz = 180.0 - raz

	pi = np.pi * 1.0 / 180.0

	scValue = np.cos( sza * pi ) * np.cos( vza * pi ) + np.sin( sza * pi ) * np.sin( vza * pi ) * np.cos( (180.0 - raz) * pi )
	sca_ang_temp = np.arccos(scValue)/pi
	sca_ang = sca_ang_temp

	glValue = np.cos( sza * pi ) * np.cos( vza * pi ) + np.sin( sza * pi ) * np.sin( vza * pi ) * np.cos( raz * pi )
	gla_ang_temp = np.arccos(glValue)/pi
	gla_ang = gla_ang_temp

	return sca_ang, gla_ang



class MLP(LightningModule):
	def __init__(self, in_channels: int, out_channels: int, 
				  mlp_layers: int = 6, mlp_dim: int = 10, 
				  dropout: float = 0.2):
		super().__init__()
		self.save_hyperparameters()

		mlp = []
		
		mlp_dim = 2**mlp_dim
		
		layer_dims = np.zeros(mlp_layers-1).astype(int)
		max_layer_id = int((mlp_layers-2)/2)
		layer_dims[max_layer_id] = mlp_dim
		
		for idx in range(0, max_layer_id):

			layer_dims[idx] = mlp_dim/2**(max_layer_id-idx)

		for idx in range(max_layer_id, (mlp_layers-1)):
			layer_dims[idx] = mlp_dim/2**(idx-max_layer_id)
			
		print(f' - model middle layer structure {layer_dims}')
		
		
		for i in range(mlp_layers-1):
			
			current_dim = layer_dims[i]
			

			mlp += [Sequential(Linear(in_channels, current_dim), BatchNorm1d(current_dim), LeakyReLU(), Dropout(p=dropout),)]
			in_channels = current_dim
		mlp += [Linear(in_channels, out_channels)]
		
		
		self.mlp = ModuleList(mlp)

		
		for i in range(mlp_layers-1):
			torch.nn.init.xavier_normal_(self.mlp[i][0].weight.data, gain=1.0)
			torch.nn.init.zeros_(self.mlp[i][0].bias.data)


	def forward(self, x):
		for nn in self.mlp:
			x = nn(x)
		return x
		
	
#-----------------------------------------------------------------------
def normalization(data, name,  p = None, operation = 'forward', adj_params = []):
	
	def nBoxCox(x):
		"""
		Returns normalized (z-scores) BoxCox transform.
		"""
		(x_,p_)=stats.boxcox(x)
		p = (p_, x_.mean(), x_.std() )
		x_ = (x_-p[1]) / p[2]

		return x_, p


	#-----------------------------------------------------------------------    
	def inv_nBoxCox(x,p):
		"""
		Inverse normalized (z-scores) BoxCox transform.
		"""
		x_ = p[1] + p[2] * x 
		x_ = special.inv_boxcox(x_,p[0])

		return x_, p	
	

	if operation == 'forward':
		# for trainning 
		
		if data.ndim == 1:
			x_, p_ = nBoxCox(data)
			p = [p_]
			print(f' - Normalize {name[i]}... coef: {p_}')
		
		
		if data.ndim == 2:
			p = []
			_, n_loops = data.shape
			x_ = np.full_like(data, np.nan)
		
			for i in range(n_loops):
# 				print(f' - Normalize {name[i]}, {np.nanmin(data[:, i])}')
				if name[i] in adj_params:
					offset = 50.
				else:
					offset = 0.
				x_[:, i], p_ = nBoxCox(data[:, i] + offset)
# 				print(f' - Normalize {name[i]}, coef: {p_}')

				p.append(p_)
				
	if operation == 'predict':
		# for prediction phase, translate the physical features to the model acceptable values
		if data.ndim == 1:
		
			
			if name[i] in adj_params:
				offset = 50.
			else:
				offset = 0.
		
			key = name[i]			
			ps = list(p[key])		
			x_ = special.boxcox(data + offset, ps[0])
			x_ =  (x_- ps[1]) / ps[2]
# 			print(f' - Normalization [Predict mode], {key}, {ps[0]},  {ps[1]}, {ps[2]}...')
		
		if data.ndim == 2:
			_, n_loops = data.shape
			
			x_ = np.full_like(data, np.nan)

			
			for i in range(n_loops):

				if name[i] in adj_params:
					offset = 50.
				else:
					offset = 0.

				key = name[i]			
				ps = list(p[key])

				x_[:, i] = special.boxcox(data[:, i]+ offset, ps[0])
				x_[:, i] = (x_[:, i]- ps[1]) / ps[2]
# 				print(f' - Normalization [Predict mode], {key}, {ps[0]},  {ps[1]}, {ps[2]}...')
						
	if operation == 'inverse':
		# for inversing the predict to the actual physical parameter...
		x_, p = inv_nBoxCox(data, p)

	return x_, p
		
#-----------------------------------------------------------------------
def process_ocean(df, order, coef, qa_dict, th_dict, verbose = False):
	
	
	norm_feat_name   = order['order_feat']
	
	encode_feat_name = order['order_one_hot_feat']	
	
	if verbose:
		print(f' -  Size of the raw ocean data: {len(df)}')
	df = df[ norm_feat_name + encode_feat_name + ['lines', 'samples', 'AOD', 'DNB_TOA_refl_std', 'BTM14_std', 'BTM15_std'] ]
	df = df.dropna()
	if verbose:
		print(f' - Size of the valid raw data {len(df)}...')

	
# 	features = order['order_feat'] + order['order_one_hot_feat']

	idx = (df['GOME2_LER_B10_mean'] == df['GOME2_LER_B10_mean'])
	
	#!!!
# 	print(f' - Filter invalid label...')
# 	for label in ['AOD']:
# 		idx = idx & (df['AOD'].values > 0)
	#!!!

	if verbose:
		print(f' - Apply threshold to segregate data...')
	for th_var in th_dict.keys():
		idx = idx & (df[th_var].values == df[th_var].values)  & \
					(df[th_var].values >  th_dict[th_var][0]) & \
					(df[th_var].values <= th_dict[th_var][1])
				
	if verbose:
		print(f' - Apply QA to filtering data...')
	for qa_var in qa_dict.keys():
		idx = idx & (df[qa_var].values == df[qa_var].values)  & \
					(df[qa_var].values >  qa_dict[qa_var][0]) & \
					(df[qa_var].values <= qa_dict[qa_var][1])


	new_df = df[idx]


	# tranform the angle to radius
	ang_names = ['solar_zenith_mean', 'sensor_zenith_mean','solar_azimuth_mean', 'sensor_azimuth_mean', 'Glint_angle', 'Scattering_angle',]
	for key in new_df.keys():
		if key in ang_names:
			new_df[key] = np.mod(new_df[key] / 180. * np.pi, 2.0 * np.pi) + 0.2

	if verbose:
		print(f' - Datasize after QA {len(new_df)}')
		print(f' - Checking NaN for dataset: {new_df.isnull().values.any()}')
		print(f' - Checking NEG for dataset: {(new_df.values < 0).any()}')
	

	one_hot_code = []
	for name in encode_feat_name:
		if verbose:
			print(f' - Apply one_hot encoding to feature {name}...')		
		one_hot = new_df[name].values.astype(int)
		one_hot = F.one_hot(torch.from_numpy(one_hot.reshape(-1,)), num_classes=33).detach().numpy()
		one_hot_code.append(one_hot)

	if verbose:
		print(f'')
	
		print(f' - Apply normalization to features...')	
	# Normalize the features

	adj_params = ['BTD_M14_mean', 'BTD_M14_med', 
	              'BTD_M15_mean', 'BTD_M15_med', 
	              'BTD_M14_2_mean', 'BTD_M14_2_med', 
	              'BTD_M15_2_mean', 'BTD_M15_2_med']	
	normed_feat, _ = normalization(new_df[norm_feat_name].values.astype(np.float64), norm_feat_name, p = coef, operation = 'predict', adj_params = adj_params)

	
	normed_feat = [normed_feat]	
	for data in one_hot_code:
		normed_feat.append(data)

	normed_feat = np.concatenate(normed_feat, axis = 1)

	return_idx= new_df[['lines', 'samples']].values
	
	return normed_feat, return_idx



#-----------------------------------------------------------------------
def load_model(MODEL_PATH, MODEL_NAME, model_type, device = 'cpu', verbose = False):
	
	print(f' - {MODEL_PATH + MODEL_NAME[model_type] + "/" + MODEL_NAME[model_type] + ".pt"}')
	checkpoint = torch.load(MODEL_PATH + MODEL_NAME[model_type] + '/' + MODEL_NAME[model_type] + '.pt',
							map_location=torch.device(device))

	label_coefficient   = checkpoint['label_dict']
	feature_coefficient = checkpoint['feature_dict']
	feature_order = checkpoint['order_dict']
	# dict_keys(['model_state_dict', 'config', 'normed_label', 'normed_feat', 'label_dict', 'feature_dict', 'order_dict'])
	
	last_layer = str(checkpoint['config']['mlp_layers'] - 1)
	model_config = checkpoint['config']
	
	model_config['dropout'] = 0
	
	# get the input size
	in_channels = np.shape(checkpoint['model_state_dict']['predict.mlp.0.0.weight'])[-1]
	out_channels = np.shape(checkpoint['model_state_dict']['predict.mlp.' + last_layer + '.weight'])[0]
	
	if verbose:
		print(f' - Number of input: {in_channels}, Number of output: {out_channels}')

	model = MLP(in_channels=in_channels, out_channels=out_channels, **model_config)


	model_state_dict = {}
	for key in checkpoint['model_state_dict'].keys():
		model_state_dict[ key[8:] ] = checkpoint['model_state_dict'][key]
	
	if verbose:
		print(f' - Loading {model_type} to {device}')
	model.load_state_dict(model_state_dict)	
	
	model.to(device)
	
	return model, label_coefficient, feature_coefficient, feature_order


#-----------------------------------------------------------------------
def process_bt(df, order, coef, qa_dict, th_dict, verbose = False):
	
	
	norm_feat_name   = order['order_feat']
	
	encode_feat_name = order['order_one_hot_feat']	
	
	if verbose:
		print(f' -  Size of the raw ocean data: {len(df)}')
	df = df[ norm_feat_name + encode_feat_name + ['lines', 'samples', 'AOD', 'DNB_TOA_refl_std', 'BTM14_std', 'BTM15_std'] ]
	df = df.dropna()
	if verbose:
		print(f' - Size of the valid raw data {len(df)}...')


	idx = (df['DNB_TOA_refl_std'] == df['DNB_TOA_refl_std'])
	

	if verbose:
		print(f' - Apply threshold to segregate data...')
	for th_var in th_dict.keys():
		idx = idx & (df[th_var].values == df[th_var].values)  & \
					(df[th_var].values >  th_dict[th_var][0]) & \
					(df[th_var].values <= th_dict[th_var][1])
				
	if verbose:
		print(f' - Apply QA to filtering data...')
	for qa_var in qa_dict.keys():
		idx = idx & (df[qa_var].values == df[qa_var].values)  & \
					(df[qa_var].values >  qa_dict[qa_var][0]) & \
					(df[qa_var].values <= qa_dict[qa_var][1])


	new_df = df[idx]
	
	# tranform the angle to radius
	ang_names = ['solar_zenith_mean', 'sensor_zenith_mean','solar_azimuth_mean', 'sensor_azimuth_mean', 'Glint_angle', 'Scattering_angle',]
	for key in new_df.keys():
		if key in ang_names:
			new_df[key] = np.mod(new_df[key] / 180. * np.pi, 2.0 * np.pi) + 0.2	
	
	if verbose:
		print(f' - Datasize after QA {len(new_df)}')
		print(f' - Checking NaN for dataset: {new_df.isnull().values.any()}')
		print(f' - Checking NEG for dataset: {(new_df.values < 0).any()}')
	

	one_hot_code = []
	for name in encode_feat_name:
		if verbose:
			print(f' - Apply one_hot encoding to feature {name}...')		
		one_hot = new_df[name].values.astype(int)
		one_hot = F.one_hot(torch.from_numpy(one_hot.reshape(-1,)), num_classes=33).detach().numpy()
		one_hot_code.append(one_hot)

	if verbose:
		print(f'')
	
		print(f' - Apply normalization to features...')	
	# Normalize the features
	adj_params = ['BTD_M14_mean', 'BTD_M14_med', 
	              'BTD_M15_mean', 'BTD_M15_med', 
	              'BTD_M14_2_mean', 'BTD_M14_2_med', 
	              'BTD_M15_2_mean', 'BTD_M15_2_med']
	normed_feat, _ = normalization(new_df[norm_feat_name].values.astype(np.float64), norm_feat_name, p = coef, operation = 'predict', adj_params = adj_params)

	
	normed_feat = [normed_feat]	
	for data in one_hot_code:
		normed_feat.append(data)

	normed_feat = np.concatenate(normed_feat, axis = 1)

	return_idx= new_df[['lines', 'samples']].values
	
	return normed_feat, return_idx




#-----------------------------------------------------------------------
def process_dt(df, order, coef, qa_dict, th_dict, verbose = False):
	
	
	norm_feat_name   = order['order_feat']
	
	encode_feat_name = order['order_one_hot_feat']	
	
	if verbose:
		print(f' -  Size of the raw ocean data: {len(df)}')
	df = df[ norm_feat_name + encode_feat_name + ['lines', 'samples', 'AOD', 'DNB_TOA_refl_std', 'BTM14_std', 'BTM15_std'] ]
	df = df.dropna()
	if verbose:
		print(f' - Size of the valid raw data {len(df)}...')


	idx = (df['DNB_TOA_refl_std'] == df['DNB_TOA_refl_std'])
	

	if verbose:
		print(f' - Apply threshold to segregate data...')
	for th_var in th_dict.keys():
		idx = idx & (df[th_var].values == df[th_var].values)  & \
					(df[th_var].values >  th_dict[th_var][0]) & \
					(df[th_var].values <= th_dict[th_var][1])
				
	if verbose:
		print(f' - Apply QA to filtering data...')
	for qa_var in qa_dict.keys():
		idx = idx & (df[qa_var].values == df[qa_var].values)  & \
					(df[qa_var].values >  qa_dict[qa_var][0]) & \
					(df[qa_var].values <= qa_dict[qa_var][1])


	new_df = df[idx]
	
	
	# tranform the angle to radius
	ang_names = ['solar_zenith_mean', 'sensor_zenith_mean','solar_azimuth_mean', 'sensor_azimuth_mean', 'Glint_angle', 'Scattering_angle',]
	for key in new_df.keys():
		if key in ang_names:
			new_df[key] = np.mod(new_df[key] / 180. * np.pi, 2.0 * np.pi) + 0.2
	
	
	if verbose:
		print(f' - Datasize after QA {len(new_df)}')
		print(f' - Checking NaN for dataset: {new_df.isnull().values.any()}')
		print(f' - Checking NEG for dataset: {(new_df.values < 0).any()}')
	

	one_hot_code = []
	for name in encode_feat_name:
		if verbose:
			print(f' - Apply one_hot encoding to feature {name}...')		
		one_hot = new_df[name].values.astype(int)
		one_hot = F.one_hot(torch.from_numpy(one_hot.reshape(-1,)), num_classes=33).detach().numpy()
		one_hot_code.append(one_hot)

	if verbose:
		print(f'')
	
		print(f' - Apply normalization to features...')	
	# Normalize the features
	adj_params = ['BTD_M14_mean', 'BTD_M14_med', 
	              'BTD_M15_mean', 'BTD_M15_med', 
	              'BTD_M14_2_mean', 'BTD_M14_2_med', 
	              'BTD_M15_2_mean', 'BTD_M15_2_med']
	normed_feat, _ = normalization(new_df[norm_feat_name].values.astype(np.float64), norm_feat_name, p = coef, operation = 'predict', adj_params = adj_params)

	
	normed_feat = [normed_feat]	
	for data in one_hot_code:
		normed_feat.append(data)

	normed_feat = np.concatenate(normed_feat, axis = 1)

	return_idx= new_df[['lines', 'samples']].values
	
	return normed_feat, return_idx

#-----------------------------------------------------------------------

def read_feature_data(filename, thres = 3, verbose = False):

	ncid = Dataset(filename)
	
	ncid.set_auto_mask(False)

	params = [  'DNB_TOA_refl_mean', 'DNB_TOA_refl_std', 'DNB_TOA_refl_med', 
				'BTM14_mean', 'BTM14_std', 'BTM14_med', 
				'BTM15_mean', 'BTM15_std', 'BTM15_med',
				'BTD_M14_mean', 'BTD_M14_med', 
				'BTD_M15_mean', 'BTD_M15_med', 
				'BTD_M14_2_mean', 'BTD_M14_2_med', 
				'BTD_M15_2_mean', 'BTD_M15_2_med',			
				'Albedo_DNB_mean', 
				'Albedo_M3_mean', 
				'Albedo_M4_mean', 
				'Albedo_M5_mean', 
				'Albedo_M7_mean', 
				'Albedo_M8_mean', 
				'Albedo_M11_mean',
				'GEOS_FP_WS_mean', 
				'GOME2_LER_B10_mean', 
				'sensor_zenith_mean', 
				'sensor_azimuth_mean', 'solar_azimuth_mean', 'solar_zenith_mean', 
				'Integer_Cloud_Mask_mean', 'land_water_mask_mean', 
				'latitude', 'longitude', 'AEROSOL_TYPE', 
				'AOD', 'AOD_DB_QA', 'AOD_DT_QA']

	data_dict = {}
	for param in params:
		data_dict[param] = ncid[param][:]

	data_dict['Scattering_angle'], data_dict['Glint_angle'] = cal_sca_ang(data_dict['solar_zenith_mean'], data_dict['solar_azimuth_mean'],
																		  data_dict['sensor_zenith_mean'], data_dict['sensor_azimuth_mean'])
																		  
	ocean_idx = np.where( (ncid['land_water_mask_mean'][:] <= 0))
	
# 	#!!!
# 	ocean_idx = np.where( (ncid['land_water_mask_mean'][:] <= 0) & (ncid['AOD'][:] == ncid['AOD'][:]))
	
	bt_idx = np.where( (ncid['land_water_mask_mean'][:] >= 1)  & \
					   (ncid['Albedo_M11_mean'][:] > 0.15 )   & \
					   (ncid['Albedo_M11_mean'][:] <= 0.6) )
	
	dt_idx = np.where( (ncid['land_water_mask_mean'][:] >= 1)  & \
		               (ncid['Albedo_M11_mean'][:] >0 )        & \
		               (ncid['Albedo_M11_mean'][:] <= 0.15) )
# 	if verbose:
	print(f' - Find {np.shape(ocean_idx)[1]} ocean pixels, {np.shape(bt_idx)[1]} BT pixels, {np.shape(dt_idx)[1]} DT pixels')

	
	if np.size(ocean_idx)>0:
		# done reading, turning the data into a tabular...
		table_data = {}

		for param in data_dict.keys():
			table_data[param] = data_dict[param][ocean_idx]
		table_data['lines']   = ocean_idx[0]
		table_data['samples'] = ocean_idx[1]	
			
		ocean_df = pd.DataFrame.from_dict(table_data,'columns')
	else:
		ocean_df = None
		ocean_idx = None

	if np.size(bt_idx)>0:
		# done reading, turning the data into a tabular...
		table_data = {}

		for param in data_dict.keys():
			table_data[param] = data_dict[param][bt_idx]
		table_data['lines']   = bt_idx[0]
		table_data['samples'] = bt_idx[1]
			
		bt_df = pd.DataFrame.from_dict(table_data,'columns')
		
		bt_df['NDVI'] = (bt_df['Albedo_M7_mean'].values - bt_df['Albedo_M5_mean'].values) / (bt_df['Albedo_M7_mean'].values + bt_df['Albedo_M5_mean'].values)
		
	else:
		bt_df = None
		bt_idx = None


	if np.size(dt_idx)>0:
		# done reading, turning the data into a tabular...
		table_data = {}

		for param in data_dict.keys():
			table_data[param] = data_dict[param][dt_idx]
		table_data['lines']   = dt_idx[0]
		table_data['samples'] = dt_idx[1]
		
		dt_df = pd.DataFrame.from_dict(table_data,'columns')
		dt_df['NDVI'] = (dt_df['Albedo_M7_mean'].values - dt_df['Albedo_M5_mean'].values) / (dt_df['Albedo_M7_mean'].values + dt_df['Albedo_M5_mean'].values)
		
	else:
		dt_df = None
		dt_idx = None

	ncid.close()
	
	return data_dict, ocean_df, bt_df, dt_df
	


#-----------------------------------------------------------------------
def write_nc4(savename, nData, **kwargs):
	import sys
	import time
	import numpy as np
	import netCDF4 as nc
	from netCDF4 import Dataset
	
	
	
	print(' - Writing', savename)
	source		= kwargs.get('source', '')
	string2attr	= kwargs.get('string2attr', True)
	
	ncid = Dataset(savename, 'w', format='NETCDF4' )
	
	ncid.history = "Created " + time.ctime(time.time()) + ' ' + sys.argv[0]
	ncid.source = source

	nRow = np.shape(nData['latitude'])[0]
	nCol = np.shape(nData['latitude'])[1]
	ncid.createDimension('lat', nRow)
	ncid.createDimension('lon', nCol)
	
	if 'level' in nData.keys():
		nLev = np.size(nData['level'])
		ncid.createDimension('lev', nLev)	

	for key in nData.keys():
		dataType = type( nData[key] )
		if dataType == np.ndarray:
			
			nDim = len(np.shape(nData[key]))
# 			print(np.shape(nData[key]))
# 			print(' - Dimention of dataset', key, ':' , nDim)
			if nDim == 3:
				tempInstance = ncid.createVariable( key,'f4', ('lev', 'lat', 'lon'), zlib=True,complevel = 8)
				tempInstance[:, :, :] = nData[key][:, :, :]
				if str(key) + '_units' in nData.keys():
					tempInstance.units = nData[str(key) + '_units']
				
				if str(key) + '_long_name' in nData.keys():
					tempInstance.long_name = nData[str(key) + '_long_name']

			if nDim == 2:
				if np.shape(nData[key]) == (nRow, nCol):
					# usually we want to geo-locate the data...
					tempInstance = ncid.createVariable( key,'f4', ('lat','lon') ,zlib=True,complevel = 8)
					tempInstance[:, :] = nData[key][:, :]
				
					if str(key) + '_units' in nData.keys():
						tempInstance.units = nData[str(key) + '_units']
					
					if str(key) + '_long_name' in nData.keys():
						tempInstance.long_name = nData[str(key) + '_long_name']
				else:
					# if the data can not be geo-located...
					# we thus define the data by itself...
					rowName = 'nRow_' + key
					colName = 'nCol_' + key
					ncid.createDimension(rowName, np.shape(nData[key])[0])
					ncid.createDimension(colName, np.shape(nData[key])[1])
					tempInstance = ncid.createVariable( key, 'i4', (rowName, colName), zlib=True,complevel = 8)
					tempInstance[:, :] = nData[key][:, :]
				
					if str(key) + '_units' in nData.keys():
						tempInstance.units = nData[str(key) + '_units']
					
					if str(key) + '_long_name' in nData.keys():
						tempInstance.long_name = nData[str(key) + '_long_name']
			if nDim == 1:
				tempInstance = ncid.createVariable( key,'f4', ('lev'), zlib=True,complevel = 8)
				tempInstance[:] = nData[key][:]
				if str(key) + '_units' in nData.keys():
					tempInstance.units = nData[str(key) + '_units']
				
				if str(key) + '_long_name' in nData.keys():
					tempInstance.long_name = nData[str(key) + '_long_name']			
			
			
		if dataType == str:
			if '_long_name' in key:
				continue
			if '_units' in key:
				continue
				
			if string2attr:
				# here we can  write string into  attributes...
				ncid.setncattr(key,  nData[key])
			else:
				# here we can aslo write string into data 
				ncid.createDimension(key, len(nData[key]))
				tempInstance = ncid.createVariable(key, 'S1', (key))
				tempInstance[:] = nc.stringtochar(np.array([nData[key]], 'S'))					


	ncid.close()
	
	return	
	
	
 
def predict(model, model_name, df, order, label_coef, feat_coef, qa_dict, th_dict, data_dict = None, device = 'cpu'):

	if 'Ocean' in model_name:
		normed_feat, idx = process_ocean(df, order, feat_coef, qa_dict, th_dict)
		
	if 'Bright' in model_name:
		normed_feat, idx = process_bt(df, order, feat_coef, qa_dict, th_dict)
	
	if 'Dark' in model_name:
		normed_feat, idx = process_dt(df, order, feat_coef, qa_dict, th_dict)
	
	
	if np.shape(idx)[0] <= 0:
	
		return None

	model.eval()
	xx = torch.from_numpy(normed_feat)
	with torch.no_grad():
		batchX = xx.to(device)
		predictions = model(batchX.float()).cpu().numpy()

		predictions = predictions.astype(np.float64)

	
		predictions, _ = normalization(predictions, 'AOD', p = label_coef['AOD'],  operation = 'inverse')
		predictions = predictions[:, 0]
		
		if data_dict is not None:
			target = data_dict['AOD'][idx[:, 0], idx[:, 1]]

			valid_idx = np.where((predictions == predictions) & (target == target))
		
			if np.shape(valid_idx)[1] > 0:

				test_r2_scores = r2_score(target[valid_idx], predictions[valid_idx])

				print(f' - {model_name} R2', test_r2_scores)	

	return predictions, idx

	
	


















	
	
	
	