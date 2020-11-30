# Tool import
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Model import
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder

# Lasso Model with GLMNET
import glmnet_python
from glmnet import glmnet
from glmnetPlot import glmnetPlot
from glmnetPredict import glmnetPredict
from glmnetCoef import glmnetCoef
from glmnetPrint import glmnetPrint

# Metric import
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,brier_score_loss,roc_curve,auc,plot_roc_curve
from scipy.stats import ks_2samp

# Rules extraction import
import re	

# GENERAL FUNCTIONS
def select_cols(var_list,initial_var,data):
	"""
	Select varialbes in var_list names not in initial_var.
	return the corresponding data
	"""

	data_cols = [] #selecte columns for data
	sel_vars = [] #selected var names

	for i,var in enumerate(var_list):
		if not (np.isin(var,initial_var)) and var != 'ij': #var in initial_var
			data_cols.append(i)
			sel_vars.append(var)

	return sel_vars,data[:,data_cols]


def string_to_intlist(string):
    return np.array(re.findall('\d+',string),dtype='int')


class PLTR():

	def __init__(self,n_predictors=None,lasso_modeling = 'glmnet'):
		self.tree_list = None # List of decision trees used to transform data
		self.ohe_list = None # List of OneHotEncoder trained to transform the output of decision tree
		self.n_predictors = n_predictors # Number of variables to consider : 
										 # if n_predictors = 5 we take the first 5  columns
										 # if None, all columns used

		self.var_names = None # variable_names used : number of columns are used as string
		self.sel_var_names = None 
		self.lasso_modeling = lasso_modeling # library for LASSO learning. (GLMNET VS SKLEARN).
		self.initial_var_names = None # Variable names in input data. Updated when training.
		self.lasso_best_lambda = None

	def fit_preproc(self,x_train,y_train):
		tree_list = []
		ohe_list = []
		var_names = []

		n_predictors = self.n_predictors if self.n_predictors != None else x_train.shape[1]
		self.n_predictors = n_predictors # number of trees

		if isinstance(x_train,pd.DataFrame):
			self.initial_var_names = x_train.columns


		for i in range(n_predictors):
			for j in range(i+1,n_predictors):
				# Input data with 2 variables (column)
				# print(i,j)
				x_input = x_train.iloc[:,[i,j]]
				tree_t = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=3).fit(x_input,y_train)
				# Output of tree data
				x_processed = tree_t.apply(x_input)
				# onehot encoder transform
				ohe = OneHotEncoder().fit(x_processed.reshape((-1,1)))
				# Setting var names : for understanding
				
				c_names = [str(i),str(j)]
				categ = ohe.categories_[0]
				feature = tree_t.tree_.feature
				feature_used = feature[feature>=0]
				if len(feature_used)>1:
					one = feature_used[0]
					two = feature_used[1]
					var = [c_names[one],'('+c_names[one]+','+c_names[two]+')'] # First name is the variable used in root and the secode one is the couple of variables used
				elif len(feature_used) == 1:
					# print(i,j)
					var = [c_names[feature_used[0]]]
				else: # variables are useless (both)
					var = ['ij']
				# Record trees and encoder
				tree_list.append(tree_t)
				ohe_list.append(ohe)
				var_names.append(var)

		self.tree_list = tree_list
		self.ohe_list = ohe_list
		self.var_names = var_names

		return self


	def transform(self,x_test,return_df = True):

		if self.tree_list == None or self.ohe_list == None or self.var_names == None:
			raise ValueError('The preprocessing is not trained. Use .fit_preproc.')

		tree_list = self.tree_list
		ohe_list = self.ohe_list
		var_names = self.var_names
		n_predictors = self.n_predictors

		n = 0
		res = np.array([])
		flatten_varnames = np.array([])

		for i in tqdm(range(n_predictors)):
			for j in range(i+1,n_predictors):
				# Input data with 2 variables (column)
				x_input = x_test.iloc[:,[i,j]]
				tree_t = tree_list[n]
				x_processed = tree_t.apply(x_input)

				ohe = ohe_list[n].transform(x_processed.reshape((-1,1))).toarray()
				# we take at most 2 columns  since the tree has at most 3 leaves
				ohe = ohe[:,:2]
				
				# add variables names
				var = var_names[n]
				# The case decision tree has only one plit
				if len(var) == 1:
					ohe = ohe[:,0].reshape((-1,1))
				
				var,ohe = select_cols(var,flatten_varnames,ohe)

				res = np.hstack([res,ohe]) if res.size else ohe
				flatten_varnames = np.append(flatten_varnames,var)

				n+=1


		self.sel_var_names = flatten_varnames

		if return_df:
			res = pd.DataFrame(res,columns = flatten_varnames)
			return res

		return res,flatten_varnames

	def plot_tree(self,tree_n):
		"""
		visualize the tree.
		"""

		plot_tree(self.tree_list[tree_n])

	def lasso_train(self,x_train,y_train,**glm_options):
		"""
		Lasso logistic regression with GLMNET.
		x_train : np or pd, must be the transformation with trees
		y_train : binary classification data
		**options : option for glmnet. For now, only nlambda is considered (default=20).
		"""

		nlambda = 100 # It is preferable to have large lambda to test. Else we encounter some trouble in glmnet (alm index out of bound.)
		if 'nlambda' in glm_options:
			nlambda = glm_options['nlambda']

		self.lasso_mod = glmnet(x = np.array(x_train).copy(),
				 y = np.array(y_train).copy(),
				 alpha = 1,nlambda = nlambda, # Alpha from elasticnet config: alph=1 is Lasso
				 family = 'binomial')
				 #parallel = True) # Not sur it works in glmnet


	def lasso_train_auto(self,x_train,y_train,x_val,y_val, #input data
							nlambda = 500 # Lasso parameters
							):
		"""
		Do automatically all necessary steps to train a PLTR model :
		1. Tranform data to tee-based binary data
		2. train the Lass model
		3. Select the hyperparameter (lambda) of Lasso model based on AUC value. x_val & y_val is used to cross-validate
		"""

		# Step 1
		self.fit_preproc(x_train,y_train)
		x_train_t = self.transform(x_train,return_df = True)

		# Step2
		self.lasso_train(x_train_t,y_train.astype('float'),nlambda = nlambda)

		# Step3
		x_val_t = self.transform(x_val,return_df=True)
		y_val_predict = self.lasso_predict(x_val_t)
		self.lasso_select_lambda(y_val,y_val_predict,'auc')

		return self

	def lasso_predict(self,x_test):
		"""
		lasso prediction with GLMNET. It returns several results with different lambda (regularization hparameter)

		x_tes : np or pd, trees transformed data.
		"""
		return glmnetPredict(self.lasso_mod,x_test,ptype = 'response') 

	def lasso_compute_metric_lambda(self,y_test,y_pred_proba,metric='auc'):
		"""
		select the best lambda by evaluating the metrics of y_res VS y_test.
		y_res : return of lasso prediction (matrix form with several lambdau).
		y_test : ground truth values
		metric : select the metric used to cross validate the lambdau.

		return the list of performance.
		"""


		metric_f = roc_auc_score
	
		if metric =='auc':
			metric_f = roc_auc_score

		else:
			print("The metric is unknown")
			return -1

		metric_list = []
		for y_pred in y_pred_proba.T:
			score = metric_f(y_test,y_pred)
			metric_list.append(score)

		return metric_list


	def lasso_select_lambda(self,y_test,y_pred_proba,metric='accuracy'):
		metric_list = self.lasso_compute_metric_lambda(y_test,y_pred_proba,metric = metric)
		self.lasso_best_lambda = np.argmax(metric_list)
		return np.argmax(metric_list)

	@property
	def lasso_coef(self):
		res = glmnetCoef(self.lasso_mod)
		if self.lasso_best_lambda is not None:
			res = res[:,self.lasso_best_lambda]
		return res

	# PLTR Analysis tool
	def find_tree(self,var_string,var_name_list = None):
		"""
		Find the tree where the variable is used.

		INPUT
		var_string : string, format should be : "(x,y,)" with x and y two positive integers
		var_name_list : list of string, specify variable names considered. If none we use the variable names used in decision tree learning
		"""

		#input data processing
		vn  = pd.DataFrame(self.var_names).values # self.var_names was list of numpy.

		
		# Computatinoal part
		tree_number = np.argwhere(vn == var_string)
		if tree_number.size == 0: # it is empty
		    raise ValueError('Tree not found')
		else:
		    tree_number = tree_number.reshape((-1,2))[0][0] # self.var_names a matrix with 2 columns

		return tree_number

	def var_to_string(self,var_string,var_name_list = None):
		"""
		Transcription of the columns name, which are in form of "(x,y)" meaning the variable x and y are coupled in the decision tree.
		return string with corresponding rules

		INPUT
		var_string : string, format should be : "(x,y,)" with x and y two positive integers
		RETURN string with corresponding rules.
		"""

		c_names = self.initial_var_names
		if isinstance(var_name_list,(list,np.ndarray)):
			c_names = var_name_list
		if c_names is None:
			raise ValueError('Cannot attribute variable names : var_names_list should be an input')

		# define threshold in the decision tree
		tree =  self.tree_list[self.find_tree(var_string,var_name_list)] # find tree with previous function.
		var_int = string_to_intlist(var_string)
		var_names = c_names[var_int]
		used_feat = np.argwhere(tree.tree_.feature >=0 ).flatten() 
		used_ths = tree.tree_.threshold[used_feat]
		rule = ""
		
		# define the sense of inequality
		signe = []
		if len(var_names) == 1:
			if tree.tree_.children_left[1]==-1:
				signe.append('<=')
			else:
				signe.append('>')
		elif len(var_names)==2:
			if tree.tree_.children_left[1] == -1: # if the left node of the racine is leaf
				signe.append('>')
				signe.append('<=')
			else:
				signe.append('<=')
				signe.append('<=')
		else:
			raise ValueError('Only a single or a couple of variables is consdered')

		# Concatenate as string
		for var,ths,s in zip(var_names,used_ths,signe):
		    rule += var+s+str(ths) +' & '
		rule = rule[:-3] 

		return rule 

	def extract_rules(self,var_name_list = None,lambda_n = None,return_coefs = True):
		"""
		Extract all rules for selected model (best_lambda should is used if lanbda_n = None, call with self.lasso_bset_lambda)
		return as a list of string

		INPUT :
		lambda_n : int or None, index of lambda selected. If none, the self.lasso_best_lambda is used.
		var_name_list : initial variable names.
		return_coefs : bool, if True, a dataframe with corresponding coefficients is returned.
		RETURN : String list, with rules
		"""

		# If the index of lambda is given, we extract rules from corresponding lasso model
		# Else it is the rules of best_lambda
		sel_lambda = lambda_n

		if sel_lambda is None:
			sel_lambda = self.lasso_best_lambda


		# Conpute the variables where the LASSO coefficient is not null.
		# list of couples of variables ( ex : ['(0,1)','0', ... , '(12,14)'] )
		sel_cols = np.argwhere(glmnetCoef(self.lasso_mod)[1:,sel_lambda]!=0)[:,0] #start from 1 because the first coef. corresponds to the intercept
		list_var_string = list(self.sel_var_names[sel_cols])

		#Compute rules
		rules = []
		for var_string in list_var_string :
			rule = self.var_to_string(var_string,var_name_list)
			rules.append(rule)


		if return_coefs:
			coefs = glmnetCoef(self.lasso_mod)[1:,sel_lambda][(sel_cols)]
			# return rules,coefs
			# print(coefs)
			# print(len(rules))
			rules = pd.DataFrame(np.array([rules,coefs]).T,columns = ['Rules','Lasso Coef.'])
			rules = rules.astype({'Lasso Coef.': 'float'})
		return rules


### Comparison and metric function
def metrics(y_test,y_pred_proba,plot_ROC = False,gini_treshold = 0.4):
	"""
	y_test : ground truth classification value (0 or 1)
	y_pred_proba : model estimated P(Y=1|X)

	return dictionnary of metrics
	"""
	y_pred = np.where(y_pred_proba >0.5,1,0)
	
	met = {
		'accu':accuracy_score(y_test,y_pred),
		'AUC':roc_auc_score(y_test,y_pred_proba),
		'BS': brier_score_loss(y_test,y_pred_proba),
		'CMat' : confusion_matrix(y_test,y_pred)
	}

	met['Gini'] = 2*met['AUC'] - 1

	# KS statistic import from scipy
  
	y0_index = np.where(y_test == 0)[0]
	y1_index = np.where(y_test == 1)[0]

	p_y0 = y_pred_proba[y0_index]
	p_y1 = y_pred_proba[y1_index]

	ks = ks_2samp(p_y0,p_y1)

	met['KS'] = ks

	# Partial Gini Index (PGI)
	# it computes the area under the curve between [a,b] and then :
	# The formula in the paper is inversed (regarding the FPR/TPR values)
	# We set a=0 and b=0.4 (initial setting in the paper)
	# The follwinog code
	# Some trouble to get the right definition of PGI

	fpr,tpr,tresholds = roc_curve(y_test,y_pred_proba)
	b = gini_treshold
	# b = gini_treshold
	# fpr2 = fpr[tpr<=b]
	# tpr2 = tpr[tpr<=b]
	# air = np.trapz(fpr2,tpr2)
	# # print(air)
	# pgi = 1 - ((2* air) / (b * b))
	# met['PGI'] = pgi

	# if plot_ROC:
	# 	plt.plot(tpr2,fpr2)
	# 	plt.show()


	# # PGI other computation
	# fpr2 = fpr[fpr<=b]
	# tpr2 = tpr[fpr<=b]
	# air = np.trapz(tpr2,fpr2)
	# pgi2 = (air)/(b - ((b*b)/2))
	# met['PGI2'] = pgi2

	# Third try for PGI
	# First compute the air with interpolation.
	 
	index_first = np.argmax(fpr>b)
	slope = (tpr[index_first] -  tpr[index_first - 1]) / (fpr[index_first] - fpr[index_first - 1])
	fpr_last = b
	tpr_last = slope *(b - fpr[fpr<=b][-1]) + tpr[index_first - 1]
	fpr_inter = np.append(fpr[fpr<=b],fpr_last)
	tpr_inter = np.append(tpr[fpr<=b],tpr_last)

	air = np.trapz(tpr_inter,fpr_inter)

	triangle = (b*b)/2
	nominator = air - triangle # the curve air above the random curve
	denominator = ( 1 - ((1-b)**2)) / 2 # The whole area above the random curve, a trapez area. 

	met['PGI'] = nominator/denominator

	# AUC from 0 to b

	met['Air_trapz'] = np.trapz(tpr[fpr<=b],fpr[fpr<=b])
	met['Partial AUC'] = roc_auc_score(y_test,y_pred_proba,max_fpr = 0.4)


	return met