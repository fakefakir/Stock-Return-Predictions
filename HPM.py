from pandas import read_csv,datetime,Series,DataFrame,set_option
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats.stats import pearsonr
from matplotlib import pyplot
import numpy as np
import time, operator

start_time = time.time()
tickers=['TCS','BHEL','WIPRO','AXISBANK','MARUTI','TATASTEEL']
results=DataFrame(index=['TCS','BHEL','WIPRO','AXISBANK','MARUTI','TATASTEEL'],columns=['MAE','MSE','Corr'])

output_folder='results/HPM/HPM_'
for ticker in tickers:
	RNN_file='results/RNN/'+ticker+'/RNN_.predict.RNN.csv'
	AR_file='results/AR/AR_'+ticker'.predict.AR.csv'

	df_RNN = read_csv(RNN_file, header=0, parse_dates=[0], index_col=0)
	se_RNN = df_RNN['Predict']

	length = len(se_RNN)

	df_AR = read_csv(AR_file, header=0, parse_dates=[0], index_col=0)
	se_AR = df_AR['Predict'][-length:]
	se_real = df_AR['Real'][-length:]

	list_RNN = se_RNN.tolist()
	list_AR = se_AR.tolist()
	list_real = se_real.tolist()

	best_wl=[]
	best_wnl=[]
	predict=[]
	for i in range(length):
		best_wl.append(0.0)
		best_wnl.append(0.0)
		min_error=99999.999
		for wl in np.arange(0,1,0.01):
		#Find the best weights (wl,wnl) for prediction at time t 
			wnl=1-wl
			hat_value=wl*list_AR[i]+wnl*list_RNN[i]
			error=abs(hat_value-list_real[i])
			if error<min_error:
				min_error=error
				best_wl[-1]=wl
				best_wnl[-1]=wnl
		predict.append(best_wl[-1]*list_AR[i]+best_wnl[-1]*list_RNN[i])

	results.at[ticker,'MSE']=mean_squared_error(list_real,predict)
	results.at[ticker,'MAE']=mean_absolute_error(list_real,predict)
	results.at[ticker,'Corr']=pearsonr(list_real,predict)[0]

	'''
	print('Performance of HPM for '+ticker+':')
	print('MSE:',mean_squared_error(list_real,predict))
	print('MAE:',mean_absolute_error(list_real,predict))
	print('Correlation:',pearsonr(list_real,predict)[0])
	print('Mean W_NL:',np.mean(best_wnl))
	print('Mean W_L:',np.mean(best_wl))
	print('---------------------')
	'''

	pyplot.clf()
	pyplot.plot(predict)
	pyplot.plot(list_real)
	pyplot.gca().legend(('Predict','Real'))
	pyplot.title('Prediction Output of HPM for '+ticker,fontname='Arial',fontweight='bold')
	pyplot.ylabel('Return',fontname='Arial')
	pyplot.xlabel('Time',fontname='Arial')
	pyplot.savefig(output_folder+ticker+'.predict.HPM.png')
	print(ticker,'Plot(Predict Returns) saved!')

results.loc['Average'] = results.mean()
results.to_csv(output_folder+'.results.csv')
print('HPM Results Saved!')

finish_time = time.time()
mins, secs = divmod(finish_time-start_time, 60)
print('Total running time: ',mins, 'minutes,',round(secs,2),'seconds')

