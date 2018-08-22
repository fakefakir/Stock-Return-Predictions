from pandas import read_csv,datetime,Series,DataFrame,set_option
import numpy as np

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats.stats import pearsonr
from matplotlib import pyplot

import time
import myUtilities as utl

start_time = time.time()
tickers=['TCS','BHEL','WIPRO','AXISBANK','MARUTI','TATASTEEL']
results=DataFrame(index=['TCS','BHEL','WIPRO','AXISBANK','MARUTI','TATASTEEL'],columns=['MAE','MSE','Corr'])

output_folder='results/AR/AR_'

#Set the autoregessive order
lag=4

for ticker in tickers:
	#Parameters
	infile='data/'+ticker+'.csv'

	df_data = read_csv(infile, header=0, parse_dates=[0], index_col=0)
	se_data = utl.get_return(df_data['Adj Close'],7)
	#From the adjusted closed prices, to get the weekly(period=7) returns

	model = AR(se_data)
	model_fit = model.fit(maxlag=lag)

	predict=[]

	for i in range(lag+1,len(se_data)):
		hat_value=0
		for j in range(1,lag+1):
			hat_value=hat_value+model_fit.params[j]*se_data[i-j]
		hat_value = model_fit.params[0]+hat_value
		predict.append(hat_value)

	mse = mean_squared_error(se_data.tolist()[lag+1:],predict)
	mae = mean_absolute_error(se_data.tolist()[lag+1:],predict)
	corr=pearsonr(se_data.tolist()[lag+1:],predict)[0]

	results.at[ticker,'MSE']=mse
	results.at[ticker,'MAE']=mae
	results.at[ticker,'Corr']=corr
	'''
	print('Ticker:',ticker)
	print('MAE:',mae)
	print('MSE:',mse)
	print('Correlation:',corr)
	'''
	df_predict = DataFrame({'Real': se_data[lag+1:],'Predict':predict},index=se_data.index[lag+1:])
	df_predict.to_csv(output_folder+ticker+'.predict.AR.csv')
	print('Results(Predict Returns) Saved!')

	'''
	pyplot.clf()
	pyplot.plot(predict)
	pyplot.plot(se_data[lag+1:])
	pyplot.gca().legend(('Predict','Real'))
	pyplot.savefig(output_folder+'.predict.AR.png')
	pyplot.show()
	print('Plot(Predict Returns) saved!')
	'''

results.loc['Average'] = results.mean()
results.to_csv(output_folder+'.results.csv')
print('HPM Results Saved!')

finish_time = time.time()
mins, secs = divmod(finish_time-start_time, 60)
print('Total running time: ',mins, 'minutes,',round(secs,2),'seconds')
