from pandas import read_csv,datetime,Series,DataFrame,set_option
from keras.models import Sequential
from keras.layers import Activation,SimpleRNN,Dense
from keras.activations import sigmoid
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats.stats import pearsonr
from matplotlib import pyplot
import myUtilities as utl
import numpy as np
import operator,time

start_time = time.time()
set_option('display.max_rows',500)

def generate_batches(x):
	batches.append(x.tolist())
	return 0

def generate_data(x):
	x=x.tolist()
	ref=x[0]
	x_seeds.append(list(map(operator.sub, x[1:-1], [ref]*p)))
	y_seeds.append([x[-1]-ref])
	ref_seeds.append([ref])
	return 0

ticker='TCS'
infile='data/'+ticker+'.csv'
output_folder='results/RNN/'+ticker+'/RNN_'

#Set the parameters for AR-MRNN(p,k) model
p=6
k=1

#Hyperparameters for the Neural Network
no_neural=16
no_epoch=1000000
batch_size=1
minloss=0.0002

batches=[]
x_train=[]
y_train=[]
x_test=[]
y_test=[]
refs=[]

df_data = read_csv(infile, header=0, parse_dates=[0], index_col=0)
se_data = utl.get_return(df_data['Adj Close'],7)
se_data.to_csv(output_folder+'.returns.csv')

se_data_len=len(se_data)
divided_point=int(se_data_len*1/2) #The point is dividing the train and test data
se_data.rolling(divided_point+1).apply(lambda x: generate_batches(x))

i=0
for batch in batches:
	i=i+1
	x_seeds=[]
	y_seeds=[]
	ref_seeds=[]
	Series(batch).rolling(p+k+1).apply(lambda x: generate_data(x))

	x_train.append(x_seeds[:-1])
	x_test.append(x_seeds)

	y_train.append(y_seeds[:-1])
	y_test.append(y_seeds)

	refs.append(ref_seeds)

x_train = np.array(x_train).reshape(se_data_len-divided_point,len(refs[0])-1,p)
y_train = np.array(y_train).reshape(se_data_len-divided_point,len(refs[0])-1,1)

x_test = np.array(x_test).reshape(se_data_len-divided_point,len(refs[0]),p)
y_test = np.array(y_test).reshape(se_data_len-divided_point,len(refs[0]),1)

#Build the model, and train (fit) it
model = Sequential()
model.add(SimpleRNN(no_neural, return_sequences=True, input_shape=(None, p), activation=lambda x: sigmoid(x)-0.5))
model.add(Dense(1,activation=lambda x: sigmoid(x)-0.5))
model.compile(loss = 'mean_squared_error', optimizer = 'RMSprop',metrics=['mae','mse'])
model.summary()
record=model.fit(x_train, y_train, epochs = no_epoch, batch_size = batch_size, callbacks=[utl.EarlyStoppingByMSE(monitor='mean_squared_error',value=minloss)])

eva=model.evaluate(x_train, y_train, batch_size = batch_size)

train_mae=eva[1]
train_mse=eva[2]

predict=model.predict(x_test)
predict=predict.reshape(len(predict),len(refs[0]))
predict=predict[:,-1]
predict=predict.tolist()

y_test=y_test.reshape(len(y_test),len(refs[0]))
y_test=y_test[:,-1]
y_test=y_test.tolist()

test_mae=mean_absolute_error(y_test, predict)
test_mse=mean_squared_error(y_test, predict)
corr=pearsonr(y_test,predict)[0]

print('Train MAE:',train_mae)
print('Train MSE:',train_mse)
print('Test MAE:',test_mae)
print('Test MSE:',test_mse)
print('Test Correlation:',corr)
print('------------------------')

predict_refs=se_data.tolist()[-(p+k+len(predict)):-(p+k)]
predict=list(map(operator.add, predict, predict_refs))
y_test=list(map(operator.add, y_test, predict_refs))

#print(len(predict))
#print(len(y_test))

pyplot.clf()
pyplot.plot(predict)
pyplot.plot(y_test)
pyplot.gca().legend(('Predict','Real'))
pyplot.savefig(output_folder+'.predict.RNN.png')
print('Plot(Predict Returns) saved!')

df_predict = DataFrame({'Real': y_test,'Predict':predict},index=se_data.index[divided_point:])
df_predict.to_csv(output_folder+'.predict.RNN.csv')
print('Results(Predict Returns) Saved!')

model.save(output_folder+'.h5')
print('Model Saved!')

f = open(output_folder+'.txt','w')
f.write('Input File = '+infile+'\n')
f.write('No. Neural = '+str(no_neural)+'\n')
f.write('------------------------\n')
f.write('Batch Size = '+str(batch_size)+'\n')
f.write('No. Epoch = '+str(no_epoch)+'\n')
f.write('Trained Epoch = '+str(len(record.history['loss']))+'\n')
f.write('------------------------\n')
f.write('Train MAE = '+str(train_mae)+'\n')
f.write('Train MSE = '+str(train_mse)+'\n')
f.write('Test MAE = '+str(test_mae)+'\n')
f.write('Test MSE = '+str(test_mse)+'\n')
f.write('Test Return Corelation = '+str(corr)+'\n')
f.write('------------------------\n')
f.write('Divided Point = '+str(divided_point)+'\n')
f.close()
print('Parameters Saved!')

#print('Divided point:',divided_point)
#pyplot.clf()
#pyplot.plot(predict)
#pyplot.plot(y_test)
#pyplot.gca().legend(('Predict','Real'))
#pyplot.show()

finish_time = time.time()
mins, secs = divmod(finish_time-start_time, 60)
print('Total running time: ',mins, 'minutes,',round(secs,2),'seconds')
