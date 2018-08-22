from keras.callbacks import Callback
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

def get_return(se_in,fre=1):
	if fre==1:
		se_out = se_in.pct_change()
	if fre==7:
		se_out = se_in.pct_change()+1
		se_out = se_out.resample('W').prod() - 1
	return se_out[1:]

#This Class is modified base on ZFTurbo's answer on Stackoverflow
#Link: https://stackoverflow.com/questions/37293642/
class EarlyStoppingByMSE(Callback):
    def __init__(self, monitor='mean_squared_error', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        #if current is None:
        #    warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %d: early stopping THR" % epoch)
            self.model.stop_training = True
