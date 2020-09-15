
from caseData import caseData 


# univariate bidirectional lstm example
import numpy
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
 

 
# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
#caseobj = caseData()
#casein = caseobj.newCaseArray()
#counter = 0 
class caseModel:

    def __init__(self, caseArray):
        self.caseArray = caseArray
    
    def train(self):
        
        
        # define input sequence
        raw_seq = self.caseArray
        # choose a number of time steps
        self.n_steps = 14
        # split into samples
        X, y = split_sequence(raw_seq, self.n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        # define model
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(self.n_steps, self.n_features)))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=200, verbose=0)
    
    def predict(self, noOfDays):
        self.noOfDays = noOfDays
        # fit model
        predictionArray = self.caseArray[-14:]
        counter = self.noOfDays
        counter2 = 0
        average = 0
        while counter > 0:
            while counter2 < 100:
                x_input = array(predictionArray[-14:])
                x_input = x_input.reshape((1, self.n_steps, self.n_features))
                yhat = self.model.predict(x_input, verbose=0)
                if yhat < 0:
                    yhat = 0
                print(yhat)
                average=average+yhat
                counter2+=1
            counter2=0
            predictionArray.append(int(average/100))
            counter-=1
            average=0
        return predictionArray
#plt.plot(casein)

#plt.ylabel('vic cases')
#plt.show()
