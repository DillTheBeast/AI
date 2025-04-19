import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def open_text_file():
    wholeFile = pd.read_csv('/Users/dillonmaltese/Documents/git/AI/StockMarketPredictor/Amazon.com Stock Price History.csv')
    avg = (wholeFile['High'] + wholeFile['Low']) / 2
    return avg
    

def splitStuff(seqLength, file):
    x = []
    y = []
    for i in range(0,len(file)-seqLength):
        x.append(file[i:i + seqLength].values.tolist())
        y.append(file[i + 1:i + seqLength + 1].values.tolist())
        
    return x, y
    

avg = open_text_file()
seqLength = 6
x, y = splitStuff(seqLength, avg)

trainSize = int(len(x) * 0.8)
xTrain, xTest = x[:trainSize], x[trainSize:]
yTrain, yTest = y[:trainSize], y[trainSize:]

reg = LinearRegression().fit(xTrain, yTrain)
print("Score: ", reg.score(xTest, yTest))  # reg.score(X, y)
print("Prediction: ",reg.predict(np.array(xTest)))

mse = mean_squared_error(yTest, reg.predict(xTest))
print("Mean squared error: ", mse)

print(x[0])
print(y[0])