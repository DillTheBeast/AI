import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

reg = LinearRegression().fit(X, y)
print("Score: ", reg.score(X, y))  # reg.score(X, y)
print("Prediction: ",reg.predict(np.array([[3, 5]])))

# print("X Training:\n", X_train)
# print("Y Training:\n", y_train)
# print("X Testing:\n", X_test)
# print("Y Testing:\n", X_test)


# X = input = date = A
# Y = output = Average of high and low = D + E