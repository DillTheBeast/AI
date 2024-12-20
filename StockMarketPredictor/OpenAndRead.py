import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

root = tk.Tk()
root.title('CSV Reader and Linear Regression')
root.geometry('700x450')

# Text widget to display data and results
text = tk.Text(root, height=30, width=90)
text.grid(column=0, row=0)


def open_text_file():
    # file type    
    filetypes = [
        ('CSV files', '*.csv')
    ]
    f = fd.askopenfile(filetypes=filetypes)
    if f is not None:
        # Load CSV file into a DataFrame
        wholeFile = pd.read_csv(f)
        
        # Assuming the first column is the feature and the eighth column is the target
        X = wholeFile.iloc[:, [0]].values  # Feature (first column)
        y = wholeFile.iloc[:, 7].values    # Target (eighth column)
        
        for i in range(len(X)):
            X[i, 0] = datetime.strptime(X[i, 0], '%m-%d-%Y').timestamp()
        
        # Splitting the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initializing and fitting the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Making predictions on the test set
        future_date = datetime.strptime('2027-02-01', '%Y-%m-%d').timestamp()
        y_pred = model.predict([[future_date]])  # Reshaped into a 2D array

        text.insert(tk.END, f"Predicted result for 2027-02-01 is: {y_pred[0]}\n")


# Button to open file and trigger the regression
open_button = ttk.Button(
    root,
    text='Open a File',
    command=open_text_file
)

open_button.grid(column=0, row=1, padx=10, pady=10)

root.mainloop()


#OUTPUT: Predicted result for 2027-02-01 is: 81.74350209844226