import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class StartupPrediction:         # creating class


    def __init__(self, file_name):   # Constructor
        self.file_name = file_name
        self.df = None
        self.model = LinearRegression()


    def load_data(self):    # Load Dataset with Exception Handling
        try:
            self.df = pd.read_csv(self.file_name)
            print("Dataset loaded successfully\n")
            print(self.df.head())

        except FileNotFoundError:
            print("File not found. Please check file path.")

        except Exception as e:
            print("Error:", e)

    # Data Preprocessing
    def preprocess_data(self):
        self.df['State'] = self.df['State'].map({'New York': 0,'California': 1,'Florida': 2}).astype(int)

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        return train_test_split(
            X, y,
            test_size=0.3,
            random_state=42
        )


    def train_model(self, X_train, y_train):    # Train Model
        self.model.fit(X_train, y_train)
        print("\nModel trained successfully")

    # Model Evaluation
    def evaluate_model(self, X_train, X_test, y_train, y_test):

        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_loss = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_loss = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_accuracy = r2_score(y_train, y_train_pred)
        test_accuracy= r2_score(y_test, y_test_pred)

        print("\nTrain loss:", train_loss)
        print("Train Accuracy (R2 Score):", train_accuracy)

        print("\nTest loss:", test_loss)
        print("Test Accuracy (R2 Score):", test_accuracy)

obj = StartupPrediction(r"C:\Users\RAMADEVI SURIPAKA\Downloads\PythonProject1\50_Startups.csv")  # Object Create

obj.load_data()

X_train, X_test, y_train, y_test = obj.preprocess_data()

obj.train_model(X_train, y_train)

obj.evaluate_model(X_train, X_test, y_train, y_test)