import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
        cwd = os.getcwd()
        file_path = cwd + "/dataset/life_stat.csv"
        lifestat = pd.read_csv(file_path)
        X = lifestat[["GDP per capita (USD)"]].values
        y = lifestat[["Life satisfaction"]].values
        lifestat.plot(kind='scatter', grid=True,
                x="GDP per capita (USD)", y="Life satisfaction")
        plt.axis([23_500, 62_500, 4, 9])
        plt.show()

        model = LinearRegression()
        # Train the model
        model.fit(X, y)
        # Make a prediction for Cyprus
        X_new = [[37_655.2]] # Cyprus' GDP per capita in 2020
        print(model.predict(X_new))


if __name__ == '__main__':
    main()
