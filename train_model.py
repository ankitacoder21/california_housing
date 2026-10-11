import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_and_save_model():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    print(f"Model trained and saved as model.pkl")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


if __name__ == "__main__":
    train_and_save_model()
