import mlflow
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

remote_server_uri = "http://127.0.0.1:8000"
mlflow.set_tracking_uri(uri=remote_server_uri)

exp_name = "practice mlflow"
mlflow.set_experiment(exp_name)


def load_data():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train(params):
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("params: ", params)
        print("accuracy", accuracy)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("Training Info", "Basic LR model for iris data")
        mlflow.log_artifact("hello.py")
        print(f"saved to: {mlflow.get_artifact_uri()}")

        mlflow.sklearn.log_model(
            lr, "model"
        )


if __name__ == "__main__":
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }
    params["random_state"] = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    params["max_iter"] = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    train(params)
