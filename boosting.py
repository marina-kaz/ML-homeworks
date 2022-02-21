from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: - np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

        self.n_features = None

    # done ?
    def fit_new_base_model(self, x, y, predictions):
        bootstrap_indices = np.random.choice(np.arange(y.shape[0]), size=int(y.shape[0] * self.subsample))
        model = self.base_model_class(**self.base_model_params)

        s = - self.loss_derivative(y[bootstrap_indices], predictions[bootstrap_indices])
        model.fit(x[bootstrap_indices], s)
        new_predictions = model.predict(x)
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)
        self.gammas.append(gamma * self.learning_rate)
        self.models.append(model)

    # done ?
    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        self.n_features = x_train.shape[1]

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions = np.array([i[1] for i in self.predict_proba(x_train)])
            valid_predictions = np.array([i[1] for i in self.predict_proba(x_valid)])

            # self.history['train'].append(self.loss_fn(y_train, [i[1] for i in self.predict_proba(x_train)]))
            self.history['train'].append(self.loss_fn(y_train, train_predictions))
            # self.history['test'].append(self.loss_fn(y_valid, [i[1] for i in self.predict_proba(x_valid)]))
            self.history['test'].append(self.loss_fn(y_valid, valid_predictions))

            if self.early_stopping_rounds is not None:
                if len(self.history['test']) - np.argmin(self.history['test']) - 1 >= self.early_stopping_rounds:
                    print('early stopping')
                    break

        if self.plot:

            fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
            fig.suptitle('Loss history')

            sns.lineplot(ax=axes, y=self.history['train'], x=np.arange(len(self.history['train']))+1, label='train')
            sns.lineplot(ax=axes, y=self.history['test'], x=np.arange(len(self.history['test']))+1, label='test')

            axes.set_ylabel('loss value')
            axes.set_xlabel('number of estimators')
            plt.show()

    # done ?
    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions = predictions + (model.predict(x) * gamma)
        class_one = np.apply_along_axis(self.sigmoid, 0, predictions)
        class_two = 1 - class_one
        return np.vstack((class_two, class_one)).T


    # done
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        weights = np.zeros(self.n_features)
        for model in self.models:
            weights = weights + model.feature_importances_
        weights = weights / self.n_estimators
        return weights / weights.sum()
        pass
