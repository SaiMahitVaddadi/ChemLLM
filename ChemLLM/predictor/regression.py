from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression

from gplearn.genetic import SymbolicRegressor  # For symbolic regression

# Define multiple linear regression model
class MultipleLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Define symbolic regression model
class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(self, population_size=1000, generations=20, stopping_criteria=0.01):
        self.model = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            stopping_criteria=stopping_criteria,
            function_set=('add', 'sub', 'mul', 'div'),
            parsimony_coefficient=0.01,
            random_state=0
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_equation(self):
        return self.model._program
        # Define logistic regression model

class LogisticRegressionModel(BaseEstimator, RegressorMixin):
    def __init__(self, penalty='l2', C=1.0, max_iter=100):
        self.model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)