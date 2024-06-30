import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
import sklearn.metrics
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class ForecastingTemplate(ABC):
    def __init__(self, data_path) -> None:
        self.data = pd.read_excel(data_path, index_col=0).sort_index()

        if self.data.isnull().sum().sum() > 0:
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')

        self.original_training_set = self.data.iloc[:-100].copy()
        self.original_test_set = self.data.iloc[-100:].copy()

        self.scaler = None
        self.reset_sets()

    def reset_sets(self):
        """Reset the training and test sets to their original state."""
        self.training_set = self.original_training_set.copy()
        self.test_set = self.original_test_set.copy()

    def calculate_r_squared(self, y_true, y_pred) -> float:
        return sklearn.metrics.r2_score(y_true, y_pred)

    def scale_data(self) -> None:
        """Scales the training and test data sets excluding the 'close' column."""
        self.scaler = StandardScaler()
        training_set = self.training_set.drop(columns=["close"])
        self.training_set_scaled = self.scaler.fit_transform(training_set)
        self.test_set_scaled = self.scaler.transform(self.test_set.drop(columns=["close"]))

    def add_rolling_average(self, window: int) -> None:
        """Add a rolling average feature to the dataset."""
        self.training_set.loc[:, f"close_{window}_day_avg"] = self.training_set["close"].shift(1).rolling(window=window).mean()
        self.test_set.loc[:, f"close_{window}_day_avg"] = self.test_set["close"].shift(1).rolling(window=window).mean()


        self.training_set = self.training_set.dropna()
        self.test_set = self.test_set.dropna()

    @abstractmethod
    def fit(self, scale=False) -> None:
        pass

    @abstractmethod
    def predict(self, set_type='test') -> np.ndarray:
        pass

class PredictPriceThroughNminus1(ForecastingTemplate):
    def fit(self, scale=False) -> None:
        pass

    def predict(self, set_type='test') -> np.ndarray:
        if set_type == 'test':
            prediction = self.data["close"].shift(1)[-100:]
        elif set_type == 'training':
            prediction = self.data["close"].shift(1)[:-100]
        return prediction.dropna().values

class Regressor(ForecastingTemplate):
    def fit(self, scale=False) -> None:
        if scale:
            self.scale_data()
            X_train = self.training_set_scaled
        else:
            X_train = self.training_set.drop(columns=["close"])

        self.model.fit(X_train, self.training_set["close"])

    def predict(self, set_type='test') -> np.ndarray:
        if self.scaler:
            if set_type == 'test':
                X_test = self.test_set_scaled
            else:
                X_test = self.training_set_scaled
        else:
            if set_type == 'test':
                X_test = self.test_set.drop(columns=["close"])
            else:
                X_test = self.training_set.drop(columns=["close"])

        prediction = self.model.predict(X_test)
        return prediction

    def tune_hyperparameters(self, param_grid) -> None:
        """Tune hyperparameters including the rolling average window."""
        best_score = -np.inf
        best_params = None

        if 'rolling_avg_window' in param_grid:
            rolling_avg_windows = param_grid.pop('rolling_avg_window')

            for window in rolling_avg_windows:
                self.reset_sets()  # Reset the training and test sets before adding rolling average
                self.add_rolling_average(window)
                X_train = self.training_set.drop(columns=["close"])
                y_train = self.training_set["close"]

                # Perform grid search with cross-validation
                grid_search = GridSearchCV(self.model, param_grid, scoring='r2', cv=5)
                grid_search.fit(X_train, y_train)

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                    best_params['rolling_avg_window'] = window

            # Reset the rolling avg window in the training set with the best window found
            self.reset_sets()
            self.add_rolling_average(best_params['rolling_avg_window'])
            X_train = self.training_set.drop(columns=["close"])
            y_train = self.training_set["close"]
        else:
            X_train = self.training_set.drop(columns=["close"])
            y_train = self.training_set["close"]

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(self.model, param_grid, scoring='r2', cv=5)
            grid_search.fit(X_train, y_train)

            best_score = grid_search.best_score_
            best_params = grid_search.best_params_

        self.model = grid_search.best_estimator_
        self.model.fit(X_train, y_train)
        print(f"Best hyperparameters for {self.model.__class__.__name__}: {best_params}")

class RidgeRegressor(Regressor):
    def __init__(self, data_path, alpha=0.1, max_iter=1000) -> None:
        super().__init__(data_path)
        self.model = Ridge(alpha=alpha, max_iter=max_iter)

class LassoRegressor(Regressor):
    def __init__(self, data_path, alpha=0.1, max_iter=10000) -> None:
        super().__init__(data_path)
        self.model = Lasso(alpha=alpha, max_iter=max_iter)
