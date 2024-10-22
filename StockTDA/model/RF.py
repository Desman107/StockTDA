


from StockTDA.model.BinaryClassification import BinaryClassificationModel

from sklearn.ensemble import RandomForestClassifier


class TDARandomForest(BinaryClassificationModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
        

    def run_classification(self, X_train, y_train, X_test, y_test):
        y_train = y_train.values
        y_test = y_test.values
        model = RandomForestClassifier(**self.params)

        # Train the RandomForest model
        model.fit(X_train, y_train.ravel())
        
        # Predict probabilities for the test set
        y_pred = model.predict_proba(X_test)[:, 1]
        return y_pred
