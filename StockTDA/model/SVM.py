"""
# File       : XGB.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: Stock topology data analysis combines with SVM
"""


from StockTDA.model.BinaryClassification import BinaryClassificationModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class TDASVM(BinaryClassificationModel):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        

    def run_classification(self, X_train, y_train, X_test, y_test):
        model = Pipeline([
            ('scaler', StandardScaler()),  
            ('svm', SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True))
        ])
        
        model.fit(X_train, y_train.ravel())  

        y_pred = model.predict_proba(X_test)[:, 1]  
        return y_pred
