
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st

class StHelper:

    def __init__(self,X,y):
        self.X = X
        self.y = y
        # Apply train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def train_xgboost_classifier(self,n_estimators,max_depth,eta,verbosity,objective,booster,tree_method,n_jobs,gamma,min_child_weight,max_delta_step,subsample,colsample_bytree,colsample_bylevel,colsample_bynode,reg_alpha,reg_lambda,scale_pos_weight,base_score,random_state ,missing,num_parallel_tree,importance_type,validate_parameters ):
            xgboost_clf = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,eta=eta,verbosity=verbosity,objective=objective,booster=booster,tree_method=tree_method,n_jobs=n_jobs,gamma=gamma,min_child_weight=min_child_weight,max_delta_step=max_delta_step,subsample=subsample,colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel,colsample_bynode=colsample_bynode,reg_alpha=reg_alpha,reg_lambda=reg_lambda,scale_pos_weight=scale_pos_weight,base_score=base_score,random_state=random_state ,missing=missing,num_parallel_tree=num_parallel_tree,importance_type=importance_type,validate_parameters=validate_parameters
                )

            xgboost_clf.fit(self.X_train, self.y_train)
            y_pred = xgboost_clf.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)

            return xgboost_clf, accuracy



    def draw_main_graph(self,xgboost_clf,ax):

        XX, YY, input_array = self.draw_meshgrid()
        labels = xgboost_clf.predict(input_array)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')



    def draw_meshgrid(self):
        a = np.arange(start=self.X[:, 0].min() - 1, stop=self.X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=self.X[:, 1].min() - 1, stop=self.X[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)

        input_array = np.array([XX.ravel(), YY.ravel()]).T

        return XX, YY, input_array

        labels = xgboost_clf.predict(input_array)