# -*-coding:utf-8 -*-

"""
# File       : __init__.py
# Time       : 2024/10/21 13:01
# Author     : DaZhi Huang
# Email      : 2548538192@qq.com
# Description: classification model definition
"""
from .XGB import TDAXGBoost
from .LGBM import TDALightGBM
from .LSTM import TDALSTM
from .SVM import TDASVM
from .RF import TDARandomForest