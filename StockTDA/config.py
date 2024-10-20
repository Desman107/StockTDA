# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       : 2024/10/20 17:02
# Author     : DaZhi Huang
# email      : 2548538192@qq.com
# Description: config file
"""
import os

index_code = '000300'
start_date = '2016-01-01'
end_date = '2024-08-29'
date_range = []
for year in range(2018, 2025):
    for month in range(1, 13):
        date_range.append(f'{year}-{str(month).zfill(2)}-01')
date_range = date_range[:-7]



project_root = ''
data_path = ''


if project_root == '':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if data_path == '':
    data_path = os.path.join(project_root, 'data')


temp_save_path = os.path.join(data_path, 'temp','temp_factor_data')
factor_data_save_path = os.path.join(data_path,'temp',f'csi{index_code}_factor_data')
mlflow_path = os.path.join(data_path,'mlruns')
temp_file_path = os.path.join(data_path,'temp','temp_file')

if not os.path.exists(temp_save_path) : os.makedirs(temp_save_path)
if not os.path.exists(factor_data_save_path) : os.makedirs(factor_data_save_path)
if not os.path.exists(mlflow_path) : os.makedirs(mlflow_path)
if not os.path.exists(temp_file_path) : os.makedirs(temp_file_path)
