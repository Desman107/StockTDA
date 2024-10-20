# -*- coding: utf-8 -*-
"""
---------------------------------------------
Created on 2023/1/29 17:38
@author: ZhangYundi
@email: yundi.xxii@outlook.com
---------------------------------------------
"""
import concurrent.futures.process
import concurrent.futures

from joblib import Parallel, delayed
from tqdm.auto import tqdm

"""
并发池：
"""

MaxGo = 8


def go(maxGo=MaxGo):
    return concurrent.futures.process.ProcessPoolExecutor(maxGo)


class pool:

    def __init__(self, n_jobs=MaxGo, tqdm_on: bool = True, verbose=0):
        """
        :param n_jobs: 并发数
        :param tqdm_on: 是否启用tqdm
        """
        self._tasks = list()
        self.task_num = 0
        self._n_jobs = n_jobs
        self._tqdm_on = tqdm_on

    def submit(self, fn, *args, **kwargs):
        task = delayed(fn)(*args, **kwargs)
        self._tasks.append(task)
        self.task_num += 1

    def do(self, description: str = None):
        if description:
            description = f"Processing: {description}"
        if self._tqdm_on:
            with Parallel(n_jobs=self._n_jobs, verbose=0, return_as='generator') as p:
                pbar = tqdm(total=len(self._tasks))
                pbar.set_description_str(description)
                result = []
                for i in p(self._tasks):
                    result.append(i)
                    pbar.update(1)
                pbar.close()
                return result

    def get_tasks(self):
        p = Parallel(n_jobs=self._n_jobs, verbose=0, return_as='generator')
        return p(self._tasks)

    def map(self, fn, iterable):
        for arg in iterable:
            self.submit(fn, arg)
        return self.do()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class ThreadPool:

    def __init__(self, n_jobs=MaxGo, tqdm_on: bool = True):
        """
        :param n_jobs: 并发线程数
        :param tqdm_on: 是否启用tqdm进度条
        """
        self._tasks = []         # 存储提交的任务
        self._n_jobs = n_jobs    # 最大并发线程数
        self._tqdm_on = tqdm_on  # 是否启用tqdm进度条
        self.task_num = 0        # 已提交的任务数

    def submit(self, fn, *args, **kwargs):
        """
        提交任务到线程池
        :param fn: 要执行的函数
        :param args: 函数的参数
        :param kwargs: 函数的关键字参数
        """
        task = (fn, args, kwargs)  # 封装任务（函数，参数，关键字参数）
        self._tasks.append(task)
        self.task_num += 1

    def _execute_task(self, task):
        """
        内部方法：执行单个任务
        :param task: 任务元组（fn, args, kwargs）
        :return: 任务执行的结果
        """
        fn, args, kwargs = task
        return fn(*args, **kwargs)

    def do(self, description: str = None):
        """
        并发执行所有提交的任务
        :param description: 任务进度条描述
        :return: 所有任务的执行结果列表
        """
        results = []
        if description:
            description = f"Processing: {description}"
        # 使用ThreadPoolExecutor来并发执行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_jobs) as executor:
            futures = [executor.submit(self._execute_task, task) for task in self._tasks]
            
            # 如果启用tqdm进度条，显示任务进度
            if self._tqdm_on:
                with tqdm(total=len(futures)) as pbar:
                    if description:
                        pbar.set_description(description)
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                        pbar.update(1)
            else:
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
        return results

    def map(self, fn, iterable):
        """
        对iterable中的每个元素调用fn，并并发执行
        :param fn: 要执行的函数
        :param iterable: 可迭代对象，作为fn的参数
        :return: 所有任务的执行结果列表
        """
        for item in iterable:
            self.submit(fn, item)
        return self.do()

    def __enter__(self):
        """
        实现上下文管理协议，用于with语句
        :return: self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文时自动清理资源
        """
        pass