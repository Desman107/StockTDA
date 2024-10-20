from itertools import combinations
from StockTDA.model.LGBM import TDALightGBM
class A():
    def __init__(self):pass
    @property
    def features_combination(self):
        features = ['betti', 'entropy', 'l2_norm']
        # 使用生成器表达式来逐步生成每个组合
        for r in range(1, len(features) + 1):
            for combo in combinations(features, r):
                yield list(combo)

    def test(self):
        for i in self.features_combination : print (i)
B = A()
B.test()
T = TDALightGBM
print(T.__name__)
