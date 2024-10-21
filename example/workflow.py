from StockTDA.model import TDALSTM, TDALightGBM, TDAXGBoost, TDASVM
from StockTDA.model.BinaryClassification import BinaryClassificationModel
from StockTDA.TDA import StockTDAConstituentCloud, StockTDAReturnSeriesCloud
from StockTDA.TDA.TDAFrame import StockTDAFrame

from StockTDA.evluate.evaluator import StockTDAClassificationEvaluator
from StockTDA.data.data_prepare import prepare_formulaic_factor


from typing import List, Iterator, Tuple

def cartesian_product(
        list1: List[StockTDAFrame], 
        list2: List[BinaryClassificationModel]
    ) -> Iterator[Tuple[StockTDAFrame, BinaryClassificationModel]]:
    for item1 in list1:
        for item2 in list2:
            yield (item1, item2)


prepare_formulaic_factor() # for the first time, place run this function
ClassificationModel1 = TDAXGBoost()
ClassificationModel2 = TDALightGBM()
ClassificationModel3 = TDALSTM()
ClassificationModel4 = TDASVM()

TDAModel1 = StockTDAReturnSeriesCloud()
TDAModel1.all_Features()

TDAModel2 = StockTDAConstituentCloud()
TDAModel2.all_Features()


for pair in cartesian_product(
        [
            TDAModel1,
            TDAModel2,
        ], 
        [
            ClassificationModel1,
            ClassificationModel2,
            ClassificationModel3,
            ClassificationModel4,
        ]
    ):
    Evaluator = StockTDAClassificationEvaluator(pair[0],pair[1])
    Evaluator.evaluate_all_combinations()
