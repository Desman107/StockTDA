from StockTDA import config
from typing import List, Iterator, Tuple

date_iter = iter(config.date_range)
prev_date = next(date_iter)  # Initialize the first date
    
for next_date in date_iter:
    print('#'*20)
    print(f'train period:{config.start_date} to {prev_date}')
    print(f'test period:{prev_date} to {next_date}')
    print('#'*20)
    prev_date = next_date

def cartesian_product(list1: List, list2: List) -> Iterator[Tuple]:
    for item1 in list1:
        for item2 in list2:
            yield (item1, item2)

# 示例用法
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

for pair in cartesian_product(list1, list2):
    print(pair)