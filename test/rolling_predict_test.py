from StockTDA import config


date_iter = iter(config.date_range)
prev_date = next(date_iter)  # Initialize the first date
    
for next_date in date_iter:
    print('#'*20)
    print(f'train period:{config.start_date} to {prev_date}')
    print(f'test period:{prev_date} to {next_date}')
    print('#'*20)
    prev_date = next_date