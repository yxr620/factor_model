import pandas as pd
import numpy as np

from sqdata import d
from tqdm import tqdm
from multiprocessing import Pool

def covert_name(stock_code:str) -> str:
    number = stock_code.split('.')[0]
    exchange = stock_code.split('.')[1]
    if exchange == "XSHG":
        exchange = "SH"
    elif exchange == "XSHE":
        exchange = "SZ"
    elif exchange == "XSHE2":
        exchange = "SZ2"
    else:
        print("Error: exchange name is wrong!")
        exit()
    return number + "." + exchange

def process_datapoint(args):
    date_group, date_key, i = args
    print(f"i {i} start {date_key[i - 4]} to {date_key[i + 2]}")

    # generate needed group
    group_list = []
    for j in range(i - 4, i + 3):
        group_list.append(date_group.get_group(date_key[j]))
    # judge T-4 to T+0 & T+1 to T+2
    stock_set = set(group_list[0]['stock_code'])
    for table in group_list:
        stock_set = stock_set & set(table['stock_code'])
    # judge sqdata exist these stocks
    stock_list = list(stock_set)
    data = d.wsd(stock_list, "open", date_key[i - 4], date_key[i + 2])
    # data is a dict of stock_name -> pd.series. If value is none means sqdata don't have this stock
    sq_set = set(data.keys())
    for stock_name in data.keys():
        if data[stock_name].isnull().values.any(): # judge if None is in this series
            sq_set.remove(stock_name)
    stock_set = set(stock_list) & sq_set

    # generate data point for each stock
    stock_list = list(stock_set)
    factor_list = []
    label_list = []
    for stock_name in tqdm(stock_list):
        # generate label T+1 to T+2
        T1 = data[stock_name][-2]
        T2 = data[stock_name][-1]
        label_list.append(T2 / T1 - 1)
        # 5 * 84
        stock_factor = []
        for j in range(5): # get rid of the stock_code & date
            date_j_table = group_list[j].set_index('stock_code')
            stock_factor.append(date_j_table.loc[stock_name].tolist()[1:])
        # 5 * 84 -> 84 * 5 -> 420
        stock_factor = np.array(stock_factor).T.reshape(-1)
        factor_list.append(stock_factor)
    # 420 * stock_num
    factor_list = np.array(factor_list)
    label_list = np.array(label_list)
    stock_name_arr = np.array(stock_list)
    # concatentate data
    total_table = np.concatenate([stock_name_arr.reshape(-1, 1), label_list.reshape(-1, 1), factor_list], axis=1)
    print(total_table.shape)
    np.savetxt(f"./data/daypoint/{date_key[i]}.csv", total_table, delimiter=",", fmt="%s")


def generate_data(date_group, date_key):
    args_list = [(date_group, date_key, i) for i in range(4, len(date_key) - 2)]
    with Pool(processes=4) as pool:
        pool.map(process_datapoint, args_list)


if __name__ == "__main__":
    # Read the data
    df = pd.read_feather("./data/factor.feather")

    # covert uquant name to normal name
    df['stock_code'] = [covert_name(stock_code) for stock_code in df['stock_code']]
    group = df.groupby('stock_code')

    date_group = df.groupby('date')
    date_key = list(date_group.groups.keys())
    date_key.sort()

    generate_data(date_group, date_key[:100])