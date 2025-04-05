from functools import reduce
import pymysql
import os
import pandas as pd
import re
import numpy as np
from constants import quarter_ranges

# def get_end_quarter(date: str):
#     start_quarter = re.findall(pattern, date)[0][1]
#     for quarter in quarter_ranges:
#         if quarter[0] == start_quarter:
#             return year + quarter[1]
#     return None

# date, total profit, total spending, total revenue
columns = ['date', 'income_after_depreciation_and_amortization', 'cost_of_goods', 'net_income']

ouput_dir = 'stocks3/'
def get_year(date: str):
    pattern =  "(\\d+-)(\\d+-\\d+)"
    year = re.findall(pattern, date)[0][0]
    return year[:-1]


def get_companies():
    return list(map(lambda x: x.removesuffix('.csv'), filter(lambda x: x.endswith('.csv'), os.listdir('stocks2'))))

def get_first_and_last_date(company: str):
    dates =  pd.read_csv('stocks2/' + company + '.csv')['Date'].to_numpy()
    return np.array([dates[0], dates[-1]])
def collect_data(row, year: int, quarter_range: str):
    date = str(year) + '-' + quarter_range
    if (row == None):
        return {'date': date, 'total_profit': None, 'total_spending': None, 'total_revenue': None}
    # date, total profit, total spending, total revenue
    # columns = ['date', 'income_after_depreciation_and_amortization', 'cost_of_goods', 'net_income']
    return {'date': date, 'total_profit': int(row['income_after_depreciation_and_amortization']) // 4, 'total_spending': int(row['cost_of_goods']) // 4, 'total_revenue': int(row['net_income']) // 4}

companies = get_companies()
connection = pymysql.connect(host='localhost',
                             port=3306,
                             user='root',
                             cursorclass=pymysql.cursors.DictCursor)


with connection.cursor() as cursor:
    cursor.execute('USE earnings;')
    value = reduce(lambda x, y: x + ', ' +  y, columns, '')[2:]
    for company in companies:
        dates = get_first_and_last_date(company)
        start_year = int(get_year(dates[0]))
        end_year = int(get_year(dates[1]))
        data = []
        for year in range(start_year, end_year + 1):
            query = f'SELECT {value} FROM income_statement WHERE \'{year}-01-01\' < date AND date < \'{year}-12-31\' AND act_symbol=\'{company}\' AND period=\'Year\''
            cursor.execute(query)
            rows = cursor.fetchone(); 
            if rows == None or len(rows) == 0:
                for ran in quarter_ranges:
                    data.append(collect_data(None, year, ran[0]))
                print('missing values in year', year)
                continue

            for ran in quarter_ranges:
                tmp = collect_data(rows, year, ran[0])
                data.append(tmp)

            pd.DataFrame(data=data).to_csv(ouput_dir + company + '.csv')

