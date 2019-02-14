"""
Created on 2018/6/27
@author: Jerry
"""

import pandas as pd
import numpy as np
import time


def translate_date_num2str(date_num):
    """
    translate date_num which is from sql into to date str
    :param date_num: origin date num
    :return: date str: "2017-5-7"
    """
    return time.strftime("%Y-%m-%d", time.localtime(int(float(date_num) / 1000)))

def translate_date_num2str_v1(date_num):
    '''
    input should be microseconds
    '''
    return time.strftime("%Y%m%d%H%M%S", time.localtime(int(float(date_num) / 1000)))

def translate_date_num2str_v2(date_num):
    '''
    input should be microseconds
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(float(date_num) / 1000)))


def translate_date_str2num(date_str):
    """
    translate date_num which is from sql into to date str
    :param date_num: origin date num
    :return: date str: "2017-5-7"
    """
    return time.mktime(time.strptime(date_str, "%Y-%m-%d")) * 1000

def translate_date_str2num_v1(date_str):
    '''
    return the microseconds of corresponding date
    '''
    return time.mktime(time.strptime(date_str, "%Y%m%d%H%M%S")) * 1000

def translate_date_str2num_v2(date_str):
    '''
    return the microseconds of corresponding date
    '''
    return time.mktime(time.strptime(date_str, "%Y-%m-%d %H:%M:%S")) * 1000



def cal_stamp_diff(last_stamp, cur_stamp):
    '''
    calculate the stamp diff between last_stamp and cur_stamp
    '''
    last_stamp = translate_date_str2num_v2(last_stamp)
    cur_stamp  = translate_date_str2num_v1(cur_stamp)
    return abs(cur_stamp - last_stamp)

def gen_time_num(date_str):
    """
    use date str to generate date time
    :param str: date str
    :return: time num
    """
    repayment_month = [int(i) for i in date_str.split('-')]

    if len(repayment_month) == 2:
        repayment_month.append(25)

    repayment_month_str = "-".join([str(i) for i in repayment_month])
    repayment_month_num = translate_data_str2num(repayment_month_str)

    return repayment_month, repayment_month_num

def gen_month_length(create_time):
    create_data_list = [int(i) for i in translate_date_num2str(create_time).split('-')]
    now = time.time() * 1000
    now_date_list = [int(i) for i in translate_date_num2str(now).split('-')]
    month_length = (
            (now_date_list[0] - create_data_list[0]) * 12 + now_date_list[1] - create_data_list[1])
    return month_length

if __name__ == '__main__':
    print('hello world')
