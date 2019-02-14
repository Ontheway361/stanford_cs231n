#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2018/11/26
@author: lujie
Note: Just for pretty print
"""

import pandas as pd
import numpy as np
from prettytable import PrettyTable


class CustomPrettyTable(object):
    def __init__(self):
        pass

    def list_to_table(self, columns_list, index_list, values_list):
        tb = PrettyTable()
        tb.field_names = columns_list
        for index, value in zip(index_list, values_list):
            if isinstance(value, list):
                tb.add_row([index] + value)
            elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
                tb.add_row([index, value])
            else:
                pass
        print(tb)

    def frame_to_table(self, df, index=False):
        # print(df)
        tb = PrettyTable()

        columns_name = list(df.columns)
        if index is True:
            tb.field_names = ['index_name'] + columns_name
        else:
            tb.field_names = columns_name

        values = df.values
        index_name = df.index
        for i, row in enumerate(values):

            if index is True:
                row = list(row)
                tb.add_row([index_name[i]] + row)
            else:
                tb.add_row(row)
        print(tb)

    def dict_to_table(self, input_dict, columns=None):

        tb = PrettyTable()
        tb.field_names = columns
        for key, value in input_dict.items():
            if isinstance(value, list):
                tb.add_row([key] + value)
            else:
                tb.add_row([key, value])
        print(tb)

