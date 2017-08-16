import os
import sys
import random
import shutil
import numpy as np
import xlwt


# install anaconda3
# install pycharm 设置pycharm使用anaconda3的python

# 写excel
def write_data_to_excel():
    # 实例化一个Workbook()对象(即excel文件)
    wbk = xlwt.Workbook()
    # 新建一个名为Sheet1的excel sheet。此处的cell_overwrite_ok =True是为了能对同一个单元格重复操作。
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)


    # xishu 整数 浮点数都可以
    xishu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24]

    # 循环365天 从0到364
    for i in range(365):
        # 循环24小时 从0到23
        for j in range(24):
            # 将每一行的每个元素按行号i,列号j,写入到excel中。
            # 天数为i 小时为j i*24+j就是行数
            # 根据需要设置 行数 和 列数
            # xishu[j]
            sheet.write(i * 24 + j, 0, xishu[j]*2)
    # 以传递的name+当前日期作为excel名称保存。
    wbk.save('test1' + '.xls')


if __name__ == "__main__":
    print('main')
    write_data_to_excel();