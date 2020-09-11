from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run

import configparser
import MySQLdb

def get_code_txt(path):
    cot_txt = ""
    lis = read_ipynb(path)
    # LR matching
    lis = Feeding(lis)
    lis = LR_run(lis)
    for paragraph in lis:
        # print(item)
        for item in paragraph['code']:
            if item:
                if (item[0] == '!') or (item[0] == '<' or (item[0] == '%')):
                    continue
            temp = item + '\n'
            cot_txt += temp
    return cot_txt

def insert_operator():
    db = MySQLdb.connect("localhost", "testuser", "test123", "TESTDB", charset='utf8')

if __name__ == '__main__':
    CONFIG = configparser.ConfigParser()
    CONFIG.read('config.ini')
    model_dic = eval(CONFIG.get('operators', 'operations'))
    print(type(model_dic))