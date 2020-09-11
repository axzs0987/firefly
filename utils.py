from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run

import configparser
import MySQLdb
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

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

def create_connection():
    global CONFIG
    db_host = CONFIG.get('database', 'host')
    db_user = CONFIG.get('database', 'user')
    db_passwd = CONFIG.get('database', 'passwd')
    db_dataset = CONFIG.get('database', 'dataset')
    db = MySQLdb.connect(db_host, db_user, db_passwd, db_dataset, charset='utf8')
    cursor = db.cursor()
    return cursor,db

def insert_db(table, column_list, value_list):
    cursor,db = create_connection()
    sql = "INSERT INTO "
    sql += table
    sql += " ("
    count = 0
    for i in column_list:
        sql += i
        if count != len(column_list) - 1:
            sql += ','
        else:
            sql += ')'
    sql += " VALUES("
    count = 0
    for i in value_list:
        sql += i
        if count != len(value_list)-1:
            sql += ','
        else:
            sql += ')'
    try:
       cursor.execute(sql)
       db.commit()
    except:
       # Rollback in case there is any error
       db.rollback()

    # 关闭数据库连接
    db.close()

def update_db(table, old_column, new_value, condition_column, compare_operator, condition_value):
    cursor, db = create_connection()
    sql = "UPDATE " + table + " SET " + old_column + " = " + new_value + " WHERE " + condition_column + compare_operator + condition_value
    try:
       cursor.execute(sql)
       db.commit()
    except:
       db.rollback()

    db.close()

def add_result(notebook_id, type, content):
    column_list = ["notebook_id","tyoe","content"]
    value_list = [notebook_id, type, content]
    insert_db("result", column_list, value_list)

def add_model(notebook_id, model_type):
    global CONFIG
    model_type_list_str = CONFIG.get("models", "model_task")
    model_type_list_str = model_type_list_str[1:-1]
    model_type_list = model_type_list_str.split(',')
    task_type = model_type_list[model_type-1]
    value_list = [notebook_id, model_type, task_type]
    column_list = ["notebook_id", "model_type", "task_type"]
    insert_db("result", column_list, value_list)

def add_operator(notebook_id, rank, data_object, operator, physic_operation, parameter_code_dict, parameter_type_dict):
    global CONFIG
    logic_operation = eval(CONFIG.get('operators', 'operations'))[operator]['logic_operations']
    value_list = [
        notebook_id,
        rank,
        logic_operation,
        data_object,
        physic_operation,
        operator
    ]
    column_list = [
        "notebook_id",
        "rank",
        "logic_operation",
        "data_object",
        "physic_operation",
        "operator"
    ]
    parameter_keys = eval(CONFIG.get('operators', 'operations'))[operator]['params']
    for i in range(0, len(parameter_keys)): #遍历这个操作下的所有参数名
        if(parameter_keys[i] in parameter_code_dict and parameter_keys[i] in parameter_type_dict): #如果传进来的参数字典包含这个字段
            column_name = "parameter_" + str(i)
            column_list.append(column_name + '_code')
            column_list.append(column_name + '_type')
            column_list.append(column_name + '_name')
            value_list.append(parameter_code_dict[parameter_keys[i]])
            value_list.append(parameter_type_dict[parameter_keys[i]])
            value_list.append(parameter_keys[i])

    insert_db("result", column_list, value_list)

def add_sequence_from_walk_logs(walk_logs, save_path):
    if walk_logs['is_img'] == False:
        return "image dataset pass"
    if len(walk_logs['operator_sequence'])==0:
        return "sequence length is 0"

    notebook_id = walk_logs['notebook_id']
    for model in walk_logs['models']:
        add_model(notebook_id, model)
    count = 1
    for operator_node in walk_logs['operator_sequence']:
        add_operator(notebook_id, count, operator_node["data_object"], operator_node["operator_name"], operator_node['physic_operation'], operator_node['parameter_code'], operator_node['parameter_type'])

    np.save(walk_logs, save_path + '/' + walk_logs['notebook_title'] + '.ipynb')
    update_db("notebook", "add_sequence", '1', 'id', "=", str(walk_logs["notebook_id"]))
    return "suceed!"

def get_batch_notebook_info():
    cursor, db = create_connection()
    sql = "SELECT id,title FROM notebook"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        notebook_info = (row[0],row[1])
        result.append(notebook_info)
    return result

if __name__ == '__main__':
    CONFIG = configparser.ConfigParser()
    CONFIG.read('config.ini')
    model_dic = eval(CONFIG.get('operators', 'operations'))
    print(type(model_dic))