from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run

import configparser
import pymysql
import numpy as np
import pandas as pd
import os

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

###################以下是一些基本的工具和数据库操作##########################

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
    db = pymysql.connect(db_host, db_user, db_passwd, db_dataset, charset='utf8')
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
            count += 1
        else:
            sql += ')'
    sql += " VALUES ("
    count = 0
    for i in value_list:
        if type(i).__name__ == 'int':
            sql += str(i)
        else:
            i = str(i)
            if i[-1] == '\n':
                i = i[0:-1]
            if i[0] == '\'' and i[-1] == '\'':
                i = i[1:-1]

            if '\'' in i:
                i = i.replace('\'','\\\'')

            if '\"' in i:
                i = i.replace('\"','\\\"')

            if i[0] != '\'' or i[-1] != '\'':
                sql = sql + '\'' + i + '\''
            else:
                sql += i

        if count != len(value_list)-1:
            sql += ','
            count += 1
        else:
            sql += ')'

    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("\033[0;32;40m\tsql fail\033[0m")
        print(e)
        return "ERROR"
       # Rollback in case there is any error
        db.rollback()

    # 关闭数据库连接
    db.close()
    return "SUCCEED"

def update_db(table, old_column, new_value, condition_column, compare_operator, condition_value):
    cursor, db = create_connection()
    sql = "UPDATE " + table + " SET " + old_column + " = " + new_value + " WHERE " + condition_column + compare_operator + condition_value
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print("\033[0;32;40m\tsql fail\033[0m")
        print(e)
        db.rollback()
        return "ERROR"
    db.close()
    return "SUCCEED"

def add_result(notebook_id, type, content):
    column_list = ["notebook_id","tyoe","content"]
    value_list = [notebook_id, type, content]
    return insert_db("result", column_list, value_list)

def check_model(notebook_id):
    cursor, db = create_connection()
    sql = "SELECT * FROM model WHERE notebook_id=" + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    is_in = False
    for row in sql_res:
        is_in = True
        break
    return is_in

def get_params_code_by_id(notebook_id):
    cursor, db = create_connection()
    sql = "SELECT operator.operator," \
          "operator.rank," \
          "operator.parameter_1_code," \
          "operator.parameter_2_code," \
          "operator.parameter_3_code," \
          "operator.parameter_4_code," \
          "operator.parameter_5_code," \
          "operator.parameter_6_code," \
          "operator.parameter_7_code," \
          "operator.data_object_value " \
          "FROM operator WHERE  operator.notebook_id  = " + notebook_id
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        one_operator_params = {
            "name": row[0],
            "rank": row[1],
            "p1": row[2],
            "p2": row[3],
            "p3": row[4],
            "p4": row[5],
            "p5": row[6],
            "p6": row[7],
            "p7": row[8],
            "data_object_value": row[9],
        }
        result.append(one_operator_params)
    return result

def add_model(notebook_id, model_type):
    global CONFIG

    model_type_list_str = CONFIG.get("models", "model_task")
    model_type_list_str = model_type_list_str[1:-1]
    model_type_list = model_type_list_str.split(',')
    new_model_type_list = []
    for i in model_type_list:
        new_model_type_list.append(int(i[-1]))
    task_type = new_model_type_list[model_type-1]

    is_in = check_model(notebook_id)
    # if is_in == True:
    value_list = [notebook_id, model_type, task_type]
    column_list = ["notebook_id", "model_type", "task_type"]
    return insert_db("model", column_list, value_list)
    # else:
    #     return "ALREADY EXIST"

def add_operator(notebook_id, rank, data_object, data_object_value, operator, physic_operation, parameter_code_dict, parameter_type_dict):
    global CONFIG
    logic_operation = eval(CONFIG.get('operators', 'operations'))[operator]['logic_operations']
    value_list = [
        notebook_id,
        rank,
        logic_operation,
        data_object,
        data_object_value,
        physic_operation,
        operator
    ]
    column_list = [
        "notebook_id",
        "rank",
        "logic_operation",
        "data_object",
        "data_object_value",
        "physic_operation",
        "operator"
    ]
    parameter_keys = eval(CONFIG.get('operators', 'operations'))[operator]['params']
    for i in range(0, len(parameter_keys)): #遍历这个操作下的所有参数名
        if(parameter_keys[i] in parameter_code_dict and parameter_keys[i] in parameter_type_dict): #如果传进来的参数字典包含这个字段
            column_name = "parameter_" + str(i+1)
            column_list.append(column_name + '_code')
            column_list.append(column_name + '_type')
            column_list.append(column_name + '_name')
            value_list.append(parameter_code_dict[parameter_keys[i]])
            value_list.append(parameter_type_dict[parameter_keys[i]])
            value_list.append(parameter_keys[i])

    return insert_db("operator", column_list, value_list)

###################以下是抽取序列需要用的工具##########################

def add_sequence_from_walk_logs(walk_logs, save_path):
    if walk_logs['is_img'] == True:
        update_db("notebook", "cant_sequence", '1', "id", '=', str(walk_logs["notebook_id"]))
        print("\033[0;33;40m\timage dataset pass\033[0m")
        return
    if len(walk_logs['operator_sequence'])==0:
        update_db("notebook", "cant_sequence", '2', "id", '=', str(walk_logs["notebook_id"]))
        print("\033[0;33;40m\tsequence length is 0\033[0m")
        return
    if len(walk_logs['models'])==0:
        update_db("notebook", "cant_sequence", '3', "id", '=', str(walk_logs["notebook_id"]))
        print("\033[0;33;40m\tmodel number is 0\033[0m")
        return

    notebook_id = walk_logs['notebook_id']
    for model in walk_logs['models']:
        res1 = add_model(notebook_id, model)
    print('model:', res1)
    if res1 == 'ERROR':
        return res1
    count = 1
    for operator_node in walk_logs['operator_sequence']:
        res2 = add_operator(notebook_id, count, operator_node["data_object"],operator_node["data_object_value"], operator_node["operator_name"], operator_node['physic_operation'], operator_node['parameter_code'], operator_node['parameter_type'])
        count += 1
        if res2 == 'ERROR':
            return res2

    np.save(save_path + '/' + walk_logs['notebook_title'] + '.npy', walk_logs)
    res3 = update_db("notebook", "add_sequence", '1', 'id', "=", str(walk_logs["notebook_id"]))
    if res3 == 'ERROR':
        return res2

    print("\033[0;32;40m\tsucceed\033[0m")
    return "SUCCEED"

def get_batch_notebook_info():
    cursor, db = create_connection()
    sql = "SELECT distinct title, id, scriptUrl FROM notebook WHERE add_sequence=0 and cant_sequence=0 and isdownload=1"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        notebook_info = (row[1],row[0],row[2])
        result.append(notebook_info)
    return result

###################以下是批量运行##########################

def found_dataset(old_path, notebook_id, root_path, origin_code):
    filename = old_path.split('/')[-1]
    print('filename', filename)
    if '.' not in filename:
        result = root_path
    else:
        result = root_path + '/' + filename
    print("result", result)
    return origin_code.replace(old_path, result)

###################以下是统计##########################
def check_object_column(data_object_value, data_object,col_list):
        object_name = data_object
        if data_object == 'unknown':
            if 'test' in data_object_value or 'Test' in data_object_value:
                object_name = 'test'
            elif 'train' in data_object_value or 'Train' in data_object_value:
                object_name = 'train'
            else:
                object_name = 'data'

        column_list = []
        for i in col_list:
            if '\''+i + '\'' in data_object_value or '\"'+i + '\"' in data_object_value or '.'+i in data_object_value:
                column_list.append(i)

        if column_list == []:
            column_list = 'all'

        return column_list, object_name

def get_all_column_list(datasetid,data_root_path):
    cursor, db = create_connection()
    sql = "SELECT dataSourceUrl FROM datasources where sourceId=" + str(datasetid)
    cursor.execute(sql)

    res = cursor.fetchall()
    for row in res:
        data_path = row[0].split('/')[-1]

    filelist = os.listdir(data_root_path + '/' + data_path + '.zip')
    path = data_root_path + '/' + data_path + '.zip/' + filelist[0]
    dataframe = pd.read_csv(path)
    column_list = dataframe.columns.values

    return column_list

def get_column_seq(notebook_id, col_list):
    # filelist = os.listdir(dataset_root_path + '/' + data_path)
    # dataframe = pd.read_csv(dataset_root_path + '/' + data_path + '/' + filelist[0])
    # column_list = dataframe.columns.values
    seq = {}
    seq['test'] = {}
    seq['data'] = {}
    seq['train'] = {}
    # print("col_list:", col_list)

    cursor, db = create_connection()
    sql = "SELECT * FROM operator where notebook_id=" + str(notebook_id)
    cursor.execute(sql)
    operations = cursor.fetchall()

    print("\033[0;31;40m" + str(notebook_id) + "\033[0m")
    print('len(operations):', len(operations))
    for row in operations:
        print(row[0:8],row[-1])
        column_list,data_object = check_object_column(row[-1],row[4],col_list)
        if(row[4] == 'unknown'):
            continue
        if column_list == 'all':
            column_list = col_list
        for column in column_list:
            if column not in seq[data_object].keys():
                seq[data_object][column] = []
            seq[data_object][column].append((row[3], row[-1]))

    return seq



def stastic_operator_sequence_by_dataset(datasetid,data_root_path):
    cursor, db = create_connection()
    sql = "SELECT notebook.id, notebook.scriptUrl FROM notebook , datasources where notebook.id=datasources.notebook_id and datasources.sourceId=" + str(datasetid) + " and (notebook.add_sequence=1 and notebook.cant_sequence=0);"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    logic_seq = []
    url_list = []
    result = []
    log_dic = []
    ope_dic = []
    log_count = []
    ope_count = []
    seq_list = []
    column_name = ''

    col_list = get_all_column_list(datasetid,data_root_path)
    print(col_list)
    for row in sql_res:
        url = row[1].split('/')[-1]
        if url in url_list:
            continue
        url_list.append(url)
        seq_list.append(get_column_seq(row[0],col_list))

    for name in seq_list[0]['train'].keys():
        column_name = name
        break
    print(column_name)
    for i in seq_list:
        if column_name in i['train']:
            print(i['train'][column_name])
        else:
            print(i['train'].keys())
            print("no column?")
            print([])
        # sql1 = "SELECT logic_operation,operator,data_object_value,data_object from operator where notebook_id="+ row[0]
        # cursor.execute(sql1)
        # sql1_result = cursor.fetchall()
        # temp_log_seq = []
        # temp_ope_seq = []
        # temp_object = []
        # temp_ope_object = []
        # for row1 in sql1_result:
        #     temp_log_seq.append(row1[0])
        #     temp_ope_seq.append(row1[1])
        #     temp_ope_object.append(row1[2])
        #     temp_object.append(row1[3])
        # print("logic:", temp_log_seq)
        # print("operator:", temp_ope_seq)
        # print("obect:", temp_ope_object)
        # print("type obect:", temp_object)

        # is_in_log = False
        # is_in_ope = False
        # for index,seq in enumerate(log_dic):
        #     if seq == temp_log_seq:
        #         is_in_log = True
        #         log_count[index] += 1
        #         # log_count.append(log_count[index])
        #     # else:
        #     #     log_count.append(0)
        # if is_in_log == False:
        #     temp_log_object.append(temp_log_object)
        #     log_dic.append(temp_log_seq)
        #     log_count.append(0)
        #
        #
        # for index,seq in enumerate(ope_dic):
        #     if seq == temp_ope_seq:
        #         is_in_ope = True
        #         ope_count[index] += 1
        # if is_in_ope == False:
        #     ope_dic.append(temp_ope_seq)
        #     ope_count.append(0)

    # for index in range(0, len(log_dic)):
    #     print("logic:", log_dic[index])
    #     print("count:", log_count[index])

    # for index in range(0, len(ope_dic)):
    #     print("operator:", ope_dic[index])
    #     print("count:", ope_count[index])


if __name__ == '__main__':
    # CONFIG = configparser.ConfigParser()
    # CONFIG.read('config.ini')
    # model_dic = eval(CONFIG.get('operators', 'operations'))
    # print(type(model_dic))
    stastic_operator_sequence_by_dataset(5407,'../spider/unzip_dataset')