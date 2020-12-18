## -*- coding: utf-8 -*-

from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run

import configparser
import pymysql
import numpy as np
import pandas as pd
import os
import socket

import ast
import sys

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

###################以下是一些基本的工具和数据库操作##########################
def get_host_ip():
    """
    查询本机ip地址
    :return:
    """
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]
    finally:
        s.close()

    if ip == '172.26.73.203':
        ip = '39.99.150.216'
    return ip

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
            if len(i) > 0:
                if i[-1] == '\n':
                    i = i[0:-1]
            if len(i) > 0:
                if i[0] == '\'' and i[-1] == '\'':
                    i = i[1:-1]
            if '\'' in i:
                i = i.replace('\'','\\\'')
            if '\"' in i:
                i = i.replace('\"','\\\"')

            if len(i) > 0:
                if i[0] != '\'' or i[-1] != '\'':
                    # i = i.decode("gbk").encode("utf-8")
                    sql = sql + '\'' + i + '\''
                else:
                    # i = i.decode("gbk").encode("utf-8")
                    sql += i

            if len(i) == 0:
                # i =  i.decode("gbk").encode("utf-8")
                sql = sql + '\'' + i + '\''



        if count != len(value_list)-1:
            sql += ','
            count += 1
        else:
            sql += ')'

    print(sql)

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
    if type(new_value).__name__ == 'str':
        new_value = new_value.replace("'", "\\'")
        new_value = new_value.replace('"', '\\"')
        new_value = '\'' + new_value + '\''
    if type(condition_value).__name__ == 'str':
        condition_value = condition_value.replace("'", "\\'")
        condition_value = condition_value.replace('"', '\\"')

    sql = "UPDATE " + table + " SET " + old_column + " = " + str(new_value) + " WHERE " + condition_column + compare_operator + str(condition_value)
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        print("\033[0;32;40m\tsql fail\033[0m")
        print(e)
        db.rollback()
        return "ERROR"
    db.close()
    return "SUCCEED"

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
          "FROM operator WHERE  operator.notebook_id  = " + str(notebook_id)
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

def update_params_value(notebook_id, rank, param_num, content):
    if type(content).__name__ == 'str' or 'int' in type(content).__name__  or 'float' in type(content).__name__:
        cursor, db = create_connection()
        content = str(content).replace("'", "\\'")
        content = content.replace('"', '\\"')
        sql = "UPDATE operator SET parameter_"+ str(param_num)+'_value' + " = '" + content + "' WHERE notebook_id=" + str(notebook_id) + " and rank=" + str(rank)
        print(sql)
        try:
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            print("\033[0;32;40m\tsql fail\033[0m")
            print(e)
            db.rollback()
            return "ERROR"
    else:
        cursor, db = create_connection()
        sql = "UPDATE operator SET parameter_" + str(param_num) + '_value' + " = '" + type(content).__name__ + "' WHERE notebook_id=" + str(notebook_id) + " and rank=" + str(rank)
        print(sql)
        try:
            cursor.execute(sql)
            db.commit()
        except Exception as e:
            print("\033[0;32;40m\tsql fail\033[0m")
            print(e)
            db.rollback()
            return "ERROR"



def add_result(notebook_id, score_type, content, code, metric_type):
    global CONFIG
    # print('zdfsd:',type(content))
    if 'int' in type(content).__name__ or 'float' in type(content).__name__:
        value_list = [notebook_id, score_type, content, code, metric_type]
        column_list = ["notebook_id", "model_type", "content", "code", "metric_type"]
        print("\033[0;33;43madd result\033[0m")
        return insert_db("result", column_list, value_list)
    else:
        value_list = [notebook_id, score_type, str(content), code, metric_type]
        column_list = ["notebook_id", "model_type", "str_content", "code", "metric_type"]
        print("\033[0;33;43madd result\033[0m")
        return insert_db("result", column_list, value_list)
    # else:
    #     return "ALREADY EXIST"

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

def add_operator_show(notebook_id, rank, data_object, data_object_value, operator, physic_operation, parameter_code_dict, parameter_type_dict):
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
            if i+1 > 7:
                continue
            column_name = "parameter_" + str(i+1)
            column_list.append(column_name + '_code')
            column_list.append(column_name + '_type')
            column_list.append(column_name + '_name')
            value_list.append(parameter_code_dict[parameter_keys[i]])
            value_list.append(parameter_type_dict[parameter_keys[i]])
            value_list.append(parameter_keys[i])

    return insert_db("show_operator", column_list, value_list)
###################以下是抽取序列需要用的工具##########################
def add_sequence_from_walk_logs_show(walk_logs, save_path):
    if walk_logs['is_img'] == True:
        print("\033[0;33;40m\timage dataset pass\033[0m")
        return


    notebook_id = walk_logs['notebook_id']
    count = 1

    for operator_node in walk_logs['operator_sequence']:
        res2 = add_operator_show(notebook_id, count, operator_node["data_object"],operator_node["data_object_value"], operator_node["operator_name"], operator_node['physic_operation'], operator_node['parameter_code'], operator_node['parameter_type'])
        count += 1
        if res2 == 'ERROR':
            return res2

    res3 = update_db("notebook", "add_sequence_1", '10', 'id', "=", str(walk_logs["notebook_id"]))
    if res3 == 'ERROR':
        return res2

    print("\033[0;32;40m\tsucceed\033[0m")
    return "SUCCEED"

def add_sequence_from_walk_logs(walk_logs, save_path):
    if walk_logs['is_img'] == True:
        update_db("notebook", "cant_sequence", '1', "id", '=', str(walk_logs["notebook_id"]))
        print("\033[0;33;40m\timage dataset pass\033[0m")
        return

    # if len(walk_logs['models'])==0:
    #     update_db("notebook", "cant_sequence", '3', "id", '=', str(walk_logs["notebook_id"]))
    #     print("\033[0;33;40m\tmodel number is 0\033[0m")
    #     return

    notebook_id = walk_logs['notebook_id']
    # res1 = ''
    # for model in walk_logs['models']:
    #     res1 = add_model(notebook_id, model)
    # print('model:', res1)
    # if res1 == 'ERROR':
    #     return res1
    count = 1

    for operator_node in walk_logs['operator_sequence']:
        res2 = add_operator(notebook_id, count, operator_node["data_object"],operator_node["data_object_value"], operator_node["operator_name"], operator_node['physic_operation'], operator_node['parameter_code'], operator_node['parameter_type'])
        count += 1
        if res2 == 'ERROR':
            return res2

    np.save(save_path + '/' + str(walk_logs['notebook_id']) + '.npy', walk_logs)
    res3 = update_db("notebook", "add_sequence", '1', 'id', "=", str(walk_logs["notebook_id"]))
    if len(walk_logs['operator_sequence'])==0:
        update_db("notebook", "cant_sequence", '2', "id", '=', str(walk_logs["notebook_id"]))
        print("\033[0;33;40m\tsequence length is 0\033[0m")
        # return
    if res3 == 'ERROR':
        return res2

    print("\033[0;32;40m\tsucceed\033[0m")
    return "SUCCEED"

def get_batch_no_seq_notebook_info(ip):
    cursor, db = create_connection()
    sql = "SELECT distinct title, id, scriptUrl FROM notebook WHERE cant_sequence = 2 and isdownload=1 and server_ip='" + ip + "'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        notebook_info = (row[1], row[0], row[2])
        result.append(notebook_info)
    return result

def get_batch_notebook_info(ip,type=1):
    cursor, db = create_connection()
    if type==1:
        sql = "SELECT distinct title, id, scriptUrl FROM notebook WHERE add_sequence=0 and cant_sequence=0 and isdownload=1 and server_ip='" + ip + "'"
    elif type==2:
        sql = "SELECT distinct title, id, scriptUrl FROM notebook WHERE add_sequence=0 and cant_sequence=2 and isdownload=1 and server_ip='" + ip + "'"
    elif type == 3:
        sql = "SELECT distinct notebook.title, notebook.id, notebook.scriptUrl FROM pair,notebook,dataset WHERE notebook.add_sequence=0 and notebook.isdownload=1 and pair.nid=notebook.id and dataset.id=pair.did and (notebook.server_ip='" + ip + "' or (notebook.is_transferred=1 and dataset.server_ip='" + ip + "'))";
    elif type == 4:
        sql = "SELECT distinct notebook.title, notebook.id, notebook.scriptUrl FROM pair,notebook,dataset WHERE notebook.add_sequence=0 and cant_sequence=2 and notebook.isdownload=1 and pair.nid=notebook.id and dataset.id=pair.did and (notebook.server_ip='" + ip + "' or (notebook.is_transferred=1 and dataset.server_ip='" + ip + "'))";
    elif type == 10:
        sql = "SELECT distinct title, id, scriptUrl FROM notebook WHERE add_sequence_1!=10 and isdownload=1 and server_ip='" + ip + "'"
    elif type == 11:
        sql = "SELECT distinct title, id, scriptUrl FROM notebook WHERE add_sequence_1=10 and add_sequence=0 and cant_sequence=0 and isdownload=1 and server_ip='" + ip + "'"
    elif type == 5:
        sql = "SELECT notebook.title,notebook.id,notebook.scriptUrl " \
              "FROM pair, notebook, dataset " \
              "WHERE notebook.id=pair.nid " \
              "and dataset.id=pair.did " \
              "and notebook.add_sequence=0 " \
              "and notebook.cant_sequence=0 " \
              "and notebook.isdownload=1 " \
              "and dataset.isdownload=1 " \
              "and dataset.server_ip='" + ip + "'"
    print(sql)
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

def get_pair(ip, type=1,notebook_id=-1):
    cursor, db = create_connection()
    if type == 0:
        sql = "SELECT pair.nid, dataset.dataSourceUrl " \
          "FROM pair, notebook, dataset " \
          "WHERE notebook.id=pair.nid " \
          "and dataset.id=pair.did " \
          "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3))" \
          "and notebook.add_run=0 " \
          "and notebook.server_ip='" + ip + "' " \
          "and dataset.isdownload=1 " \
          "and dataset.server_ip='" + ip + "' "
    if type == 2:
        sql = "SELECT pair.nid, dataset.dataSourceUrl " \
          "FROM pair, notebook, dataset " \
          "WHERE notebook.id=pair.nid " \
          "and dataset.id=pair.did " \
          "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3))" \
          "and notebook.add_run=2 " \
          "and notebook.server_ip='" + ip + "' " \
          "and dataset.isdownload=1 " \
          "and dataset.server_ip='" + ip + "' "
    elif type == 1:
        sql = "SELECT pair.nid, dataset.dataSourceUrl " \
              "FROM pair, notebook, dataset " \
              "WHERE notebook.id=pair.nid " \
              "and dataset.id=pair.did " \
              "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3))" \
              "and notebook.add_run=1 " \
              "and notebook.server_ip='" + ip + "' " \
              "and dataset.isdownload=1 " \
              "and dataset.server_ip='" + ip + "' " \
              "and notebook.id not in (select distinct notebook_id from result)"
    elif type == 3:
        sql = "SELECT pair.nid, dataset.dataSourceUrl " \
              "FROM pair, notebook, dataset " \
              "WHERE notebook.id=pair.nid " \
              "and dataset.id=pair.did " \
              "and notebook.add_run=3 " \
              "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3)) " \
              "and notebook.server_ip='" + ip + "' " \
              "and dataset.isdownload=1 " \
              "and dataset.server_ip='" + ip + "' " \
              "and notebook.id not in (select distinct notebook_id from result)"
    elif type == 4:
        sql = "SELECT pair.nid, dataset.dataSourceUrl " \
              "FROM pair, notebook, dataset " \
              "WHERE notebook.id=pair.nid " \
              "and dataset.id=pair.did " \
              "and notebook.add_run=4 " \
              "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3)) " \
              "and notebook.server_ip='" + ip + "' " \
              "and dataset.isdownload=1 " \
              "and dataset.server_ip='" + ip + "'"
    elif type == 5:
        sql = "SELECT pair.nid, dataset.dataSourceUrl, notebook.server_ip " \
              "FROM pair, notebook, dataset " \
              "WHERE notebook.id=pair.nid " \
              "and dataset.id=pair.did " \
              "and notebook.add_run=0 " \
              "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3)) " \
              "and dataset.isdownload=1 " \
              "and dataset.server_ip='" + ip + "'"
    elif type == 6:
        sql = "SELECT pair.nid, dataset.dataSourceUrl, notebook.server_ip FROM pair, notebook, dataset where pair.nid=notebook.id and pair.did=dataset.id and notebook.id=" + str(notebook_id)
    elif type == 7:
        sql = "SELECT pair.nid, dataset.dataSourceUrl, notebook.server_ip " \
              "FROM pair, notebook, dataset " \
              "WHERE notebook.id=pair.nid " \
              "and dataset.id=pair.did " \
              "and notebook.add_model=9 " \
              "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3)) " \
              "and dataset.isdownload=1 " \
              "and dataset.server_ip='" + ip + "'"
    print(sql)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        data_path = row[1].split('/')[-1].strip()
        result.append((row[0],data_path, row[2]))
    return result
def get_save_pair(ip):
    cursor, db = create_connection()
    sql = "SELECT pair.nid, dataset.dataSourceUrl, notebook.server_ip " \
          "FROM pair, notebook, dataset " \
          "WHERE notebook.id=pair.nid " \
          "and dataset.id=pair.did " \
          "and notebook.add_model=1 " \
          "and notebook.add_seq_df=0 " \
          "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3)) " \
          "and dataset.isdownload=1 " \
          "and dataset.server_ip='" + ip + "'"

    print(sql)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        data_path = row[1].split('/')[-1].strip()
        result.append((row[0],data_path, row[2]))
    return result

###################以下是统计##########################
def check_object_column(data_object_value, data_object,col_list):
        object_name = data_object
        if data_object == 'unknown':
            if 'test' in data_object_value or 'Test' in data_object_value:
                object_name = 'test'
            elif 'train' in data_object_value or 'Train' in data_object_value:
                object_name = 'train'
            elif 'data' in data_object_value or 'Data' in data_object_value:
                object_name = 'data'

        column_list = []
        # print(col_list)
        for i in col_list:
            # if '\''+i + '\'' in data_object_value or '\"'+i + '\"' in data_object_value or '.'+i in data_object_value:
            #     column_list.append(i)
            if data_object_value == None:
                continue
            if i in data_object_value:
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
    sql = "SELECT id,notebook_id,rank,logic_operation,data_object,physic_operation,operator,parameter_1_code,data_object_value FROM operator where notebook_id=" + str(notebook_id)
    cursor.execute(sql)
    operations = cursor.fetchall()

    print("\033[0;31;40m" + str(notebook_id) + "\033[0m")
    print('len(operations):', len(operations))
    column_tag_list = {}
    for i in col_list:
        column_tag_list[i] = 0
    for row in operations:
        print(row[0:8],row[-1])
        if row[6] == 'drop' or row[6]=='expm1':
            column_list,data_object = check_object_column(row[7],row[4],col_list)
            # if (row[4] == 'unknown'):
            #     continue
            #
            # # if type(column_list).__name__ == 'str' and column_list == 'all':
            # #     column_list = col_list
            # for column in column_list:
            #     if column not in seq[data_object].keys():
            #         seq[data_object][column] = []
            #     seq[data_object][column].append((row[3], row[-1], row[7]))
            #     column_tag_list[column] = 1
        else:
            column_list, data_object = check_object_column(row[-1], row[4], col_list)
            if(data_object == 'unknown'):
                continue

            if type(column_list).__name__ == 'str' and column_list == 'all':
                column_list = col_list
            for column in column_list:
                if column not in seq[data_object].keys():
                    seq[data_object][column] = []
                if column_tag_list[column]==0:
                    # seq[data_object][column].append((row[3], row[-1], row[7]))
                    seq[data_object][column].append(row[3])
    return seq

def find_special_notebook(type=3):
    if type == 3:
        cursor, db = create_connection()
        sql = "select id from notebook where isdownload=1 and server_ip='10.77.70.123' and cant_sequence = 3;"
        cursor.execute(sql)
        sql_res = cursor.fetchall()

        result = []
        for row in sql_res:
            result.append(row[0])
        return result


def stastic_operator_sequence_by_dataset(datasetid,data_root_path):
    notebook_title = []
    cursor, db = create_connection()
    sql = "SELECT notebook.id, notebook.title FROM notebook , datasources where notebook.id=datasources.notebook_id and datasources.sourceId=" + str(datasetid) + " and (notebook.add_sequence=1 and notebook.cant_sequence=0);"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    seq_list = []

    col_list = get_all_column_list(datasetid,data_root_path)
    print(col_list)
    count = 0

    ope_count_list = {}

    for row in sql_res:
        if row[1] in notebook_title:
            continue
        # if count == 2:
        notebook_title.append(row[1])
        res = get_column_seq(row[0],col_list)
        seq_list.append(res)

        ope_set = set()
        for i in res.keys():
            for j in res[i].keys():
                for k in res[i][j]:
                    ope_set.add(k)

        for i in ope_set:
            if i not in ope_count_list.keys():
                ope_count_list[i] = 0
            ope_count_list[i] += 1
    print(ope_count_list)
            # break
        # else:
        #     count += 1
    # print("\033[0;31;40mtrain\033[0m")
    # # print(seq_list[0]['train'])
    # for i in seq_list[0]['train']:
    #     print("\033[0;31;42m"+ i + "\033[0m")
    #     print(seq_list[0]['train'][i])
    # print("\033[0;31;40mtest\033[0m")
    # for i in seq_list[0]['test']:
    #     print("\033[0;31;42m"+ i + "\033[0m")
    #     print(seq_list[0]['test'][i])
    # print("\033[0;31;40mdata\033[0m")
    # for i in seq_list[0]['data']:
    #     print("\033[0;31;42m" + i + "\033[0m")
    #     print(seq_list[0]['data'][i])
    # column_name = col_list[2]
    # print(column_name)
    # for i in seq_list:
    #     if column_name in i['train']:
    #         print(i['train'][column_name])
    #     else:
    #         # print(i['train'].keys())
    #         # print("no column?")
    #         print([])
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

def operator_distribution_by_modeltyle(modeltype):
    cursor, db = create_connection()
    # 找到所有datasetid的id和title
    sql = "SELECT distinct(pair.did) " \
          "FROM pair, model, notebook " \
          "WHERE notebook.id = pair.nid " \
          "and notebook.id = model.notebook_id " \
          "and model.model_type = " + str(modeltype) + \
          " and (notebook.add_sequence=1 or notebook.cant_sequence=2)"
    print(sql)
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    now_count = []
    all_count = 0
    for row in sql_res:
        dataset_id = row[0]
        count_dic, this_all_count = count_operator(dataset_id, modeltype)
        all_count += this_all_count
        if now_count == []:
            now_count = count_dic
        else:
            for i in count_dic:
                if i not in now_count:
                    now_count[i] = 0
                now_count[i] += count_dic[i]

    # all_count = 0
    # for i in now_count:
    #     all_count += now_count[i]
    for i in now_count:
        now_count[i] = now_count[i]/all_count

    sql = "select (select count(distinct notebook.id) from notebook,model where notebook.add_sequence=1 and notebook.id=model.notebook_id and model.model_type=" + str(modeltype)  + ") as seq_num, (select count(distinct notebook.id) from notebook,model where notebook.cant_sequence=2 and notebook.id=model.notebook_id and model.model_type=" + str(modeltype) + ") as no_seq_num;"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    model = {}
    model["distribution"] = now_count
    for row in sql_res:
        model["no_seq_num"] = row[1]
        model["seq_num"] = row[0]
        print("have operations num:", row[0])
        print("no operations num:", row[1])
    return model

def static_all_model():
    model_dic = eval(CONFIG.get('models', 'model_dic'))
    result = {}
    for i in model_dic:
        model = operator_distribution_by_modeltyle(model_dic[i])
        result[i] = model
    print(result)
    np.save("../static/model_distribution_static_0921.npy", result)

def add_error(notebook_id, error_str):
    value_list = [notebook_id, error_str]
    column_list = ["notebook_id", "error_str"]
    print("\033[0;34;44madd error\033[0m")
    return insert_db("error", column_list, value_list)

def count_operator(datasetid, modeltype):
    title_list = []
    cursor, db = create_connection()
    #找到所有datasetid的id和title
    sql = "SELECT notebook.id, notebook.title " \
          "FROM notebook, pair, model " \
          "WHERE notebook.id = pair.nid " \
          "and notebook.id = model.notebook_id " \
          "and model.model_type = " + str(modeltype)+ \
          " and pair.did=" + str(datasetid)

    cursor.execute(sql)
    sql_res = cursor.fetchall()

    operator_count = {}
    all_count = 0
    for row in sql_res:
        if row[1] in title_list:
            continue
        title_list.append(row[1])

        sql1 = "SELECT logic_operation FROM operator WHERE notebook_id=" + str(row[0])
        cursor.execute(sql1)
        sql1_res = cursor.fetchall()
        is_in = set()
        for operator in sql1_res:
            if operator[0] not in operator_count.keys():
                operator_count[operator[0]] = 0
            if operator[0] in is_in:
                continue
            else:
                operator_count[operator[0]] += 1
                is_in.add(operator[0])
        all_count += 1
    print(operator_count)
    return operator_count, all_count

def delete_error_operator():
    cursor, db = create_connection()
    sql = 'SELECT id,notebook_id from operator'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    id_dic = {}

    for row in sql_res:
        opid = row[0]
        notebook_id = row[1]
        if notebook_id not in id_dic: #notebook的第一个operator
            id_dic[notebook_id] = opid
        elif id_dic[notebook_id]+1 == opid: #递增operator
            id_dic[notebook_id] = opid
        elif id_dic[notebook_id] < opid:
            sql = 'delete from operator where id=' + str(opid)
            cursor.execute(sql)
            print('delete id:' + str(opid) + ',end id:' + str(id_dic[notebook_id]))

def need_move(notebook_source_ip):
    cursor, db = create_connection()
    dataset_source_ip = '39.99.150.216'
    sql = "SELECT notebook.id FROM notebook,pair,dataset WHERE notebook.id = pair.nid and pair.did=dataset.id and notebook.server_ip != dataset.server_ip  and notebook.isdownload=1 and dataset.isdownload=1 and dataset.server_ip='" + dataset_source_ip  + "' and notebook.server_ip='" +  notebook_source_ip + "'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    if not os.path.exists('../mov2aliyun/'):
        os.mkdir('../mov2aliyun/')
    for row in sql_res:
        source_file_path = '../notebook/' + str(row[0]) + '.ipynb'
        object_file_path = '../mov2aliyun/' + str(row[0]) + '.ipynb'
        os.system('cp '+ source_file_path + ' ' + object_file_path)
        print(str(row[0]) + ' copy')


def file_transfer(dataset_source_ip, notebook_source_ip):
    import paramiko  # 用于调用scp命令
    from scp import SCPClient
    cursor, db = create_connection()
    sql = "SELECT notebook.id FROM notebook,pair,dataset WHERE notebook.id = pair.nid and pair.did=dataset.id and dataset.server_ip='" + dataset_source_ip  + "' and notebook.server_ip='" +  notebook_source_ip + "'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    count = 0
    t_file_list = []
    if os.path.exists('./filelist.npy'):
        t_file_list = list(np.load('./filelist.npy', allow_pickle=True))
    # print(t_file_list)
    # # print(sql)
    # sql_res = []
    # file_list = os.listdir('../walklogs/')
    # file_list.sort()
    # for file in file_list:
    #     sql_res.append(file.split('.')[0])
    if notebook_source_ip == '39.99.150.216':
        notebook_source_ip = 'localhost'
    if os.path.exists('./faillist.npy'):
        fail_list = list(np.load('./faillist.npy', allow_pickle=True))
    else:
        fail_list = []
    if os.path.exists('./transferred.npy'):
        transferred = list(np.load('./transferred.npy', allow_pickle=True))
    else:
        transferred = []
    for index,row in enumerate(sql_res):
        print(count)
        notebook_id = row[0]
        if notebook_id in transferred:
            continue
        if str(notebook_id)+'.npy' in t_file_list:
            print('in')
            continue
        if index % 100==0:
            np.save('./transferred.npy', transferred)
            np.save('./faillist.npy', fail_list)
        transferred.append(notebook_id)
        server_dict = eval(CONFIG.get('server', 'server'))

        host = dataset_source_ip
        port = server_dict[dataset_source_ip]['port']
        username = server_dict[dataset_source_ip]['username']
        password = server_dict[dataset_source_ip]['password']
        target_notebook_path = server_dict[dataset_source_ip]['npath']
        source_notebook_path = server_dict[notebook_source_ip]['npath']
        target_walklogs_path = server_dict[dataset_source_ip]['wpath']
        source_walklogs_path = server_dict[notebook_source_ip]['wpath']

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        ssh_client.connect(host, port, username, password)

        target_path = target_notebook_path + str(notebook_id) + '.ipynb'
        source_path = source_notebook_path + str(notebook_id) + '.ipynb'

        # cmd = 'scp '+ source_path + ' ' + username + '@' + host + ':' + target_path
        # os.system(cmd)
        # os.system(password)
        # cmd = 'scp ' + source_walklogs_path + ' ' + username + '@' + host + ':' + target_walklogs_path
        # os.system(cmd)
        # os.system(password)


        scpclient = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
        # conn = scpclient.invoke_shell()
        try:
            scpclient.put(source_path, target_path)
            count += 1
        except FileNotFoundError as e:
            print(e)
            print("系统找不到指定文件ipynb" + source_path)
        else:
            print(count, "notebook文件上传成功")
            update_db('notebook', 'is_transferred', 1, 'id', '=', row[0])

        # target_path = target_walklogs_path + str(notebook_id) + '.npy'
        # source_path = source_walklogs_path + str(notebook_id) + '.npy'
        #
        # scpclient = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
        # try:
        #     scpclient.put(source_path, target_path)
        #     count += 1
        # except Exception as e:
        #     print(e)
        #     print("系统找不到指定文件npy" + source_path)
        #     fail_list.append(notebook_id)
        # else:
        #     print(count, "walklogs文件上传成功")
    np.save('./transferred.npy',transferred)
    np.save('./faillist.npy', fail_list)

def update_model_3():
    cursor, db = create_connection()
    sql = "select id from notebook where add_model = 9"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    notebook_list = []
    for row in sql_res:
        notebook_list.append(int(row[0]))

    sql = "select distinct notebook_id from result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    co = 0
    for row in sql_res:
        if int(row[0]) in notebook_list:
            # sql = "update notebook set add_model=9 where id=" + str(row[0])
            # cursor.execute(sql)
            sql = 'delete from result where notebook_id=' + str(row[0])
            cursor.execute(sql)
            db.commit()
            co += 1
            print(co)

def get_walklogs_list():
    file_list = os.listdir('../walklogs/')
    np.save('./filelist.npy',file_list)

def update_notebook_by_mode():
    cursor, db = create_connection()
    sql = "select distinct notebook_id from result where model_type != '-1' and model_type != '1' and model_type != '2' and model_type != '3' and model_type != '4' and model_type != '5' and model_type != '6' and model_type != '7' and model_type != '8' and model_type != '9' and model_type != '10' and model_type != '11' and model_type != '12' and model_type != '13'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        # notebook_list.append(int(row[0]))
        sql = "select add_model from notebook where id=" + str(row[0])
        cursor.execute(sql)
        sql1_res = cursor.fetchall()
        for row1 in sql1_res:
            if row1[0] != 1:
                update_db('notebook','add_model',1,'id','=',int(row[0]))

def delete_zip_file():
    zip_file_list = os.listdir('../dataset')
    unzip_file_list = os.listdir('../unzip_dataset')
    for filename in zip_file_list:
        if filename in unzip_file_list:
            os.system('rm -f '+'../dataset/' +filename)
def get_example_notebookid():
    cursor, db = create_connection()
    sql = 'select distinct notebook_id from operator'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    in_operator = []
    for row in sql_res:
        in_operator.append(int(row[0]))
        # print(row[1],row[0])

    in_result = []
    sql = 'select distinct notebook_id from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    result = []
    for row in sql_res:
        # print('no_such')
        if int(row[0]) not in in_operator:
            sql = 'select dataset.server_ip from pair,dataset where pair.did=dataset.id and pair.nid=' + str(row[0])
            cursor.execute(sql)
            sql_res1 = cursor.fetchall()
            for row1 in sql_res1:
                if row1[0] =='39.99.150.216':
                    sql = 'select server_ip from notebook where id=' + str(row[0])
                    cursor.execute(sql)
                    sql_res2 = cursor.fetchall()
                    for row2 in sql_res2:
                        if row2[0] == '39.99.150.216':
                            result.append(row[0])
                        print(row[0],row1[0],row2[0])
                break
    return result

def delete_result(ip):
    cursor, db = create_connection()
    sql = 'select distinct notebook_id from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    in_result = []
    for row in sql_res:
        in_result.append(int(row[0]))

    sql = 'select pair.nid from pair,dataset where pair.did=dataset.id and dataset.server_ip=\'' + ip + '\' and isdownload=1'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        if int(row[0]) in in_result:
            sql = 'delete from result where notebook_id=\'' + str(row[0]) + '\''
            cursor.execute(sql)
            db.commit()
            print('delete:', str(row[0]))
            sql = 'update notebook set add_run=0 where id=\'' + str(row[0]) + '\''
            cursor.execute(sql)

def delete_operator(ip):
    cursor, db = create_connection()
    sql = 'select distinct notebook_id from operator'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    in_result = []
    for row in sql_res:
        in_result.append(int(row[0]))

    sql = 'select id from notebook where server_ip=\'' + ip + '\' and isdownload=1'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        if int(row[0]) in in_result:
            sql = 'delete from operator where notebook_id=\'' + str(row[0]) + '\''
            cursor.execute(sql)
            db.commit()
            print('delete:', str(row[0]))
            sql = 'update notebook set add_sequence=0 where id=\'' + str(row[0]) + '\''
            cursor.execute(sql)
            sql = 'update notebook set cant_sequence=0 where id=\'' + str(row[0]) + '\''
            cursor.execute(sql)

def check_iloc(data_object_value):
    r_node = ast.parse(data_object_value.strip())
    # print('data_object_value',data_object_value)
    if type(r_node.body[0].value.slice).__name__ != 'ExtSlice':
        return -1
    if type(r_node.body[0].value.slice.dims[0]).__name__ != 'Slice':
        return -1
    if r_node.body[0].value.slice.dims[0].lower == None and r_node.body[0].value.slice.dims[0].upper == None and r_node.body[0].value.slice.dims[0].step == None:
        if type(r_node.body[0].value.slice.dims[1]).__name__ == 'Index':
            return 1 # 取一列
        elif r_node.body[0].value.slice.dims[1].lower == None and type(r_node.body[0].value.slice.dims[1].upper).__name__ == 'Constant':# [:,:1]
            if r_node.body[0].value.slice.dims[1].upper.value == 1:
                return 1  # 取一列
            else:
                return 2
        elif type(r_node.body[0].value.slice.dims[1].lower).__name__ == 'Constant' and r_node.body[0].value.slice.dims[1].upper == None: # [:,-1:]
            if r_node.body[0].value.slice.dims[1].lower.value == -1:
                return 1  # 取一列
            else:
                return 2
        elif type(r_node.body[0].value.slice.dims[1].lower).__name__ == 'Constant' and type(r_node.body[0].value.slice.dims[1].upper).__name__ == 'Constant': # [:,0:1]
            if r_node.body[0].value.slice.dims[1].upper.value - r_node.body[0].value.slice.dims[1].lower.value == 1:
                return 1  # 取一列
            else:
                return 2
        else:
            return 2 # 取多列
    else:
        return -1 # 不符合取列
def delete_unfitable_iloc():
    cursor, db = create_connection()
    sql = 'select id,notebook_id,data_object_value from operator where operator=\'iloc\''
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    # left = 0
    # right = 1000
    count = 0
    for row in sql_res:
        # if count < left or count > right:
        #     count += 1
        #     continue
        check = check_iloc(row[2])
        print(row[2])
        print(check)
        count += 1
        notebook_id = row[1]
        if check == -1:
            sql = 'delete from operator where id=' + str(row[0])
            cursor.execute(sql)
            db.commit()
            sql = 'select rank from operator where notebook_id=' + str(notebook_id)
            cursor.execute(sql)
            sql_res1 = cursor.fetchall()
            rank_list = []
            for row1 in sql_res1:
                rank_list.append(int(row1[0]))
            rank_list.sort()
            print('rank_list_1:',rank_list)
            for index in range(0, len(rank_list)):
                if index == len(rank_list)-1:
                    break
                if rank_list[index+1] - rank_list[index] == 1:
                    continue
                else:
                    sql = 'update operator set rank = ' + str(rank_list[index]+1) + ' where notebook_id='+str(notebook_id)+' and rank=' + str(rank_list[index+1])
                    cursor.execute(sql)
                    db.commit()
                    rank_list[index + 1] = rank_list[index] + 1
            print('rank_list_1:', rank_list)
            print('delete:', str(row[0]))
if __name__ == '__main__':
    # CONFIG = configparser.ConfigParser()
    # CONFIG.read('config.ini')
    # model_dic = eval(CONFIG.get('operators', 'operations'))
    # print(type(model_dic))
    # stastic_operator_sequence_by_dataset(5407,'../spider/unzip_dataset')
    # operator_distribution_by_modeltyle(2)
    # static_all_model()
    # model = np.load('../static/model_distribution_static_0921.npy', allow_pickle=True).item()
    # for i in model:
    #     print("\033[0;33;40m" + i +"\033[0m")
    #     print("\033[0;33;41mdistribution\033[0m")
    #     print(model[i]['distribution'])
    #     print("\033[0;33;41mseq_num\033[0m")
    #     print(model[i]['seq_num'])
    #     print("\033[0;33;41mno_seq_num\033[0m")
    #     print(model[i]['no_seq_num'])


    # ip = '39.99.150.216'
    # cursor, db = create_connection()
    # sql = 'select count(distinct notebook_id), error_str from error group by error_str order by count(id)'
    # cursor.execute(sql)
    # sql_res = cursor.fetchall()
    # coun = 0
    # for row in sql_res:
    #     if 'No such file or directory: ' in row[1]:
    #         continue
    #     print(row[1])
    #     coun += 1
    # print(coun)
    # get_walklogs_list()
    # delete_result('10.77.70.127')
    # delete_operator('10.77.70.123')
    # delete_operator('10.77.70.124')
    # delete_operator('10.77.70.125')
    # get_example_notebookid()
    # delete_unfitable_iloc()
    # file_transfer('10.77.70.127','39.99.150.216')
    # delete_error_operator()
    # need_move('10.77.70.125')

    # update_model_3()
    # update_notebook_by_mode()
    cursor, db = create_connection()
    sql = 'select pair.nid,notebook.scriptUrl from pair,notebook where notebook.id=pair.nid and pair.did=30 or pair.did=16'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    # left = 0
    # right = 1000
    count = 0
    notebook_id_list = []
    for row in sql_res:
        if int(row[0]) not in notebook_id_list:
            notebook_id_list.append((int(row[0]),row[1]))
            count+=1
            print(count)
    np.save('30_16_notebooks.npy', notebook_id_list)
    # print(len(dataset_error_notebook_id_list))
