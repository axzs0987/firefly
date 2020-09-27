## -*- coding: utf-8 -*-

from utils import get_code_txt
from utils import add_sequence_from_walk_logs
from utils import get_batch_notebook_info
from utils import update_db
from utils import get_batch_no_seq_notebook_info
import ast
import astunparse
import configparser
import sys, getopt

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')

condition_switch = {
    "is_assign": False,
    "is_funcdef": False,
    "is_for": False,
    "is_if": False,
    "is_while": False,
    "now_func_name": "",
    "now_func_args": [],
}

walk_logs = {
    "notebook_id": -1,
    "notebook_title": "",
    "is_img": False, #是否是处理图像的notebook

    "import_lis": set(), #一共引用了那些包
    "pandas_alias": "", #pandas在代码中的简写
    "funcdef_sequence" : {}, #自定义函数的子序列
    "funcdef_values" : {}, #自定义函数被赋值的全局变量
    "operator_sequence": [], #notebook的序列是啥（我们想要的结果）

    "data_values": [], # 数据变量流
    "data_types": [], # 数据变量流的类型
    "models": set(), #notebook里边用到了哪些模型
    "models_values": {},  # 模型的变量
    "models_pred": {},  # 模型预测值变量

    "read_function": "", # 如果是用自定义函数读入文件，这个函数的名字
    "function_read_file_values": [], # 自定义读入文件函数里边，读入文件表示的变量
    "function_read_file_types": [],# 自定义读入文件函数里边，读入文件表示的变量类型

    "local_values": [], #所有的变量
    "estiminator_values": {}, #估计器的变量
    "estiminator_args": {}, #估计器的参数
    "estiminator_keywords": {}, #估计器的参数
}

def reflush_walk_logs_and_condition_switch(notebook_id, notebook_title):
    global condition_switch, walk_logs
    condition_switch = {
        "is_assign": False,
        "is_funcdef": False,
        "is_for": False,
        "is_if": False,
        "is_while": False,
        "now_func_name": "",
        "now_func_args": [],
    }

    walk_logs = {
        "notebook_id": notebook_id,
        "notebook_title": notebook_title,
        "notebook_id": -1,
        "notebook_title": "",
        "is_img": False, #是否是处理图像的notebook

        "import_lis": set(), #一共引用了那些包
        "pandas_alias": "", #pandas在代码中的简写
        "funcdef_sequence" : {}, #自定义函数的子序列
        "funcdef_values" : {}, #自定义函数被赋值的全局变量
        "operator_sequence": [], #notebook的序列是啥（我们想要的结果）

        "data_values": [], # 数据变量流
        "data_types": [], # 数据变量流的类型
        "models": set(), #notebook里边用到了哪些模型
        "models_values": {},  # 模型的变量
        "models_pred": {},  # 模型预测值变量

        "read_function": "", # 如果是用自定义函数读入文件，这个函数的名字
        "function_read_file_values": [], # 自定义读入文件函数里边，读入文件表示的变量
        "function_read_file_types": [],# 自定义读入文件函数里边，读入文件表示的变量类型

        "local_values": [], #所有的变量
        "estiminator_values": {}, #估计器的变量
        "estiminator_args": {}, #估计器的参数
        "estiminator_keywords": {}, #估计器的参数
    }

def check_model(st):
    # print("st:",st)
    global CONFIG, walk_logs
    model_dic = eval(CONFIG.get('models', 'model_dic'))
    if (type(st).__name__ == 'str'):
        is_in = False
        for i in model_dic.keys():
            if i == st:
                print("check_st:", st)
                walk_logs["models"].add(model_dic[i])
                is_in = True
        if is_in == True:
            return True
        else:
            return False
    else:
        return False

def check_img(st):
    """
    :param st: 字符串
    :return: 返回节点相关度模型

    判断代码节点是否包含模型信息，加入到logs中，并返回所包括的模型数量
    """

    if (type(st).__name__ == 'str'):
        if ("img" in st or "imread" in st or "image" in st):
            walk_logs["is_img"] = True

def check_operators(st, type):
    """
    :param st: 需要判断的字符串
    :param type: 需要判断的字符串是否属于某一种类型 -1则是不判断类型
    :return: True or False

    判断字符串是否是operator set中的一个
    """
    global CONFIG, walk_logs
    operator_set = eval(CONFIG.get('operators', 'operations')).keys()
    if st in operator_set:
        operator_info = eval(CONFIG.get('operators', 'operations'))[st]
        if type == -1 or operator_info["call_type"] == type:
            return True
        else:
            return False
    else:
        return False

def function_def(node):
    """
    :param node: 默认节点肯定为函数定义类型
    :return: 无

    当遇到函数定义类型节点时当处理方式：设置一些阀门，然后调用递归遍历函数
    """
    global condition_switch, walk_logs
    walk_logs["funcdef_sequence"][node.name] = []
    condition_switch["now_func_name"] = node.name
    condition_switch["is_funcdef"] = True
    for i in node.args.args:
        condition_switch["now_func_args"].append(walking(i))

    for item in node.body:
        if type(item).__name__ == 'Assign':
            condition_switch["is_assign"] = 1
            walking(item)
            condition_switch["is_assign"] = 0
        if type(item).__name__ == 'For':
            condition_switch["is_for"] = 1
            walking(item)
            condition_switch["is_for"] = 0
        if type(item).__name__ == 'If':
            condition_switch["is_if"] = 1
            walking(item)
            condition_switch["is_if"] = 0
        if type(item).__name__ == 'While':
            condition_switch["is_while"] = 1
            walking(item)
            condition_switch["is_while"] = 0
        else:
            walking(item)

    condition_switch["is_funcdef"] = False

def data_flowing(left, right):
    now_type = set()
    is_in = False
    for right_item in right:
        for i in range(0, len(walk_logs["data_values"])):  # 如果赋值右边的值，在数据流里面，则把赋值左边的值添加到数据流
            if (walk_logs["data_values"][i] == right_item):
                is_in = True
                now_type.add(walk_logs["data_types"][i])
    now_type = list(now_type)
    if is_in == True:
        for left_item in left:
            walk_logs["data_values"].append(left_item)
            if('train' in left_item or 'Train' in left_item or 'TRAIN' in left_item):
                walk_logs["data_types"].append('train')
            elif('test' in left_item or 'Test' in left_item or 'TEST' in left_item):
                walk_logs["data_types"].append('test')
            else:
                if len(now_type) == 1:  # 右边判断对象只有一个
                    walk_logs["data_types"].append(now_type[0])
                elif (len(now_type) == 0 or "unknown" in now_type):  # 右边判断对象为0个 或者 不为1个但unknown在其中
                    walk_logs["data_types"].append("Unknown")
                else:  # 右边判断对象大于1个且其中没有unknown
                    walk_logs["data_types"].append("data")


def create_new_sequence_node(operator_name, physic_operation, args_name, keywords_name, data_object, data_object_value):
    global walk_logs
    params_key = eval(CONFIG.get('operators', 'operations'))[operator_name]["params"]
    new_operator_node = {}
    new_operator_node["operator_name"] = operator_name
    new_operator_node["parameter_code"] = {}
    new_operator_node["parameter_type"] = {}
    new_operator_node["physic_operation"] = physic_operation
    new_operator_node["data_object"] = data_object
    new_operator_node["data_object_value"] = data_object_value
    for i in range(0, len(args_name)):
        args_code = args_name[i][0]
        args_type = args_name[i][1]
        for value in walk_logs['local_values']:
            if args_code == value[0]:
                args_code = value[2]
                break
        new_operator_node["parameter_code"][params_key[i]] = args_code
        new_operator_node["parameter_type"][params_key[i]] = args_type
    for i in keywords_name.keys():
        args_code = keywords_name[i][0]
        args_type = keywords_name[i][1]
        for value in walk_logs['local_values']:
            if args_code == value[0]:
                args_code = value[2]
                break
        new_operator_node["parameter_code"][i] = args_code
        new_operator_node["parameter_type"][i] = args_type

    walk_logs["operator_sequence"].append(new_operator_node)
    return new_operator_node

def create_new_funcdef_sequence_node(operator_name, physic_operation, args_name, keywords_name, data_param_id, data_param_name, data_object_value):
    global walk_logs,condition_switch
    params_key = eval(CONFIG.get('operators', 'operations'))[operator_name]["params"]
    new_operator_node = {}
    new_operator_node["operator_name"] = operator_name
    new_operator_node["parameter_code"] = {}
    new_operator_node["parameter_type"] = {}
    new_operator_node["physic_operation"] = physic_operation
    new_operator_node["data_param_id"] = data_param_id
    new_operator_node["data_param_name"] = data_param_name
    new_operator_node["data_object_value"] = data_object_value
    for i in range(0, len(args_name)):
        args_code = args_name[i][0]
        args_type = args_name[i][1]
        for value in walk_logs['local_values']:
            if args_code == value[0]:
                args_code = value[2]
                break
        new_operator_node["parameter_code"][params_key[i]] = args_code
        new_operator_node["parameter_type"][params_key[i]] = args_type
    for i in keywords_name.keys():
        key = i
        args_code = keywords_name[i][0]
        args_type = keywords_name[i][1]
        for value in walk_logs['local_values']:
            if args_code == value[0]:
                args_code = value[2]
                break
        new_operator_node["parameter_code"][key] = args_code
        new_operator_node["parameter_type"][key] = args_type

    walk_logs["funcdef_sequence"][condition_switch["now_func_name"]].append(new_operator_node)
    return new_operator_node

def functiondef_sequence_node_to_operator_sequence_node(fsn, data_object):
    new_operator_node = {}
    new_operator_node["operator_name"] = fsn["operator_name"]
    new_operator_node["parameter_code"] = fsn["parameter_name"]
    new_operator_node["parameter_type"] = fsn["parameter_type"]
    new_operator_node["physic_operation"] = fsn["physic_operation"]
    new_operator_node["data_object"] = data_object
    new_operator_node["parameter_code"] = fsn["parameter_code"]
    new_operator_node["parameter_type"] = fsn["parameter_type"]
    new_operator_node["data_object_value"] = fsn["data_object_value"]
    walk_logs["operator_sequence"].append(new_operator_node)
    return new_operator_node

def walking(node):
    global condition_switch, walk_logs, CONFIG
    #-----------------------------递归类型-----------------------------
    if type(node).__name__ == 'Expr':
        walking(node.value)
        return
    elif type(node).__name__ == 'ClassDef':
        return
    elif type(node).__name__ == 'Assign':
        # print(astunparse.unparse(node))
        ####################得到左，右数据对象########################
        assign_value_list = []
        assign_target_list = []
        result = walking(node.value)# 等号右边的值，可能是基础数据类型和高级数据类型,可能是list或者str
        if(type(result).__name__ == 'list'):
            assign_value_list = result
        else:
            assign_value_list.append(result)

        if(result == []):
            assign_value_list.append([])
        for target_node in node.targets:
            result = walking(target_node)  # 等号左边的值，可能是基础数据类型和高级数据类型(除了Call),可能是list或者str
            if (type(result).__name__ == 'list'):
                for i in result:
                    assign_target_list.append(i)
            elif (type(result).__name__ == 'str'):
                assign_target_list.append(result)
        #############判断估计器#################################
        is_estiminator = False
        for i in eval(CONFIG.get('operators', 'operations')).keys():
            if eval(CONFIG.get('operators', 'operations'))[i]["call_type"] == 3 and assign_value_list[0] == i:
                is_estiminator = True
        if is_estiminator == True and len(assign_value_list) == 3:
            walk_logs["estiminator_values"][assign_value_list[0]] = assign_target_list[0]
            walk_logs["estiminator_args"][assign_value_list[0]]= assign_value_list[1]
            walk_logs["estiminator_keywords"][assign_value_list[0]] = assign_value_list[2]
        #############判断模型#################################
        is_model = False
        for i in eval(CONFIG.get('models', 'model_dic')).keys():
            if assign_value_list[0] == i:
                is_model = True
        if is_model == True:
            walk_logs["models_values"][assign_value_list[0]] = assign_target_list[0]
        #############判断pred#################################
        is_pred = False
        model_name = ''
        for i in walk_logs["models_values"]:
            if assign_value_list[0] == walk_logs["models_values"][i]:
                is_pred = True
                model_name = i
        if is_pred == True:
            walk_logs["models_pred"][model_name] = assign_target_list[0]
        #############增加local数据##############################
        for assign_target in assign_target_list:
            is_in = False
            for i in range(0,len(walk_logs["local_values"])):
                if(walk_logs["local_values"][i][0] == assign_target):
                    if(type(node.value).__name__ != 'Name'):
                        walk_logs["local_values"][i][1] = type(node.value).__name__
                        walk_logs["local_values"][i][2] = astunparse.unparse(node.value)
                    else:
                        for j in walk_logs["local_values"]:
                            if(j[0] == assign_value_list[0]):
                                walk_logs["local_values"][i][1] = type(node.value).__name__
                                walk_logs["local_values"][i][2] = astunparse.unparse(node.value)
                    is_in = True
            if is_in == False:
                if (type(node.value).__name__ != 'Name'):
                    try:
                        walk_logs["local_values"].append([assign_target,type(node.value).__name__, astunparse.unparse(node.value)])
                    except:
                        continue
                else:
                    for j in walk_logs["local_values"]:
                        if (j[0] == assign_value_list[0]):
                            walk_logs["local_values"].append([assign_target, j[1], astunparse.unparse(node.value)])
        #############判断数据对象，更改数据流#########################
        if(condition_switch["is_funcdef"] == False): #函数定义部分的数据流不加入到其中
            data_flowing(assign_target_list, assign_value_list)

        #############处理读入收据事件#########################
        if assign_value_list[0] == "This is a read file functions!!!!!": #自定义函数读入，由于是赋值，所以肯定是外部读入
            for assign_target in assign_target_list:
                if (len(assign_target_list) == 1):
                    walk_logs["data_values"].append(assign_target)
                    if ('train' in assign_target or 'Train' in assign_target or 'TRAIN' in assign_target):
                        walk_logs["data_types"].append("train")
                    elif ('test' in assign_target or 'Test' in assign_target or 'TEST' in
                          assign_target):
                        walk_logs["data_types"].append("test")
                    else:
                        walk_logs["data_types"].append("data")


        elif type(node.value).__name__ == 'Call': # 只考虑read函数了
            if type(node.value.func).__name__ == 'Attribute':
                if node.value.func.attr == 'read_csv' or node.value.func.attr == 'read_pickle' or node.value.func.attr == 'read_table' or node.value.func.attr == 'read_fwf' or \
                        node.value.func.attr == 'read_clipboard' or node.value.func.attr == 'read_excel' or node.value.func.attr == 'ExcelFile.parse' or node.value.func.attr == 'ExcelWriter' or \
                        node.value.func.attr == 'read_json' or node.value.func.attr == 'json_normalize' or node.value.func.attr == 'build_table_schema' or node.value.func.attr == 'read_html' or \
                        node.value.func.attr == 'read_hdf' or node.value.func.attr == 'read_feather' or node.value.func.attr == 'read_parquet' or node.value.func.attr == 'read_orc' or \
                        node.value.func.attr == 'read_sas' or node.value.func.attr == 'read_spss' or node.value.func.attr == 'read_sql_table' or node.value.func.attr == 'read_sql_query' or \
                        node.value.func.attr == 'read_sql' or node.value.func.attr == 'read_gbq' or node.value.func.attr == 'read_stata':
                    if (condition_switch["is_funcdef"] == True): #如果读入数据发生在用户自定义函数内，则记录这个函数名，在调用函数时记得处理，把读入数据的变量存入function_read_file_values
                        condition_switch["read_function"] = condition_switch["now_func_name"]
                        for assign_target in assign_target_list:
                            if (len(assign_target_list) == 1):
                                walk_logs["function_read_file_values"].append(assign_target)
                                if ('train' in assign_target or 'Train' in assign_target or 'TRAIN' in assign_target):
                                    walk_logs["function_read_file_types"].append("train")
                                elif ('test' in assign_target or 'Test' in assign_target or 'TEST' in
                                      assign_target):
                                    walk_logs["function_read_file_types"].append("test")
                                else:
                                    walk_logs["function_read_file_types"].append("data")
                    else: # 如果正常读入函数，或者函数内部定义全局变量
                        if (len(assign_target_list) == 1):
                            walk_logs["data_values"].append(assign_target_list[0])
                            if ('train' in assign_target_list[0] or 'Train' in assign_target_list[0] or 'TRAIN' in assign_target_list[0]):
                                walk_logs["data_types"].append("train")
                            elif ('test' in assign_target_list[0] or 'Test' in assign_target_list[0] or 'TEST' in assign_target_list[0] ):
                                walk_logs["data_types"].append("test")
                            else:
                                walk_logs["data_types"].append("data")

        return
    elif type(node).__name__ == 'For':
        condition_switch["is_for"] = 1
        for body_node in node.body:
            walking(body_node)
        condition_switch["is_for"] = 0
        return
    elif type(node).__name__ == 'If':
        condition_switch["is_if"] = 1
        for body_node in node.body:
            walking(body_node)
        condition_switch["is_if"] = 0
        return
    elif type(node).__name__ == 'While':
        condition_switch["is_while"] = 1
        for body_node in node.body:
            walking(body_node)
        condition_switch["is_while"] = 0
        return
    elif type(node).__name__ == 'Try':
        condition_switch["is_try"] = 1
        for body_node in node.body:
            walking(body_node)
        condition_switch["is_try"] = 0
        return
    #-------------------------Call,包含插入序列操作--------------------
    elif type(node).__name__ == 'Call':
        func_name = ""
        return_list = []
        # print('in call')
        if(type(node.func).__name__ == 'Name'): # 需要得到自定义函数名或者函数名
            func_name = walking(node.func) #有可能是attr和name，返回函数主体，自定义函数只有可能是name
            return_list.append(func_name)
        # add_ret = walking(node.func)
        ##############处理参数和keywords##################
        args_name = [] #[(1,int),(2,str),,...]
        keywords_name = {} #[("axis": (0, int)),.....]

        for arg_node in node.args:
            one_arg_result = walking(arg_node)
            if(type(arg_node).__name__ != "Name"):
                try:
                    args_name.append((astunparse.unparse(arg_node), type(arg_node).__name__))
                except:
                    continue
            else:
                is_in = False
                for j in walk_logs["local_values"]:
                    if (j[0] == walking(arg_node)):
                        args_name.append((astunparse.unparse(arg_node), j[1]))
                        is_in = True
                        break
                if is_in == False:
                    args_name.append((astunparse.unparse(arg_node), 'unknown variable'))
            if type(one_arg_result).__name__ == 'str':
                return_list.append(one_arg_result)
            elif type(one_arg_result).__name__ == 'List':
                for item in one_arg_result:
                    return_list.append(item)



        for keyword_node in node.keywords:
            key = keyword_node.arg
            one_arg_result = walking(keyword_node.value)
            if(type(keyword_node).__name__ != "Name"):
                keywords_name[key] = (astunparse.unparse(keyword_node), type(keyword_node).__name__)
            else:
                for j in walk_logs["local_values"]:
                    if (j[0] == walking(keyword_node)):
                        keywords_name[key] = (astunparse.unparse(keyword_node), j[1])
                    else:
                        keywords_name[key] = (astunparse.unparse(keyword_node), 'unknown variable')
            if type(one_arg_result).__name__ == 'str':
                return_list.append(one_arg_result)
            elif type(one_arg_result).__name__ == 'List':
                for item in one_arg_result:
                    return_list.append(item)


        ##############处理自定义函数#######################
        if func_name in walk_logs["funcdef_sequence"].keys(): # 如果这个函数是自定义函数
            this_funcdef_sequence = walk_logs["funcdef_sequence"][func_name]
            for operator_node in this_funcdef_sequence:
                for num in range(0, len(walk_logs["data_values"])):
                    if operator_node["data_param_name"] != None:
                        if walk_logs["data_values"][num] == args_name[operator_node["data_param_id"]][0] or walk_logs["data_values"][num] == keywords_name[operator_node["data_param_name"]][0]: # 如果自定义函数内部序列的数据参数对象，在这里对应数据是数据流里，则把增额操作加入到主序列里边
                            data_object = walk_logs["data_types"][num]
                            functiondef_sequence_node_to_operator_sequence_node(operator_node, data_object)

            #************处理读入函数******************
            if func_name == walk_logs["read_function"] and condition_switch["is_assign"] == True: #如果这个函数是读入文件函数，且在外赋值
                return "This is a read file functions!!!!!"
            elif func_name == walk_logs["read_function"] and condition_switch["is_assign"] == False: #如果这个函数是读入文件函数，在内部赋值，把function_read_file_values迁移到data_values
                for i in walk_logs["function_read_file_values"]:
                    walk_logs["data_values"].append(i)
                for i in walk_logs["function_read_file_types"]:
                    walk_logs["data_types"].append(i)
                return return_list

        ###############处理主流Call#####################
        if condition_switch['is_funcdef'] == True:
            #####增加序列节点###########
            # *******判断type0*************
            if type(node.func).__name__ == 'Attribute':
                if check_operators(node.func.attr, 0):  # 如果命中操作
                    physic_operation = eval(CONFIG.get('operators','operations'))[node.func.attr]["physic_operations"][0]
                    if node.func.attr == 'fillna':
                        physic_operation = "filling_by_stratage"
                        if len(args_name) != 0:
                            if (args_name[0][1] == 'Str' or args_name[0][1] == 'Num'): # 如果fillna的第一个参数的类型是常数
                                physic_operation = "filling_constant"
                    if node.func.attr == 'map':
                        if len(node.args) > 0:
                            if type(node.args[0]).__name__ == 'Dict':
                                dataset_name = walking(node.func.value)
                                for i in range(0, len(condition_switch["now_func_args"])):
                                    if (dataset_name == condition_switch["now_func_args"][i]):
                                        create_new_funcdef_sequence_node(node.func.attr, physic_operation, args_name,
                                                                         keywords_name, data_param_id=i,
                                                                         data_param_name=dataset_name,
                                                                         data_object_value=astunparse.unparse(node.func))
                    elif node.func.attr == 'drop':
                        can_add = 0
                        if len(node.args) > 1:
                            if type(node.args[1]).__name__ == 'Num':
                                if node.args[1].n == 1: # 如果存在两个参数，且第二个参数为1
                                    can_add = 1
                        else:
                            for key_node in node.keywords:
                                key = key_node.arg
                                if key == "axis":
                                    if type(key_node.value).__name__ == 'Num':
                                        if key_node.value.n == 1:  # 如果存在两个参数，且第二个参数为1
                                            can_add = 1
                                            break
                                elif key == 'columns':
                                    can_add = 1
                                    break
                                else:
                                    continue
                        if can_add == 1:
                            dataset_name = walking(node.func.value)
                            for i in range(0, len(condition_switch["now_func_args"])):
                                if (dataset_name == condition_switch["now_func_args"][i]):
                                    create_new_funcdef_sequence_node(node.func.attr, physic_operation, args_name,
                                                                     keywords_name, data_param_id=i,
                                                                     data_param_name=dataset_name,
                                                                     data_object_value=astunparse.unparse(node.func))



                    else:
                        dataset_name = walking(node.func.value)
                        for i in range(0, len(condition_switch["now_func_args"])):
                            if (dataset_name == condition_switch["now_func_args"][i]):
                                create_new_funcdef_sequence_node(node.func.attr, physic_operation, args_name, keywords_name, data_param_id=i, data_param_name=dataset_name, data_object_value=astunparse.unparse(node.func))

                elif check_operators(node.func.attr, 2):  # 如果命中操作
                    attr_body = walking(node.func.value)
                    if attr_body == walk_logs["pandas_alias"]:
                        physic_operation = eval(CONFIG.get('operators', 'operations'))[node.func.attr]["physic_operations"][0]
                        if(len(node.args) != 0):
                            dataset_name = walking(node.args[0])
                        else:
                            dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[node.func.attr]['params'][0]]
                        for i in range(0, len(condition_switch["now_func_args"])):
                            if (dataset_name == condition_switch["now_func_args"][i]):
                                create_new_funcdef_sequence_node(node.func.attr, physic_operation, args_name,
                                                                  keywords_name, data_param_id=i,
                                                                  data_param_name=dataset_name,data_object_value=astunparse.unparse(node.args[0]))
                elif check_operators(node.func.attr, 4):  # 如果命中4操作 np.clip
                    attr_body = walking(node.func.value)
                    physic_operation = eval(CONFIG.get('operators', 'operations'))[node.func.attr]["physic_operations"][
                        0]
                    if (len(node.args) != 0):
                        dataset_name = walking(node.args[0])
                    elif eval(CONFIG.get('operators', 'operations'))[node.func.attr]['params'][0] in keywords_name:
                        dataset_name = keywords_name[
                            eval(CONFIG.get('operators', 'operations'))[node.func.attr]['params'][0]]
                    else:
                        dataset_name = attr_body
                    is_in = False
                    for i in range(0, len(walk_logs["data_values"])):
                        if (dataset_name == walk_logs["data_values"][i]):
                            data_object = walk_logs["data_types"][i]
                            is_in = True
                        else:
                            # print(walk_logs["data_values"][i])
                            if type(walk_logs["data_values"][i]).__name__ == 'list':
                                for j in walk_logs["data_values"][i]:
                                    if j in astunparse.unparse(node.func.value):
                                        data_object = j
                                        is_in = True
                            elif walk_logs["data_values"][i] in astunparse.unparse(node.func.value):
                                data_object = walk_logs["data_types"][i]
                                is_in = True

                    # if is_in == False:
                    #     data_object = "unknown"
                    #     walk_logs["data_values"].append(dataset_name)
                    #     walk_logs["data_types"].append(data_object)
                    if is_in == True:
                        if len(node.args) != 0:
                            create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                     data_object, astunparse.unparse(node.args[0]))
                        else:
                            in_key = False
                            ind = 0
                            for i in range(0, len(node.keywords)):
                                key = node.keywords[i].arg
                                if key == eval(CONFIG.get('operators', 'operations'))[node.func.attr]["params"][0]:
                                    ind = i
                                    in_key = True
                                    break
                            if in_key == True:
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                         data_object, astunparse.unparse(node.keywords[ind]))
                            else:
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                         data_object, astunparse.unparse(node.func.value))


                elif type(node.func.value).__name__ == "Name":
                    attr_body = walking(node.func.value)
                    is_estiminator = False
                    estiminator_name = ""
                    estiminator_args = []
                    estiminator_keywords = {}

                    is_model = False
                    for i in walk_logs["estiminator_values"]:
                        if walk_logs["estiminator_values"][i] == attr_body:
                            is_estiminator = True
                            estiminator_name = i
                            estiminator_args = walk_logs["estiminator_args"][i]
                            estiminator_keywords = walk_logs["estiminator_args"][i]

                    for i in walk_logs["models_values"]:
                        if walk_logs["models_values"][i] == attr_body:
                            is_model = True


                    if is_estiminator == True and (node.func.attr == 'fit_transform' or node.func.attr == 'transform'):
                        if(len(node.args) != 0):
                            dataset_name = walking(node.args[0])
                        else:
                            dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[estiminator_name]['params'][0]]
                        physic_operation = eval(CONFIG.get('operators', 'operations'))[estiminator_name]["physic_operations"][0]
                        for i in range(0, len(condition_switch["now_func_args"])):
                            if (dataset_name == condition_switch["now_func_args"][i]):
                                create_new_funcdef_sequence_node(node.func.attr, physic_operation, estiminator_args,
                                                                 estiminator_keywords, data_param_id=i,
                                                                  data_param_name=dataset_name,data_object_value=astunparse.unparse(node.args[0]))

                    elif is_model == True and (node.func.attr == 'predict'):
                        return attr_body
                    else:
                        res = walking(node.func)
                elif type(node.func.value).__name__ == 'Call':
                    args_name1 = []  # [(1,int),(2,str),,...]
                    keywords_name1 = {}  # [("axis": (0, int)),.....]
                    for arg_node in node.func.value.args:
                        one_arg_result = walking(arg_node)
                        if (type(arg_node).__name__ != "Name"):
                            args_name1.append((astunparse.unparse(arg_node), type(arg_node).__name__))
                        else:
                            is_in = False
                            for j in walk_logs["local_values"]:
                                if (j[0] == walking(arg_node)):
                                    args_name1.append((astunparse.unparse(arg_node), j[1]))
                                    is_in = True
                                    break
                            if is_in == False:
                                args_name1.append((astunparse.unparse(arg_node), 'unknown variable'))
                        if type(one_arg_result).__name__ == 'str':
                            return_list.append(one_arg_result)
                        elif type(one_arg_result).__name__ == 'List':
                            for item in one_arg_result:
                                return_list.append(item)

                    for keyword_node in node.func.value.keywords:
                        key = keyword_node.arg
                        one_arg_result = walking(keyword_node.value)
                        if (type(keyword_node).__name__ != "Name"):
                            keywords_name1[key] = (astunparse.unparse(keyword_node), type(keyword_node).__name__)
                        else:
                            for j in walk_logs["local_values"]:
                                if (j[0] == walking(keyword_node)):
                                    keywords_name1[key] = (astunparse.unparse(keyword_node), j[1])
                                else:
                                    keywords_name1[key] = (astunparse.unparse(keyword_node), 'unknown variable')
                        if type(one_arg_result).__name__ == 'str':
                            return_list.append(one_arg_result)
                        elif type(one_arg_result).__name__ == 'List':
                            for item in one_arg_result:
                                return_list.append(item)

                    func_name = ""
                    if(type(node.func.value.func).__name__ == 'Name'):
                        func_name = walking(node.func.value)
                    is_estiminator = False
                    estiminator_name = ""
                    for i in eval(CONFIG.get('operators', 'operations')).keys():
                        if eval(CONFIG.get('operators', 'operations'))[i]["call_type"] == 3 and func_name == i:
                            is_estiminator = True
                            estiminator_name = func_name
                    if is_estiminator == True and (node.func.attr == 'fit_transform' or node.func.attr == 'transform'):
                        if(len(node.args) != 0):
                            dataset_name = walking(node.args[0])
                        else:
                            dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[estiminator_name]['params'][0]]
                        physic_operation = eval(CONFIG.get('operators', "operations"))[estiminator_name]["physic_operations"][0]
                        for i in range(0, len(condition_switch["now_func_args"])):
                            if (dataset_name == condition_switch["now_func_args"][i]):
                                create_new_funcdef_sequence_node(estiminator_name, physic_operation, args_name,
                                                                  keywords_name, data_param_id=i,
                                                                  data_param_name=dataset_name,data_object_value=astunparse.unparse(node.args[0]))
                    else:
                        res = walking(node.func)
                        if (type(node.func) != "Name"):
                            if type(res).__name__ == 'list':
                                for i in res:
                                    return_list.insert(0, i)
                            else:
                                return_list.insert(0, res)
                else:
                    res = walking(node.func)

            elif type(node.func).__name__ == 'Name':
                func_body = walking(node.func)
                # *******判断type2*************
                if check_operators(func_body, 2):
                    physic_operation = eval(CONFIG.get('operators', 'operations'))[func_body]["physic_operations"][0]
                    if(len(node.args) != 0):
                        dataset_name = walking(node.args[0])
                    else:
                        dataset_name = keywords_name[
                            eval(CONFIG.get('operators', 'operations'))[func_body]['params'][0]]
                    for i in range(0, len(condition_switch["now_func_args"])):
                        if (dataset_name == condition_switch["now_func_args"][i]):
                            create_new_funcdef_sequence_node(node.func.attr, physic_operation, args_name, keywords_name, data_param_id=i, data_param_name=dataset_name,data_object_value=astunparse.unparse(node.args[0]))

                # *******判断type3*************
                elif check_operators(func_body, 3):
                    return [func_body,args_name,keywords_name]
                elif check_operators(func_body, 4):  # 如果命中4操作 np.clip
                    physic_operation = eval(CONFIG.get('operators', 'operations'))[func_body]["physic_operations"][0]
                    if (len(node.args) != 0):
                        dataset_name = walking(node.args[0])
                    else:
                        dataset_name = keywords_name[
                            eval(CONFIG.get('operators', 'operations'))[func_body]['params'][0]]
                    for i in range(0, len(condition_switch["now_func_args"])):
                        if (dataset_name == condition_switch["now_func_args"][i]):
                            create_new_funcdef_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                             data_param_id=i, data_param_name=dataset_name,
                                                             data_object_value=astunparse.unparse(node.args[0]))


                elif check_model(func_body):
                    return func_body
                else:
                    res = walking(node.func)

            return return_list
        else: #如果不是自定义函数内
            #####增加序列节点###########
            #*******判断type0*************
            if type(node.func).__name__ == 'Attribute':
                # print('in call attribute')
                if check_operators(node.func.attr, 0): # 如果1命中操作 dataset.fillna
                    physic_operation = eval(CONFIG.get('operators', 'operations'))[node.func.attr]["physic_operations"][0]
                    if node.func.attr == 'fillna':
                        physic_operation = "filling_by_stratage"
                        if len(args_name) != 0:
                            if (args_name[0][1] == 'Str' or args_name[0][1] == 'Num'): # 如果fillna的第一个参数的类型是常数
                                physic_operation = "filling_constant"
                    if node.func.attr == 'map':
                        if len(node.args) > 0:
                            if type(node.args[0]).__name__ == 'Dict':
                                dataset_name = walking(node.func.value)
                                return_list.insert(0, dataset_name)
                                is_in = False
                                for i in range(0, len(walk_logs["data_values"])):
                                    if (dataset_name == walk_logs["data_values"][i]):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                                    else:
                                        if walk_logs["data_values"][i] in astunparse.unparse(node.func.value):
                                            data_object = walk_logs["data_types"][i]
                                            is_in = True

                                # if is_in == False:
                                #     data_object = "unknown"
                                #     walk_logs["data_values"].append(dataset_name)
                                #     walk_logs["data_types"].append(data_object)
                                if is_in == True:
                                    create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                         data_object, astunparse.unparse(node.func))
                    elif node.func.attr == 'drop':
                        can_add = 0
                        if len(node.args) > 1:
                            if type(node.args[1]).__name__ == 'Num':
                                if node.args[1].n == 1:  # 如果存在两个参数，且第二个参数为1
                                    can_add = 1
                        else:
                            for key_node in node.keywords:
                                key = key_node.arg
                                if key == "axis":
                                    if type(key_node.value).__name__ == 'Num':
                                        if key_node.value.n == 1:  # 如果存在两个参数，且第二个参数为1
                                            can_add = 1
                                            break
                                elif key == 'columns':
                                    can_add = 1
                                    break
                                else:
                                    continue
                        if can_add == 1:
                            dataset_name = walking(node.func.value)
                            return_list.insert(0, dataset_name)
                            is_in = False
                            for i in range(0, len(walk_logs["data_values"])):
                                if (dataset_name == walk_logs["data_values"][i]):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                                else:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.func.value):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True

                            # if is_in == False:
                            #     data_object = "unknown"
                            #     walk_logs["data_values"].append(dataset_name)
                            #     walk_logs["data_types"].append(data_object)
                            if is_in == True:
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                     data_object, astunparse.unparse(node.func))

                    else:
                        dataset_name = walking(node.func.value)
                        return_list.insert(0, dataset_name)
                        is_in = False
                        for i in range(0, len(walk_logs["data_values"])):
                            if(dataset_name == walk_logs["data_values"][i]):
                                data_object = walk_logs["data_types"][i]
                                is_in = True
                            else:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.func.value):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True

                        # if is_in == False:
                        #     data_object = "unknown"
                        #     walk_logs["data_values"].append(dataset_name)
                        #     walk_logs["data_types"].append(data_object)
                        if is_in == True:
                            create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name, data_object, astunparse.unparse(node.func))

                elif check_operators(node.func.attr, 2):  # 如果命中2操作 pd.dummies
                    attr_body = walking(node.func.value)
                    # print(node.func.attr)
                    if attr_body == walk_logs["pandas_alias"]:
                        physic_operation = eval(CONFIG.get('operators', 'operations'))[node.func.attr]["physic_operations"][0]
                        if(len(node.args) != 0):
                            dataset_name = walking(node.args[0])
                        else:
                            dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[node.func.attr]['params'][0]]
                        is_in = False
                        for i in range(0, len(walk_logs["data_values"])):
                            if (dataset_name == walk_logs["data_values"][i]):
                                data_object = walk_logs["data_types"][i]
                                is_in = True
                            else:
                                if len(node.args) != 0:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.args[0]):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                                else:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.keywords):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                        # if is_in == False:
                        #     data_object = "unknown"
                        #     walk_logs["data_values"].append(dataset_name)
                        #     walk_logs["data_types"].append(data_object)
                        if is_in == True:
                            if len(node.args) != 0:
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name, data_object, astunparse.unparse(node.args[0]))
                            else:
                                ind = 0
                                for i in range(0, len(node.keywords)):
                                    key = node.keywords[i].arg
                                    if key == eval(CONFIG.get('operators', 'operations'))[node.func.attr]["params"][0]:
                                        ind = i
                                        break
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                         data_object, astunparse.unparse(node.keywords[ind]))
                elif check_operators(node.func.attr, 4):  # 如果命中4操作 np.clip
                    attr_body = walking(node.func.value)
                    physic_operation = eval(CONFIG.get('operators', 'operations'))[node.func.attr]["physic_operations"][0]
                    # print(astunparse.unparse(node))
                    # print(node.args)
                    if(len(node.args) != 0):
                        dataset_name = walking(node.args[0])
                    elif eval(CONFIG.get('operators', 'operations'))[node.func.attr]['params'][0] in keywords_name:
                        dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[node.func.attr]['params'][0]]
                    else:
                        dataset_name = attr_body
                    is_in = False
                    for i in range(0, len(walk_logs["data_values"])):
                        if (dataset_name == walk_logs["data_values"][i]):
                            data_object = walk_logs["data_types"][i]
                            is_in = True
                        else:
                            if len(node.args) != 0:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.args[0]):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                            else:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.keywords):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                                elif walk_logs["data_values"][i] in astunparse.unparse(node.func.value):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                    # if is_in == False:
                    #     data_object = "unknown"
                    #     walk_logs["data_values"].append(dataset_name)
                    #     walk_logs["data_types"].append(data_object)
                    if is_in == True:
                        if len(node.args) != 0:
                            create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name, data_object, astunparse.unparse(node.args[0]))
                        else:
                            in_key = False
                            ind = 0
                            for i in range(0, len(node.keywords)):
                                key = node.keywords[i].arg
                                if key == eval(CONFIG.get('operators', 'operations'))[node.func.attr]["params"][0]:
                                    ind = i
                                    in_key = True
                                    break
                            if in_key == True:
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                     data_object, astunparse.unparse(node.keywords[ind]))
                            else:
                                create_new_sequence_node(node.func.attr, physic_operation, args_name, keywords_name,
                                                         data_object, astunparse.unparse(node.func.value))
                elif type(node.func.value).__name__ == "Name": # 如果Attribute的主体是一个名字 这里有分布esitiminator.fit_transform
                    # print("in call attribute name")
                    attr_body = walking(node.func.value)
                    is_estiminator = False
                    estiminator_name = ""
                    estiminator_args = []
                    estiminator_keywords = {}

                    is_model = False
                    for i in walk_logs["estiminator_values"]:
                        if walk_logs["estiminator_values"][i] == attr_body:
                            is_estiminator = True
                            estiminator_name = i
                            estiminator_args = walk_logs["estiminator_args"][i]
                            estiminator_keywords = walk_logs["estiminator_keywords"][i]
                    for i in walk_logs["models_values"]:
                        if walk_logs["models_values"][i] == attr_body:
                            is_model = True

                    if is_estiminator == True and (node.func.attr == 'fit_transform' or node.func.attr == 'transform'):
                        # print(astunparse.unparse(node))
                        if(len(node.args) != 0):
                            dataset_name = walking(node.args[0])
                        else:
                            dataset_name = walking(node.keywords[0].value)
                        is_in = False
                        for i in range(0, len(walk_logs["data_values"])):
                            if (dataset_name == walk_logs["data_values"][i]):
                                data_object = walk_logs["data_types"][i]
                                is_in = True
                            else:
                                if len(node.args) != 0:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.args[0]):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                                else:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.keywords[0]):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                        # if is_in == False:
                        #     data_object = "unknown"
                        #     walk_logs["data_values"].append(dataset_name)
                        #     walk_logs["data_types"].append(data_object)
                        if is_in == True:
                            physic_operation = eval(CONFIG.get('operators', 'operations'))[estiminator_name]["physic_operations"][0]
                            if(len(node.args) != 0):
                                create_new_sequence_node(estiminator_name, physic_operation, estiminator_args, estiminator_keywords, data_object, astunparse.unparse(node.args[0]))
                            else:
                                create_new_sequence_node(estiminator_name, physic_operation, estiminator_args,
                                                         estiminator_keywords, data_object,
                                                         astunparse.unparse(node.keywords[0].value))

                    elif is_model == True and (node.func.attr == 'predict'):
                        return attr_body
                    else:
                        res = walking(node.func)
                        if(type(node.func) != "Name"):
                            if type(res).__name__ == 'list':
                                for i in res:
                                    return_list.insert(0, i)
                            else:
                                return_list.insert(0, res)
                elif type(node.func.value).__name__ == 'Call':
                    args_name1 = []  # [(1,int),(2,str),,...]
                    keywords_name1 = {}  # [("axis": (0, int)),.....]
                    for arg_node in node.func.value.args:
                        one_arg_result = walking(arg_node)
                        if (type(arg_node).__name__ != "Name"):
                            args_name1.append((astunparse.unparse(arg_node), type(arg_node).__name__))
                        else:
                            is_in = False
                            for j in walk_logs["local_values"]:
                                if (j[0] == walking(arg_node)):
                                    args_name1.append((astunparse.unparse(arg_node), j[1]))
                                    is_in = True
                                    break
                            if is_in == False:
                                args_name1.append((astunparse.unparse(arg_node), 'unknown variable'))
                        if type(one_arg_result).__name__ == 'str':
                            return_list.append(one_arg_result)
                        elif type(one_arg_result).__name__ == 'List':
                            for item in one_arg_result:
                                return_list.append(item)


                    for keyword_node in node.func.value.keywords:
                        key = keyword_node.arg
                        one_arg_result = walking(keyword_node.value)
                        if (type(keyword_node).__name__ != "Name"):
                            keywords_name1[key] = (astunparse.unparse(keyword_node), type(keyword_node).__name__)
                        else:
                            for j in walk_logs["local_values"]:
                                if (j[0] == walking(keyword_node)):
                                    keywords_name1[key] = (astunparse.unparse(keyword_node), j[1])
                                else:
                                    keywords_name1[key] = (astunparse.unparse(keyword_node), 'unknown variable')
                        if type(one_arg_result).__name__ == 'str':
                            return_list.append(one_arg_result)
                        elif type(one_arg_result).__name__ == 'List':
                            for item in one_arg_result:
                                return_list.append(item)


                    func_name = ""
                    if(type(node.func.value.func).__name__ == 'Name'):
                        func_name = walking(node.func.value)
                    is_estiminator = False
                    estiminator_name = ""
                    for i in eval(CONFIG.get('operators', 'operations')).keys():
                        if eval(CONFIG.get('operators', 'operations'))[i]["call_type"] == 3 and func_name == i:
                            is_estiminator = True
                            estiminator_name = func_name
                    if is_estiminator == True and (node.func.attr == 'fit_transform' or node.func.attr == 'transform'):
                        if(len(node.args) != 0):
                            dataset_name = walking(node.args[0])
                        else:
                            dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[estiminator_name]['params'][0]]
                        is_in = False
                        for i in range(0, len(walk_logs["data_values"])):
                            if (dataset_name == walk_logs["data_values"][i]):
                                data_object = walk_logs["data_types"][i]
                                is_in = True
                            else:
                                if len(node.args) != 0:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.args[0]):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                                else:
                                    if walk_logs["data_values"][i] in astunparse.unparse(node.keywords):
                                        data_object = walk_logs["data_types"][i]
                                        is_in = True
                        # if is_in == False:
                        #     data_object = "unknown"
                        #     walk_logs["data_values"].append(dataset_name)
                        #     walk_logs["data_types"].append(data_object)
                        if is_in == True:
                            physic_operation = eval(CONFIG.get('operators', "operations"))[estiminator_name]["physic_operations"][0]
                            if len(node.args) != 0:
                                create_new_sequence_node(estiminator_name, physic_operation, args_name1, keywords_name1, data_object, astunparse.unparse(node.args[0]))
                            else:
                                ind = 0
                                for i in range(0, len(node.keywords)):
                                    key = node.keywords[i].arg
                                    if key == eval(CONFIG.get('operators', 'operations'))[node.func.attr]["params"][0]:
                                        ind = i
                                        break
                                create_new_sequence_node(estiminator_name, physic_operation, args_name1, keywords_name1,
                                                         data_object, astunparse.unparse(node.keywords[ind]))
                    else:
                        res = walking(node.func)
                        if (type(node.func) != "Name"):
                            if type(res).__name__ == 'list':
                                for i in res:
                                    return_list.insert(0, i)
                            else:
                                return_list.insert(0, res)
                else:
                    res = walking(node.func)
                    if (type(node.func) != "Name"):
                        if type(res).__name__ == 'list':
                            for i in res:
                                return_list.insert(0, i)
                        else:
                            return_list.insert(0, res)

            elif type(node.func).__name__ == 'Name':
                func_body = walking(node.func)
                # *******判断type2*************
                if check_operators(func_body, 2):
                    print(func_body)
                    physic_operation = eval(CONFIG.get('operators', "operations"))[func_body]["physic_operations"][0]
                    if(len(node.args) != 0):
                        dataset_name = walking(node.args[0])
                    else:
                        if eval(CONFIG.get('operators', 'operations'))[func_body]['params'][0] in keywords_name:
                            dataset_name = keywords_name[eval(CONFIG.get('operators', 'operations'))[func_body]['params'][0]]
                        else:
                            return
                    is_in = False
                    for i in range(0, len(walk_logs["data_values"])):
                        if (dataset_name == walk_logs["data_values"][i]):
                            data_object = walk_logs["data_types"][i]
                            is_in = True
                        else:
                            if len(node.args) != 0:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.args[0]):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                            else:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.keywords):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True

                    # if is_in == False:
                    #     data_object = "unknown"
                    #     walk_logs["data_values"].append(dataset_name)
                    #     walk_logs["data_types"].append(data_object)
                    if is_in == True:
                        if len(node.args) != 0:
                            create_new_sequence_node(func_body, physic_operation, args_name, keywords_name, data_object, astunparse.unparse(node.args[0]))
                        else:
                            ind = 0
                            for i in range(0, len(node.keywords)):
                                key = node.keywords[i].arg
                                if key == eval(CONFIG.get('operators', 'operations'))[node.func.attr]["params"][0]:
                                    ind = i
                                    break
                            create_new_sequence_node(func_body, physic_operation, args_name, keywords_name, data_object,
                                                     astunparse.unparse(node.keywords[ind]))

                # *******判断type3*************
                elif check_operators(func_body, 3):
                    return [func_body, args_name, keywords_name]
                elif check_model(func_body):
                    return func_body
                elif check_operators(func_body, 4):  # 如果命中4操作 np.clip
                    # print(astunparse.unparse(node))
                    # print(node.args)
                    physic_operation = eval(CONFIG.get('operators', "operations"))[func_body]["physic_operations"][0]
                    if (len(node.args) != 0):
                        dataset_name = walking(node.args[0])
                    else:
                        dataset_name = keywords_name[
                            eval(CONFIG.get('operators', 'operations'))[func_body]['params'][0]]
                    is_in = False
                    for i in range(0, len(walk_logs["data_values"])):
                        if (dataset_name == walk_logs["data_values"][i]):
                            data_object = walk_logs["data_types"][i]
                            is_in = True
                        else:
                            if len(node.args) != 0:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.args[0]):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                            else:
                                if walk_logs["data_values"][i] in astunparse.unparse(node.keywords):
                                    data_object = walk_logs["data_types"][i]
                                    is_in = True
                    # if is_in == False:
                    #     data_object = "unknown"
                    #     walk_logs["data_values"].append(dataset_name)
                    #     walk_logs["data_types"].append(data_object)
                    if is_in == True:
                        if len(node.args) != 0:
                            create_new_sequence_node(func_body, physic_operation, args_name, keywords_name, data_object,
                                                     astunparse.unparse(node.args[0]))
                        else:
                            ind = 0
                            for i in range(0, len(node.keywords)):
                                key = node.keywords[i].arg
                                if key == eval(CONFIG.get('operators', 'operations'))[node.func.attr]["params"][0]:
                                    ind = i
                                    break
                            create_new_sequence_node(func_body, physic_operation, args_name, keywords_name, data_object,
                                                     astunparse.unparse(node.keywords[ind]))

        return return_list

    #-------------------------基础数据类型----------------------------
    elif type(node).__name__ == 'Name': #返回str，变量名
        return node.id
    elif type(node).__name__ == "Subscript": #返回str，主体名，这里先不考虑slice值，之后可以加上
        data_object = walking(node.value) #主体有可能是Call，Attribute，Name，返回这三个当主体对象
        return data_object
    elif type(node).__name__ == 'List': #返回list，变量/主体名
        lis = []
        if len(node.elts) > 0:
            for lis_node in node.elts:
                result = walking(lis_node) #lis_node可能是Call，Attribute，Name，Subscript, 返回这四个主体对象
                lis.append(result)
        return lis
    elif type(node).__name__ == 'Tuple':
        lis = []
        if len(node.elts) > 0:
            for lis_node in node.elts:
                result = walking(lis_node)  # Tuple元素可能是Call，Attribute，Name，Subscript, 返回这四个主体对象
                lis.append(result)
        return lis
    elif type(node).__name__ == 'Dict':
        lis = []
        if len(node.values) > 0:
            for lis_node in node.values:
                result = walking(lis_node)  # dict的value元素可能是Call，Attribute，Name，Subscript, 返回这四个主体对象
                lis.append(result)
        return lis
    elif type(node).__name__ == 'Attribute':
        return walking(node.value)
    elif type(node).__name__ == 'Str':
        return node.s
    elif type(node).__name__ == 'Num':
        return node.n

    #---------------------其他不重要的节点——————————————————————————————————————
    elif type(node).__name__ == 'ListComp' or type(node).__name__ == 'setComp' or type(node).__name__ == 'GeneratorExp':
        comprehension_node_list = node.generators
        for comprehension_node in comprehension_node_list:
            walking(comprehension_node.iter)
        walking(node.elt)
        return
    elif type(node).__name__ == 'DictComp':
        walking(node.value)
        return
    elif type(node).__name__ == 'BinOp':
        lef = walking(node.left)
        rig = walking(node.right)
        res = []
        if lef in walk_logs['data_values']:
            res.append(lef)
        if rig in walk_logs['data_values']:
            res.append(rig)
        return res
    elif type(node).__name__ == 'Return':
        # walking(node.value)
        return
    else:
        return

class CodeVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Import(self, node):
        for alias in node.names:
            check_img(alias.asname)
            walk_logs["import_lis"].add(alias.name)
            walk_logs["import_lis"].add(alias.asname)
            # if alias.name not in package_dict.keys():
            #     package_dict[alias.name] = 0
            # if alias.asname not in package_dict.keys():
            #     package_dict[alias.asname] = 0
            # package_dict[alias.name] += 1
            # package_dict[alias.asname] += 1
            if (alias.name == 'pandas'):
                walk_logs["pandas_alias"] = alias.asname
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        walk_logs["import_lis"].add(node.module)
        check_img(node.module)
        # if node.module not in package_dict.keys():
        #     package_dict[node.module] = 0
        # package_dict[node.module] += 1
        for alias in node.names:
            check_img(alias.name)
            check_img(alias.asname)
            walk_logs["import_lis"].add(alias.name)
            walk_logs["import_lis"].add(alias.asname)
            # if alias.name not in package_dict.keys():
            #     package_dict[alias.name] = 0
            # if alias.asname not in package_dict.keys():
            #     package_dict[alias.asname] = 0
            # package_dict[alias.name] += 1
            # package_dict[alias.asname] += 1

        self.generic_visit(node)

    def visit_Assign(self, node):
        global is_Assign
        is_Assign = 1
        walking(node)
        # ast.NodeVisitor.generic_visit(self, node)
        is_Assign = 0

    def visit_Expr(self, node):
        # print('in Expr')
        walking(node)
        # ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        global is_funcdef
        is_funcdef = 1
        function_def(node)
        is_funcdef = 0
        # ast.NodeVisitor.generic_visit(self, node)

    def visit_For(self, node):
        global is_for
        is_for = 1
        walking(node)
        is_for = 0
        # self.generic_visit(node)

    def visit_If(self, node):
        global is_if
        is_if = 1
        walking(node)
        is_if = 0
        # self.generic_visit(node)

    def visit_While(self, node):
        global is_While
        is_While = 1
        walking(node)
        is_While = 0

def single_running(notebook_id, notebook_url, notebook_path, is_save=False, save_walk_logs_path=""):
    """
    :param notebook_id: 数据库里notebook的id
    :param notebook_title: 数据库里notebook的title，根路径+title = 文件路径
    :param notebook_path: 存储notebook的根路径
    :return: 返回当前notebook的walk_logs
    """
    global condition_switch, walk_logs
    notebook_title = notebook_url.split('/')[-1]
    reflush_walk_logs_and_condition_switch(notebook_id, notebook_title)
    walk_logs["notebook_id"] = int(notebook_id)
    walk_logs["notebook_title"] = notebook_title
    try:
        notebook_title_list = notebook_title.split(' ')
        new_title = ''
        for i in range(0, len(notebook_title_list)):
            new_title += notebook_title_list[i]
            if i != len(notebook_title_list)-1:
                new_title += '-'
        
        code_txt = get_code_txt(notebook_path + '/' + str(notebook_id) + '.ipynb')
    except Exception as e:
        print('str(Exception):\t', str(e))
        print("\033[0;31;40m\tread error\033[0m")
        return "ERROR"
    visitor = CodeVisitor()
    try:
        r_node = ast.parse(code_txt)
    except Exception as e:
        print('str(Exception):\t', str(e))
        print("\033[0;31;40m\tparse error\033[0m")
        return "ERROR"


    visitor.visit(r_node)
    count = 0
    seq = []
    for i in walk_logs["operator_sequence"]:
        count+=1
        seq.append((i["operator_name"], i['data_object']))
    print("seq:", seq)

    if is_save == True:
        try:
            add_sequence_from_walk_logs(walk_logs, save_walk_logs_path)
        except:
            print("\033[0;31;40m\tadd database fail\033[0m")
    return walk_logs

def batch_running(notebook_path, save_walk_logs_path,ip):
    """
    :param notebook_path: 存储notebook的根路径
    :param save_walk_logs_path: 用来存储walk_logs
    :return: 无
    """
    notebook_info_list = get_batch_notebook_info(ip)
    for notebook_info in notebook_info_list:
        notebook_id = notebook_info[0]
        notebook_title = notebook_info[1]
        notebook_url = notebook_info[2]

        print("\033[0;34;40m\tid:" + str(notebook_id) + '\ttitle:' + notebook_title + "\033[0m")
        try:
            this_walk_logs = single_running(notebook_id, notebook_url, notebook_path)
        except:
            print("\033[0;31;40m\tsingle running fail\033[0m")
        if this_walk_logs == 'ERROR':
            continue
        # try:
        if True:
            result = add_sequence_from_walk_logs(this_walk_logs, save_walk_logs_path)
            if result == "ERROR":
                print(("\033[0;31;40m\tadd database fail\033[0m"))
        # except:
        #     print("\033[0;31;40m\tadd database fail\033[0m")


def check_table_data(notebook_path, ip):
    notebook_info_list = get_batch_no_seq_notebook_info(ip)
    count = 0
    n_c = 0
    # print(notebook_info_list)
    for notebook_info in notebook_info_list:
        notebook_id = notebook_info[0]
        code_txt = get_code_txt(notebook_path + '/' + str(notebook_id) + '.ipynb')
        if 'read_csv' in code_txt or 'read_pickle'  in code_txt  or 'read_table'  in code_txt or 'read_fwf'  in code_txt  or \
                'read_clipboard'  in code_txt or 'read_excel'  in code_txt or 'ExcelFile.parse'  in code_txt or 'ExcelWriter'  in code_txt or \
                'read_json'  in code_txt or 'json_normalize'  in code_txt or 'build_table_schema'  in code_txt or 'read_html'  in code_txt or \
                'read_hdf'  in code_txt or 'read_feather'  in code_txt or 'read_parquet'  in code_txt or 'read_orc'  in code_txt or \
                'read_sas' in code_txt or 'read_spss'  in code_txt or 'read_sql_table'  in code_txt or 'read_sql_query'  in code_txt or \
                'read_sql'  in code_txt or 'read_gbq'  in code_txt or 'read_stata'  in code_txt:
            n_c += 1
            # print(n_c)
            continue
        else:
            count += 1
            print(count)
            update_db("notebook", "cant_sequence", '4', 'id', '=', notebook_id)

def main(argv):
    opts, args = getopt.getopt(argv, "hr:n:w:i:t:s:p:", ["rtype=", "npath=","wpath=","nid=","ntitle=", "save=", "ip="])
    running_type = 'batch'
    notebook_path = '../notebook'
    save_walk_logs_path = '../walklogs'
    notebook_id = 0
    notebook_title = ""
    ip = "10.77.70.123"
    is_save = False
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-h':
            print("if you want running batches:")
            print("notebook2sequence.py -r batch -n <npath> -w <wpath>")
            print("else if you want running single:")
            print("notebook2sequence.py -r single -n <npath> -w <wpath> -i <nid> -t <ntitle> -s <save>")
            print('-r <type>: running_type, single or batch')
            print('-n <npath>: notebook_path')
            print('-w <wpath>: save_walk_logs_path')
            print('-i <nid>: notebook_id')
            print('-t <ntitle>: notebook_title')
            print('-s <save>: is_save')
            print('-p <ip>: ip')
            sys.exit()
        elif opt in ("-r", "--rtype"):
            running_type = arg
            if running_type != 'batch' and running_type != 'single':
                print("r must be batch or single")
                sys.exit()
        elif opt in ("-n", "--npath"):
            notebook_path = arg
        elif opt in ("-w", "--wpath"):
            save_walk_logs_path = arg
        elif opt in ("-i", "--nid"):
            notebook_id = int(arg)
        elif opt in ("-t", "--ntitle"):
            notebook_title = arg
        elif opt in ("-p", "--ip"):
            ip = arg
        elif opt in ("-s", "--save"):
            if arg == 'True':
                is_save = True
            elif arg == 'False':
                is_save = False

    print(notebook_path)
    if running_type == 'batch':
        batch_running(notebook_path, save_walk_logs_path,ip)
    elif running_type == 'single':
        single_running(notebook_id, notebook_title, notebook_path, is_save, save_walk_logs_path)

    return
    
if __name__ == "__main__":
   # main(sys.argv[1:])
   check_table_data('../spider/notebook', '39.99.150.216')