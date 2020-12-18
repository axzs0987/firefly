
from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run
from notebook2sequence import single_running
from utils import get_params_code_by_id
from utils import CONFIG
from utils import find_special_notebook
from utils import get_pair
from utils import get_save_pair
from utils import update_db
from utils import add_error
from utils import create_connection
from utils import get_host_ip
import numpy as np
import os
import ast
import astunparse
from modifySequence import get_result_code
from modifySequence import get_operator_code
from modifySequence import running_temp_code
# from add_model import save_dataframe_and_update_reuslt
import re
import traceback


def get_code_txt(path, remove_notes=True):
    cot_txt = ""
    lis = read_ipynb(path)
    # LR matching
    lis = Feeding(lis)
    lis = LR_run(lis)
    for paragraph in lis:
        for item in paragraph['code']:
            if item:
                if (item[0] == '!') or (item[0] == '<' or (item[0] == '%')):
                    continue
            temp = item + '\n'
            cot_txt += temp

    if remove_notes == True:
        code_list = cot_txt.split('\n')
        new_cot_txt = ''
        in_beizhu = False
        for line in code_list:
            index = 0
            if index == len(line):
                continue
            # print(line,'\'\'\'' not in line and '"""' not in line and in_beizhu==False)
            if '\'\'\'' not in line and '"""' not in line and in_beizhu==False:
                # print(line)
                while line[index] != '#' and index < len(line) -1:
                    # print(':-)')
                    # print(index)
                    # print(line[index])
                    if line[index] == '\'':
                        index += 1
                        while line[index] != '\'':
                            index += 1
                    if line[index] == '\"':
                        index += 1
                        while line[index] != '\"':
                            index += 1
                    if index != len(line) -1:
                        index += 1

                if line[index] != '#':
                    index += 1
                temp = line[0:index]
            elif '\'\'\'' in line or '"""' in line:
                if in_beizhu == False:
                    in_beizhu = True
                elif in_beizhu == True:
                    in_beizhu = False
                temp = line
            else:
                temp = line
            # print(temp)
            if temp == '':
                continue
            if temp[-1] != '\n':
                temp += '\n'
            new_cot_txt += temp
        return new_cot_txt
    return cot_txt

def found_dataset(old_path, notebook_id, root_path, origin_code):
    """
    :param old_path:
    :param notebook_id:
    :param root_path:
    :param origin_code:
    :return:
    如果运行时发现路径不对，找到需要替换的路径
    """

    old_root_path = ''
    if '/' not in old_path:
        result = root_path + '/' + old_path
        old_root_path = old_path
    else:
        for index, i in enumerate(old_path.split('/')):
            if index != len(old_path.split('/')) - 1:
                old_root_path = old_root_path + i + '/'
            else:
                if '.' not in i:
                    old_root_path = old_root_path + i
                if '/' == old_root_path[-1]:
                    old_root_path = old_root_path[0:-1]

        result = root_path



    print('old_root_path', old_root_path)

    print("result", result)
    return origin_code.replace(old_root_path, result)


# def get_pairs(path):
#     """
#     :param path:
#     :return:
#     暂时使用的函数，pair数据库建起来之前，使用pair.txt找对应关系
#     """
#     notebook_name_list = []
#     dataset_name_list = []
#     with open(path) as f:
#         for line in f:
#             line_list = line.split(']')
#             notebook_name = line_list[0].split('/')[-1]
#             dataset_name = line_list[1].split('/')[-1][0:-1]
#             notebook_name_list.append(notebook_name)
#             dataset_name_list.append(dataset_name)
#
#     return notebook_name_list, dataset_name_list

def insert_one_line_in_code(origin_code, under_line, target_line, is_same = True, up_down = 'down'):
    """
    :param origin_code: 原本代码
    :param under_line: 目标插入的位置，int，就是第几行（从0开始），string就是在匹配到包含此字符串的行的下面都插入
    :param target_line: 目标插入代码
    :return: 转换后的代码
    """
    # print('?')
    def get_space_num(line):
        count = 0
        for char in line:
            if char == ' ':
                count += 1
            else:
                break
        # print(line[count+1:count+3])
        # if line[count:count+3] == 'if ' and ':' == line[-1] or line[count:count+4] == 'for ' or line[count:count+6] == 'while ' and ':' in line or line[count:count+4] == 'def ' in line and ':' in line:
        #     count += 4
        return count

    def add_space(line, count):
        for i in range(0,count):
            line = ' ' + line
        return line

    def converse_target(underline, targetline):
        return add_space(targetline, get_space_num(underline))

    # 把文本都拆成行list
    code_list = origin_code.split('\n')
    if target_line[-1] == '\n': #如果目标结尾有换行符，删掉
        target_line = target_line[0:-1]

    #如果underline是字符串类型，找到所有的包含关键字都行，在下面插入目标行
    if type(under_line).__name__ == 'str':
        index_list = []
        for index,line in enumerate(code_list):
            if under_line in line:
                index_list.append(index)
            # print(index, line)
        for index in range(0, len(index_list)):
            if is_same == True:
                count = get_space_num(code_list[index_list[index]])
                if index == 0:
                    target = converse_target(code_list[index_list[index]], target_line)
                else:
                    # print('cout:',count)
                    # print(len(code_list[index_list[index]]))
                    ed = 0
                    while index_list[index]+ed < len(code_list):
                        if len(code_list[index_list[index] + ed]) == count:
                            count = get_space_num(code_list[index_list[index] + ed])
                            ed += 1
                        else:
                            break

                    if index_list[index]+ed != len(code_list):
                        if code_list[index_list[index]][count:count+7] == 'except:' or code_list[index_list[index]][count:count+5] == 'else:':
                            target = converse_target(code_list[index_list[index-1]], target_line)
                        else:
                            target = converse_target(code_list[index_list[index+ed]], target_line)
                    else:
                        target = converse_target(code_list[index_list[index - 1]], target_line)
            if up_down == 'down':
                code_list.insert(index_list[index], target)
            if up_down == 'up':
                code_list.insert(index_list[index]-1, target)
            if index != len(index_list)-1:
                for after_index in range(index + 1, len(index_list)):
                    index_list[after_index] += 1

    #如果underline是数字，则在该行都下一行插入目标
    elif type(under_line).__name__ == 'int':
        if is_same == True:
            if under_line == 0:
                target = converse_target(code_list[under_line], target_line)
            else:
                count = get_space_num(code_list[under_line])
                # print('cout:', count)
                # print(len(code_list[under_line]))
                ed = 0
                while under_line+ed < len(code_list):
                    if len(code_list[under_line+ed]) == count:
                        count = get_space_num(code_list[under_line+ed])
                        ed += 1
                    else:
                        break
                if under_line+ed != len(code_list):
                    if code_list[under_line][count:count + 7] == 'except:' or code_list[under_line][count:count + 5] == 'else:':
                        # print('11111111111111')
                        # print(code_list[under_line])
                        # print('target-line:', target_line)
                        target = converse_target(code_list[under_line-1], target_line)
                    else:
                        # print('1222222222')
                        # print(code_list[under_line])
                        # print('target-line:',target_line)
                        target = converse_target(code_list[under_line+ed], target_line)
                else:
                    # print('1222222222')
                    # print(code_list[under_line])
                    # print('target-line:', target_line)
                    target = converse_target(code_list[under_line - 1], target_line)


        code_list.insert(under_line, target)

    # 把list转文本
    result = ''
    for i in code_list:
        result = result + i+'\n'
    return result

def add_package(package_name, package_type, package_alias, origin_code):
    if package_type == 1:
        target_code = "import " + package_name
    elif package_type == 2:
        target_code = "import " + package_name + " as " + package_alias
    elif package_type == 3:
        target_code = "from " + package_name + " import " + package_alias

    return insert_one_line_in_code(origin_code, 0, target_code)

def print_readcsv(origin_code):
    """
    :param origin_code: 原始代码
    :return:
    打印读入的数据，并且找到所有为字符串的列
    """
    origin_code = insert_one_line_in_code(origin_code,0,"now_df = {}")
    origin_code = insert_one_line_in_code(origin_code, 0, "new_count = {}")
    code_list = origin_code.split('\n')
    csv_index = []
    csv_varible = []
    for index, line in enumerate(code_list):
        if "read_csv" in line and '=' in line:
            csv_index.append(index)
            varible = line.split('=')[0].strip()
            csv_varible.append(varible)

    for i in range(0, len(csv_varible)):
        if i==0:
            column_name_code = "column_name=[]"
            origin_code = insert_one_line_in_code(origin_code, csv_index[i], column_name_code)
            for after_index in range(i , len(csv_index)):
                csv_index[after_index] += 1
        target_line = 'origin_data_' + str(i) + '= ' + csv_varible[i]
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 1, target_line)
        target_line = 'print(origin_data_' + str(i) + ')'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 2, target_line)
        target_line = 'for col in ' + csv_varible[i] + ':'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 3, target_line)
        target_line = '    if str(' + csv_varible[i] + '[col].dtype) == "object":'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 4, target_line)
        target_line = '    column_name.append(str(col))'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 5, target_line)
        target_line = 'now_df[str(col)] = []'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 6, target_line)
        target_line = 'new_count[str(col)] = 0'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 7, target_line)
        if i != len(csv_index) - 1:
            for after_index in range(i + 1, len(csv_index)):
                csv_index[after_index] += 4

    code_list = origin_code.split('\n')
    code_list.append("print(column_name)")
    result = ''

    for i in code_list:
        result = result + i + '\n'
    return result,len(csv_varible)


def running(notebook_id, func_def, new_path,count, found=False):
    """
    :param func_def: 需要运行的代码字符串
    :param new_path: 替换路径
    :param count: 第几次运行了
    :return: 返回修改过后或者成功运行的代码
    运行代码
    """
    try:
        cm = compile(func_def, '<string>', 'exec')
    except Exception as e:
        print("compile fail", e)
        return "compile fail"
    print("\033[0;33;40m" + str(count) +"\033[0m")
    can_run = False
    try:
        ns = {}
        exec(cm,ns)
        print("\033[0;32;40msucceed\033[0m")
        can_run = True
        # return 'succeed'
    except Exception as e:
        # traceback.print_exc()
        error_str = str(e)
        new_code = func_def
        foun = 0
        # traceback.print_exc()
        # print("\033[0;31;40merror_str\033[0m", error_str)
        # print("\033[0;31;40merror_str\033[0m", error_str)
        if "[Errno 2] No such file or directory: " in error_str:
            error_path = error_str.replace("[Errno 2] No such file or directory: " , "")
            error_path = error_path[1:-1]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            foun=1
            print('error 1')
            # running(new_code)
        elif "does not exist:" in error_str and '[Errno 2] File ' in error_str:
            error_path = error_str.split(':')[-1].strip()
            error_path = error_path[1:-1]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            # print('new_code:', new_code)
            print('error 2')
            foun=1
        elif "No module named " in error_str and '_tkinter' not in error_str:
            package = error_str.replace("No module named ", "")
            package = package[1:-1]
            # command = ' pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0]
            # os.system(command)
            command = ' pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0] + ' --trusted-host pypi.tuna.tsinghua.edu.cn'
            # command = ' pip install ' + package.split('.')[0]
            os.system(command)
            print('error 3')
        elif  ": No such file or directory" in error_str:
            index1 = error_str.find("'")
            index2 = error_str.find("'", index1+1)
            error_path = error_str[index1+1:index2]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            print('error 4')
        elif "Command '['ls'," in error_str:
            index1 = error_str.find('ls')
            # print(index1)
            el_line = error_str[index1+6:]
            # print(el_line)
            right_index  = el_line.find('\'')
            error_path = el_line[0:right_index]
            # print('error_path:', error_path)
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('new_code:', new_code)
            foun = 1
            print('error 5')
        elif "File b" in error_str:
            index1 = error_str.find("'")
            index2 = error_str.find("'", index1 + 1)
            error_path = error_str[index1 + 1:index2]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            print('error 4')
            foun = 1
            print('error 5')
        elif "'DataFrame' object has no attribute 'ix'" in error_str or "'Series' object has no attribute 'ix'" in error_str:
            new_code = func_def.replace('.ix', '.iloc')
            print('error 6')
        elif "'DataFrame' object has no attribute 'sort'" in error_str:
            new_code = func_def.replace('.sort(', '.sort_values(')
            print('error 7')
        else:
            # print("?")
            # traceback.print_exc()
            print("\033[0;31;40merror_str\033[0m", error_str)
            print('error 8')

            add_error(notebook_id, error_str)
            return "False"
        if count < 7:
            # print(new_code)
            if foun ==1:
                found = True
            res = running(notebook_id, new_code, new_path, count + 1,found)
            if res == 'compile fail' or res== 'False':
                return res
            # return res
        else:
            add_error(notebook_id, error_str)
            print('error 9')
            return "False"
    return func_def

def add_changed_result(notebook_id, origin_code, walk_logs_path = "../walklogs"):
    func_def = ''
    func_def += "add_changed_result = []\n"
    func_def += "def insert_result(model_type, content, code, metric_type, model_candicate_list):\n"
    func_def += "    notebook_id = " + str(notebook_id) + '\n'
    # func_def += "    print('model_candicate_list:',model_candicate_list)\n"
    func_def += "    new_list = []\n"
    func_def += "    ope_dic = ['SimpleImputer','Imputer','OneHotEncoder','LabelEncoder','LabelBinarizer','StandardScaler','MinMaxScaler','RobustScaler','PCA']\n"
    func_def += "    for item in list(model_candicate_list):\n"
    func_def += "        if item != 'ndarray' and item != 'Series' and item != 'DataFrame' and item != 'dict' and item != 'PCA' and item != 'List' and item not in ope_dic:\n"
    func_def += "            new_list.append(item)\n"
    func_def += "    print(new_list)\n"
    func_def += "    if len(new_list) == 0:\n"
    func_def += "        add_changed_result.append((notebook_id, model_type , content, code, metric_type))\n"
    func_def += "    else:\n"
    func_def += "        if model_type != new_list[-1]:\n"
    func_def += "            model_type = new_list[-1]\n"
    func_def += "        if model_type != 'unknown':\n"
    func_def += "            add_changed_result.append((notebook_id,model_type, content, code, metric_type))\n"
    func_def += "        else:\n"
    func_def += "            add_changed_result.append((notebook_id, new_list[-1], content, code, metric_type))\n"

    try:
        this_walk_logs = np.load(walk_logs_path + '/' + str(notebook_id) + '.npy',allow_pickle=True).item()
        model_pred = this_walk_logs['models_pred']
    except:
        model_pred = []

    origin_code = func_def + origin_code

    model_dic  = eval(CONFIG.get('models', 'model_dic'))
    model_result_log = {}

    line = 0
    metirc_dic = eval(CONFIG.get('metrics', 'metrics'))
    add = False
    ins_num = 0
    origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
    add_running = False
    while(line < len(origin_code.split('\n'))):
        code_list = origin_code.split('\n')
        code = code_list[line]
        # print(code)
        m_type = -1
        now_key = ''
        now_len = -1
        now_name = ''
        for key in metirc_dic.keys():
            if key in code:
                m_type = metirc_dic[key]['type']
                now_key = key
                now_len = metirc_dic[key]['len']
                now_name = metirc_dic[key]['name']
                break
        if m_type == 1:
            index = code.find(now_key)
            head = index - 1
            # print("index:",index)
            while (code[head].isalpha() or code[head] == '_' \
                   or code[head] == ']' \
                   or code[head] == ')' \
                   or code[head] == '.' \
                   or code[head] == '\'' or code[head] == '\"' \
                   or code[head].isalnum()) \
                    and head != -1:
                if code[head] == ']':
                    while code[head] != '[':
                        head -= 1
                if code[head] == ')':
                    while code[head] != '(':
                        # print(code[head])
                        head -= 1
                head -= 1
            left_index = index + now_len
            left_num = 1
            right_index = 0
            for ind in range(left_index + 1, len(code)):
                if code[ind] == '(':
                    left_num += 1
                elif code[ind] == ')':
                    left_num -= 1
                if left_num == 0:
                    right_index = ind
                    break
            model_id = -1
            for item in model_pred:
                if model_pred[item] in code:
                    model_id = model_dic[item]
                elif item in code:
                    model_id = model_dic[item]

            temp_code = code
            temp_code = temp_code.replace(' ', '')
            temp_code = temp_code.replace('\t', '')
            temp_code = temp_code.replace('\'', '\\\'')
            temp_code = temp_code.replace('"', '\\"')

            ctxt = code[head + 1:right_index + 1]
            add_model_result = get_model_from_code(ctxt, origin_code, line, ins_num)
            origin_code = add_model_result[1]
            add_line = add_model_result[2]
            # line += 1
            line += add_line

            add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
            origin_code = add_model_result_1[1]
            add_line_1 = add_model_result_1[2]
            line += add_line_1

            need2print = "insert_result('" +add_model_result[0] + "'," + code[head + 1:right_index + 1] + ',"' + temp_code + '", "'+ now_name +'", mdtypes_'+ str(ins_num) +')'

            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            add_running = True
            ins_num += 1
            line += 1

            origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
            line += 1
            add = True
        elif m_type == 2:
            index = code.find(now_key)
            head = index - 2

            while (code[head].isalpha() or code[head] == '_' \
                   or code[head] == ']' \
                   or code[head] == ')' \
                   or code[head] == '.' \
                   or code[head] == '\'' or code[head] == '\"' \
                   or code[head].isalnum()) \
                    and head != -1:
                if code[head] == ']':
                    while code[head] != '[':
                        head -= 1
                if code[head] == ')':
                    while code[head] != '(':
                        # print(code[head])
                        head -= 1
                head -= 1

            right_index = index + now_len

            model_id = -1
            for item in model_pred:
                if model_pred[item] in code:
                    model_id = model_dic[item]
                elif item in code:
                    model_id = model_dic[item]

            temp_code = code
            temp_code = temp_code.replace(' ','')
            temp_code = temp_code.replace('\t', '')
            temp_code = temp_code.replace('\'', '\\\'')
            temp_code = temp_code.replace('"', '\\"')

            ctxt = code[head + 1:right_index + 1]
            # print('code:',code)
            add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
            origin_code = add_model_result[1]
            add_line = add_model_result[2]
            # line += 1
            line += add_line

            add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
            origin_code = add_model_result_1[1]
            add_line_1 = add_model_result_1[2]
            line += add_line_1

            need2print = "insert_result('" + add_model_result[0] + "'," + code[head+1:right_index + 1] +',"' + temp_code + '", "' + now_name +'", mdtypes_'+ str(ins_num) +')'

            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            add_running = True
            ins_num += 1
            line += 1
            origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
            line += 1
            add = True
        elif m_type == 3:
            now_len = len(now_key)-1
            add = True
            index = code.find(now_key)
            # print('code:',code)
            # print('index:',index)
            if index == 0:
                head = index
                left_index = index  + now_len
                left_num = 1

                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break

                model_id = -1
                for item in model_pred:
                    if model_pred[item] in code:
                        model_id = model_dic[item]
                    elif item in code:
                        model_id = model_dic[item]


                temp_code = code
                temp_code = temp_code.replace(' ','')
                temp_code = temp_code.replace('\t', '')
                temp_code = temp_code.replace('\'', '\\\'')
                temp_code = temp_code.replace('"', '\\"')

                ctxt = code[head:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # line += 1
                line += add_line

                add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
                origin_code = add_model_result_1[1]
                add_line_1 = add_model_result_1[2]
                line += add_line_1

                need2print = "insert_result('" + add_model_result[0] + "'," + code[head:right_index + 1] + ',"' + temp_code + '", "' + now_name +'", mdtypes_'+ str(ins_num) +')'
                # print('vvvv')
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                add_running = True
                ins_num += 1
                line += 1
                origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
                line += 1

            elif code[index-1] != '.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index
                left_index = index + now_len
                left_num = 1

                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break

                model_id = -1
                for item in model_pred:
                    if model_pred[item] in code:
                        model_id = model_dic[item]
                    elif item in code:
                        model_id = model_dic[item]

                temp_code = code
                temp_code = temp_code.replace(' ', '')
                temp_code = temp_code.replace('\t', '')
                temp_code = temp_code.replace('\'', '\\\'')
                temp_code = temp_code.replace('"', '\\"')

                ctxt = code[head:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
                need2print = "insert_result('" + add_model_result[0] + "'," + code[
                                                                      head:right_index + 1] + ',"' + temp_code + '", "'+ now_name +'", mdtypes_'+ str(ins_num) +')'
                origin_code = add_model_result[1]
                add_line = add_model_result[2]

                # line += 1
                # print(add_line)
                # print('zzzzzz')
                line += add_line

                add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
                origin_code = add_model_result_1[1]
                add_line_1 = add_model_result_1[2]
                line += add_line_1

                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                add_running = True
                ins_num += 1
                line += 1
                origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
                line += 1


            else:
                head = index - 2
                while (code[head].isalpha() or code[head] == '_' \
                        or code[head] == ']' \
                        or code[head] == ')' \
                        or code[head] == '.' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum())\
                        and head != -1:
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    if code[head] == ')':
                        while code[head] != '(':
                            # print(code[head])
                            head -= 1
                    head -= 1
                left_index = index + now_len
                left_num = 1

                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break

                model_id = -1
                for item in model_pred:
                    if model_pred[item] in code:
                        model_id = model_dic[item]
                    elif item in code:
                        model_id = model_dic[item]

                temp_code = code
                temp_code = temp_code.replace(' ', '')
                temp_code = temp_code.replace('\t', '')
                temp_code = temp_code.replace('\'', '\\\'')
                temp_code = temp_code.replace('"', '\\"')

                ctxt = code[head+1:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
                need2print = "insert_result('" + add_model_result[0] + "'," + code[head+1:right_index + 1] + ',"' + temp_code + '", "' + now_name +'", mdtypes_'+ str(ins_num) +')'
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # line += 1
                line += add_line
                # print('xxsxxs')
                add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
                origin_code = add_model_result_1[1]
                add_line_1 = add_model_result_1[2]
                line += add_line_1

                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                add_running = True
                ins_num += 1
                line += 1
                origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
                line += 1



        line += 1
    origin_code += '\n'
    origin_code += 'print("add_changed_result",add_changed_result)\n'
    origin_code += 'np.save("./temp_result.npy", add_changed_result)\n'
    # origin_code += 'print(add_changed_result)\n'
    # print(origin_code)
    return origin_code, add,add_running


def add_variable_code(origin_code, variable_list,notebook_name, save_root="../strcol"):
    function_def = \
        """
def insert_param_db():

def save_middle_df():


def compare_dataframe(new_df, column_list, save_path, origin_data_0=[]):
    global new_count,now_df
    import pandas as pd 
    
    def can_save(series1, series2):
        print(series1.shape)
        if len(series1) != len(series2):
            print(len(series1),len(series2))
            return 0
        if (series1.index != series2.index).any():
            print("index not same")
            return -1
        if not (series1.values == series2.values).any():
            print("not same")
            return 1
        print("same")
        return -1
        
    print('type new_df', type(new_df).__name__)
    if type(new_df).__name__ != 'DataFrame' and type(new_df).__name__ != 'Series':
        return

    for column in column_list:
        if type(now_df[column]).__name__ == 'list':
            if str(new_df[column].dtype) != 'object':
                continue
            if os.path.exists(save_path) == False:
                os.mkdir(save_path)
            print("path:",save_path + '/' + column)
            col_path = column
            if '/' in column :
                col_path = column.replace('/','_')
            if os.path.exists(save_path + '/' + col_path) == False:
                os.mkdir(save_path + '/' + col_path)
            new_df[column].to_csv(save_path + '/' + col_path + '/' + str(new_count[column]) + '.csv',encoding='gbk')
            now_df[column] = new_df[column]
            new_count[column] += 1
        elif type(new_df).__name__ == 'Series':
            if column != new_df.name:
                continue
            print('series_name:', new_df.name)
            temp_type = new_df.dtype
                
            if str(temp_type) != 'object':
                continue
            if str(now_df[new_df.name].dtype) != 'object':
                continue
                    
                
            resu = can_save(new_df, now_df[new_df.name])
            print("series:",resu)
            if resu != -1:
                print(new_count[column], resu)
                col_path = new_df.name
                if '/' in new_df.name :
                    col_path = column.replace('/','_')
                if os.path.exists(save_path + '/' + col_path) == False:
                    os.mkdir(save_path + '/' + col_path)
                
                new_df.to_csv(save_path + '/' + col_path + '/' + str(new_count[new_df.name]) + '.csv')
                new_count[new_df.name] += 1
                if resu == 1:
                    pair = pd.DataFrame({'before':now_df[column], 'after': new_df})
                    if os.path.exists(save_path + '/' + col_path + '/pair') == False:
                        os.mkdir(save_path + '/' + col_path + '/pair')
                    pair.to_csv(save_path + '/' + col_path + '/pair/' + str(new_count[new_df.name]) + '.csv')
                now_df[column] = new_df   
            if column in origin_data_0:
                resu = can_save(new_df, origin_data_0[column])
                if resu == 1:
                    pair = pd.DataFrame({'before':origin_data_0[column], 'after': new_df})
                    if os.path.exists(save_path + '/' + col_path + '/pair') == False:
                        os.mkdir(save_path + '/' + col_path + '/pair')
                    pair.to_csv(save_path + '/' + col_path + '/pair/origin_' + str(new_count[column]) + '.csv')
                    now_df[column] = new_df 
        else:
            if column not in new_df.columns.values:
                continue
            temp_type = new_df[column].dtype
            
            if str(temp_type) != 'object':
                continue
            if str(now_df[column].dtype) != 'object':
                continue
                    
                
            resu = can_save(new_df[column], now_df[column])
            print("df:",resu)
            if resu != -1:
                print(new_count[column], resu)
                col_path = column
                if '/' in column :
                    col_path = column.replace('/','_')
                if os.path.exists(save_path + '/' + col_path) == False:
                    os.mkdir(save_path + '/' + col_path)
                
                new_df[column].to_csv(save_path + '/' + col_path + '/' + str(new_count[column]) + '.csv')
                new_count[column] += 1
                
                print(now_df.keys())
                
                if resu == 1:
                    pair = pd.DataFrame({'before':now_df[column], 'after': new_df[column]})
                    if os.path.exists(save_path + '/' + col_path + '/pair') == False:
                        os.mkdir(save_path + '/' + col_path + '/pair')
                    pair.to_csv(save_path + '/' + col_path + '/pair/' + str(new_count[column]) + '.csv')
                        
                now_df[column] = new_df[column]
            if column in origin_data_0:
                resu = can_save(new_df[column], origin_data_0[column])
                if resu == 1:
                    pair = pd.DataFrame({'before':origin_data_0[column], 'after': new_df[column]})
                    if os.path.exists(save_path + '/' + col_path + '/pair') == False:
                        os.mkdir(save_path + '/' + col_path + '/pair')
                    pair.to_csv(save_path + '/' + col_path + '/pair/origin_' + str(new_count[column]) + '.csv')
                    now_df[column] = new_df[column]
        """
    cod_li = origin_code.split('\n')
    head = 0
    for ind, i in enumerate(cod_li):
        if i == 'column_name=[]':
            head = ind
            break
    origin_code = insert_one_line_in_code(origin_code, head+1, function_def)
    # print("weewrwL:", origin_code)
    code_list = origin_code.split('\n')
    add_num = 0
    is_add_column = False
    for line_num in range(0, len(code_list)):
        varible_li = code_list[line_num].split('=')
        if len(varible_li) == 1:
            continue
        for variable in variable_list:
            if variable == varible_li[0].strip():
                if 'read_csv' in code_list[line_num] and is_add_column == False:
                    origin_code = insert_one_line_in_code(origin_code, line_num + 1 + add_num + 8, target_line="compare_dataframe("+ variable + ", column_name, '" + save_root +'/' + notebook_name + "')")
                    is_add_column = True
                elif 'read_csv' in code_list[line_num] and is_add_column == True:
                    origin_code = insert_one_line_in_code(origin_code, line_num + 1 + add_num + 7, target_line="compare_dataframe("+ variable + ", column_name, '" + save_root +'/' + notebook_name + "')")
                elif 'origin_data' in variable:
                    origin_code = insert_one_line_in_code(origin_code, line_num + 1 + add_num + 5, target_line="compare_dataframe("+ variable + ", column_name, '" + save_root +'/' + notebook_name + "')")
                else:
                    origin_code = insert_one_line_in_code(origin_code,line_num + 1 + add_num,target_line="compare_dataframe("+ variable + ", column_name, '" + save_root +'/' + notebook_name + "')")
                add_num += 1
                break
    print("weewrwLssssssssss:", origin_code)

    return origin_code

def add_params_miresult(notebook_id, origin_code, save_root_path="../midresult"):
    func_def = ''
    func_def += "from utils import update_params_value\n"
    func_def += "import pandas as pd\n"
    func_def += "import os\n"
    func_def += "def insert_params(rank, param_num, content):\n"
    func_def += "    notebook_id = " + str(notebook_id) + '\n'
    func_def += "    update_params_value(notebook_id, rank, param_num, content)\n"
    func_def += "def insert_miresult(rank, content, root_path):\n"
    func_def += "    notebook_id = " + str(notebook_id) + '\n'
    func_def += "    if os.path.exists(root_path) == False:\n"
    func_def += "        os.mkdir(root_path)\n"
    func_def += "    if os.path.exists(root_path + '/' + str(notebook_id)) == False:\n"
    func_def += "        os.mkdir(root_path + '/' + str(notebook_id))\n"
    func_def += "    if type(content).__name__ == 'numpy.ndarray':\n"
    func_def += "        np.save(root_path + '/' + str(notebook_id) + '/' + str(rank) + '.csv', content)\n"
    func_def += "    elif type(content).__name__ == 'DataFrame' or type(content).__name__ == 'Series':\n"
    func_def += "        content.to_csv(root_path + '/' + str(notebook_id) + '/' + str(rank) + '.csv')\n"
    param_code_list = get_params_code_by_id(notebook_id)
    line = 0
    origin_code = func_def + origin_code
    while (line < len(origin_code.split('\n'))):
        code_list = origin_code.split('\n')
        if code_list[line] == None:
            line+=1
            continue
        for index, i in enumerate(param_code_list):
            # print(i['name'] + '(' + i['p1'])
            item = i['p1']
            co = 1
            while item == None:
                co += 1
                if co == 8:
                    break
                item = i['p'+str(co)]
            if item == None:
                item = ''

            # print(i)
            if i['name'] + '(' + item.replace(' ','') in code_list[line].replace(' ','') and i['data_object_value'] in code_list:
                for j in i.keys():
                    if 'p' not in j:
                        continue
                    number = j[-1]
                    if i[j] == None:
                        continue
                    con = i[j]
                    if '=' in i[j]:
                        con = i[j].split('=')[-1].strip()
                    need2print = 'insert_params(' + str(i['rank']) + ',' + str(number) + ',' + con + ')'

                    line += 1
                    origin_code = insert_one_line_in_code(origin_code, line, need2print)

                need2print = 'insert_miresult(' +  str(i['rank']) + ',' +  i['data_object_value'].split('.')[0] + ',"' + save_root_path +'")'
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                break
        line += 1
    return origin_code

def add_result(notebook_id, origin_code, walk_logs_path = "../walklogs"):
    func_def = ''
    func_def += "from utils import add_result\n"
    func_def += "def insert_result(model_type, content, code, metric_type, model_candicate_list):\n"
    func_def += "    notebook_id = " + str(notebook_id) + '\n'
    func_def += "    print('model_candicate_list:',model_candicate_list)\n"
    func_def += "    new_list = []\n"
    func_def += "    ope_dic = ['SimpleImputer','Imputer','OneHotEncoder','LabelEncoder','LabelBinarizer','StandardScaler','MinMaxScaler','RobustScaler','PCA']\n"
    func_def += "    for item in list(model_candicate_list):\n"
    func_def += "        if item != 'ndarray' and item != 'Series' and item != 'DataFrame' and item != 'dict' and item != 'PCA' and item != 'List' and item not in ope_dic:\n"
    func_def += "            new_list.append(item)\n"
    func_def += "    if len(new_list) == 0:\n"
    func_def += "        add_result(notebook_id, model_type , str(content), code, metric_type)\n"
    func_def += "    else:\n"
    func_def += "        if model_type != new_list[-1]:\n"
    func_def += "            model_type = new_list[-1]\n"
    func_def += "        if model_type != 'unknown':\n"
    func_def += "            add_result(notebook_id, model_type, str(content), code, metric_type)\n"
    func_def += "        else:\n"
    func_def += "            add_result(notebook_id, new_list[-1], str(content), code, metric_type)\n"

    try:
        this_walk_logs = np.load(walk_logs_path + '/' + str(notebook_id) + '.npy',allow_pickle=True).item()
        model_pred = this_walk_logs['models_pred']
    except:
        model_pred = []

    origin_code = func_def + origin_code

    model_dic  = eval(CONFIG.get('models', 'model_dic'))
    model_result_log = {}

    line = 0
    metirc_dic = eval(CONFIG.get('metrics', 'metrics'))
    add = False
    ins_num = 0
    origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
    add_running = False
    while(line < len(origin_code.split('\n'))):
        code_list = origin_code.split('\n')
        code = code_list[line]
        # print(code)
        m_type = -1
        now_key = ''
        now_len = -1
        now_name = ''
        for key in metirc_dic.keys():
            if key in code:
                m_type = metirc_dic[key]['type']
                now_key = key
                now_len = metirc_dic[key]['len']
                now_name = metirc_dic[key]['name']
                break
        if m_type == 1:
            index = code.find(now_key)
            head = index - 1
            # print("index:",index)
            while (code[head].isalpha() or code[head] == '_' \
                   or code[head] == ']' \
                   or code[head] == ')' \
                   or code[head] == '.' \
                   or code[head] == '\'' or code[head] == '\"' \
                   or code[head].isalnum()) \
                    and head != -1:
                if code[head] == ']':
                    while code[head] != '[':
                        head -= 1
                if code[head] == ')':
                    while code[head] != '(':
                        # print(code[head])
                        head -= 1
                head -= 1
            left_index = index + now_len
            left_num = 1
            right_index = 0
            for ind in range(left_index + 1, len(code)):
                if code[ind] == '(':
                    left_num += 1
                elif code[ind] == ')':
                    left_num -= 1
                if left_num == 0:
                    right_index = ind
                    break
            model_id = -1
            for item in model_pred:
                if model_pred[item] in code:
                    model_id = model_dic[item]
                elif item in code:
                    model_id = model_dic[item]

            temp_code = code
            temp_code = temp_code.replace(' ', '')
            temp_code = temp_code.replace('\t', '')
            temp_code = temp_code.replace('\'', '\\\'')
            temp_code = temp_code.replace('"', '\\"')

            ctxt = code[head + 1:right_index + 1]
            add_model_result = get_model_from_code(ctxt, origin_code, line, ins_num)
            origin_code = add_model_result[1]
            add_line = add_model_result[2]
            # line += 1
            line += add_line

            add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
            origin_code = add_model_result_1[1]
            add_line_1 = add_model_result_1[2]
            line += add_line_1

            need2print = "insert_result('" +add_model_result[0] + "'," + code[head + 1:right_index + 1] + ',"' + temp_code + '", "'+ now_name +'", mdtypes_'+ str(ins_num) +')'

            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            add_running = True
            ins_num += 1
            line += 1

            origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
            line += 1
            add = True
        elif m_type == 2:
            index = code.find(now_key)
            head = index - 2

            while (code[head].isalpha() or code[head] == '_' \
                   or code[head] == ']' \
                   or code[head] == ')' \
                   or code[head] == '.' \
                   or code[head] == '\'' or code[head] == '\"' \
                   or code[head].isalnum()) \
                    and head != -1:
                if code[head] == ']':
                    while code[head] != '[':
                        head -= 1
                if code[head] == ')':
                    while code[head] != '(':
                        # print(code[head])
                        head -= 1
                head -= 1

            right_index = index + now_len

            model_id = -1
            for item in model_pred:
                if model_pred[item] in code:
                    model_id = model_dic[item]
                elif item in code:
                    model_id = model_dic[item]

            temp_code = code
            temp_code = temp_code.replace(' ','')
            temp_code = temp_code.replace('\t', '')
            temp_code = temp_code.replace('\'', '\\\'')
            temp_code = temp_code.replace('"', '\\"')

            ctxt = code[head + 1:right_index + 1]
            # print('code:',code)
            add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
            origin_code = add_model_result[1]
            add_line = add_model_result[2]
            # line += 1
            line += add_line

            add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
            origin_code = add_model_result_1[1]
            add_line_1 = add_model_result_1[2]
            line += add_line_1

            need2print = "insert_result('" + add_model_result[0] + "'," + code[head+1:right_index + 1] +',"' + temp_code + '", "' + now_name +'", mdtypes_'+ str(ins_num) +')'

            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            add_running = True
            ins_num += 1
            line += 1
            origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
            line += 1
            add = True
        elif m_type == 3:
            now_len = len(now_key)-1
            add = True
            index = code.find(now_key)
            print('code:',code)
            if index == 0:
                head = index
                left_index = index  + now_len
                left_num = 1

                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break

                model_id = -1
                for item in model_pred:
                    if model_pred[item] in code:
                        model_id = model_dic[item]
                    elif item in code:
                        model_id = model_dic[item]


                temp_code = code
                temp_code = temp_code.replace(' ','')
                temp_code = temp_code.replace('\t', '')
                temp_code = temp_code.replace('\'', '\\\'')
                temp_code = temp_code.replace('"', '\\"')

                ctxt = code[head:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # line += 1
                print("line0:", line)
                line += add_line

                add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
                origin_code = add_model_result_1[1]
                add_line_1 = add_model_result_1[2]
                line += add_line_1

                need2print = "insert_result('" + add_model_result[0] + "'," + code[head:right_index + 1] + ',"' + temp_code + '", "' + now_name +'", mdtypes_'+ str(ins_num) +')'

                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                add_running = True
                ins_num += 1
                line += 1
                origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
                line += 1
                print("line1:",line)

            elif code[index-1] != '.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index
                left_index = index + now_len
                left_num = 1

                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break

                model_id = -1
                for item in model_pred:
                    if model_pred[item] in code:
                        model_id = model_dic[item]
                    elif item in code:
                        model_id = model_dic[item]

                temp_code = code
                temp_code = temp_code.replace(' ', '')
                temp_code = temp_code.replace('\t', '')
                temp_code = temp_code.replace('\'', '\\\'')
                temp_code = temp_code.replace('"', '\\"')

                ctxt = code[head:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
                need2print = "insert_result('" + add_model_result[0] + "'," + code[
                                                                      head:right_index + 1] + ',"' + temp_code + '", "'+ now_name +'", mdtypes_'+ str(ins_num) +')'
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # line += 1
                # print(add_line)
                line += add_line

                add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
                origin_code = add_model_result_1[1]
                add_line_1 = add_model_result_1[2]
                line += add_line_1

                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                add_running = True
                ins_num += 1
                line += 1
                origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
                line += 1


            else:
                head = index - 2
                while (code[head].isalpha() or code[head] == '_' \
                        or code[head] == ']' \
                        or code[head] == ')' \
                        or code[head] == '.' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum())\
                        and head != -1:
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    if code[head] == ')':
                        while code[head] != '(':
                            # print(code[head])
                            head -= 1
                    head -= 1
                left_index = index + now_len
                left_num = 1

                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break

                model_id = -1
                for item in model_pred:
                    if model_pred[item] in code:
                        model_id = model_dic[item]
                    elif item in code:
                        model_id = model_dic[item]

                temp_code = code
                temp_code = temp_code.replace(' ', '')
                temp_code = temp_code.replace('\t', '')
                temp_code = temp_code.replace('\'', '\\\'')
                temp_code = temp_code.replace('"', '\\"')

                ctxt = code[head+1:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code, line,ins_num)
                need2print = "insert_result('" + add_model_result[0] + "'," + code[head+1:right_index + 1] + ',"' + temp_code + '", "' + now_name +'", mdtypes_'+ str(ins_num) +')'
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # line += 1
                line += add_line

                add_model_result_1 = get_model_from_error_param_code(ctxt, origin_code, line, ins_num)
                origin_code = add_model_result_1[1]
                add_line_1 = add_model_result_1[2]
                line += add_line_1

                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                add_running = True
                ins_num += 1
                line += 1
                origin_code = insert_one_line_in_code(origin_code, 0, 'mdtypes_' + str(ins_num) + '  = []')
                line += 1



        line += 1
    # print(origin_code)
    return origin_code, add,add_running

        # if '.score(' in code:
        #     # model = check_model(code)
        #     # index[count] = (model,1)
        #     print("score:",code)
        # elif 'accuracy_score(' in code:
        #     # model = check_model(code)
        #     # index[count] = (model,1)
        #     print("accuracy_score:",code)
        # else:
        #     print(code)
def check_no_model(root_path = "../notebook"):
    no_model_notebook_id_list = find_special_notebook()
    model_dic = eval(CONFIG.get('models', 'model_dic'))
    for id in no_model_notebook_id_list:
        origin_code = get_code_txt(root_path + '/' + str(id) + '.ipynb')
        for model_name in model_dic:
            if model_name + '(' in origin_code:
                print(str(id)+ ":found model!", model_name)

def test_running(notebook_id, notebook_root, dataset_root, notebook_name, dataset_name):
    """
    获取字符串变化
    :param notebook_root:
    :param dataset_root:
    :param notebook_name:
    :param dataset_name:
    :return:
    """
    notebook_path = notebook_root + '/' + notebook_name
    dataset_path_root = dataset_root + '/' + dataset_name
    func_def = get_code_txt(notebook_path)
    can_run_code = running(notebook_id, func_def,dataset_path_root,0)
    if can_run_code == "compile fail":
        print("\033[0;31;40mcompile fail\033[0m")
        return


    walk_logs = single_running(1, notebook_name.split('.')[0], notebook_root)

    add_csv_code,num_origin = print_readcsv(can_run_code)
    varible_list = []
    for i in range(0, num_origin):
        varible_list.append("origin_data_" + str(i))
    for i in walk_logs["data_values"]:
        varible_list.append(i)

    add_varible_code_content = add_variable_code(add_csv_code, varible_list,notebook_name.split('.')[0])
    can_run_code = running(notebook_id, add_varible_code_content, dataset_path_root, 0)
def get_model_from_code(ctxt,origin_code,line_num,ins_num):
    add_line = 0
    # print('ctxt:',ctxt)
    def get_type1(line,origin_code,line_num,model_variable,add_line,ins_num):
        # print(line.strip())
        try:
            r_node = ast.parse(line.strip())
        except:
            if 'for' in line and 'in' in line and ':' in line:
                origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                                      "if type(" + model_variable + ').__name__ not in mdtypes_' + str(ins_num) +  ": mdtypes_" + str(ins_num) +".append(type(" + model_variable + ').__name__)')
                add_line += 1
                line_num += 1
                return 'unknown',origin_code,add_line
            elif 'def' in line and '(' in line and '):' in line:
                left = line.replace(' ','').index('(')
                right = line.replace(' ','').index(')')
                p_list = line.replace(' ','')[left+1:right].split(',')
                if model_variable in p_list:
                    origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                                          "if type(" + model_variable + ').__name__ not in mdtypes_' + str(
                                                              ins_num) + ": mdtypes_" + str(
                                                              ins_num) + ".append(type(" + model_variable + ').__name__)')
                    add_line += 1
                    line_num += 1
                    return 'unknown',origin_code,add_line
            return False
        if len(r_node.body) == 0:
            return False
        # print('/////')
        # print(line_num+1)
        # print(model_variable)

        origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                              "if type(" + model_variable + ').__name__ not in mdtypes_' + str(
                                                  ins_num) + ": mdtypes_" + str(
                                                  ins_num) + ".append(type(" + model_variable + ').__name__)')
                                               # "mdtypes_" + str(ins_num) +".add(type(" + model_variable + ').__name__)')
        add_line += 1
        line_num += 1
        # print(type(r_node.body[0]).__name__)
        if type(r_node.body[0]).__name__ == 'Assign':
            model_node = r_node.body[0].value
            target_code = astunparse.unparse(r_node.body[0].targets)
            value_code = astunparse.unparse(r_node.body[0].value)
            # print('target_code:',target_code)
            # print('value_code:',value_code)
            if target_code[-1] == '\n':
                target_code = target_code[0:-1]
            if target_code in value_code and target_code != 'pred':
                return 'unknown',origin_code,add_line
            code_list = origin_code.split('\n')
            # for index,line in enumerate(code_list):
            #     print('mmm'+str(index),line)
            return get_type(model_node,origin_code,line_num,add_line,ins_num)
        # elif type(r_node.body[0]).__name__ == 'Assign':
        #     model_node = r_node.body[0].value
        #     code_list = origin_code.split('\n')
        #     for index,line in enumerate(code_list):
        #         print('mmm'+str(index),line)
        #     return get_type(model_node,origin_code,line_num)

    def check_line(model_variable,line):

        try:
            r_node = ast.parse(line.strip())
        except:
            if 'for' in line and 'in' in line and ':' in line:
                index_for = line.find('for')
                index_in = line.find('in')
                subline = line[index_for+4:index_in].strip()
                subline_list = subline.split(',')
                # print(subline_list)
                # print(model_variable)
                for code in subline_list:
                    if code.strip() == model_variable:
                        # print('true')
                        return True
                return False
            elif 'def' in line and '(' in line and '):' in line:
                left = line.replace(' ','').index('(')
                right =line.replace(' ','').index(')')
                p_list = line.replace(' ','')[left+1:right].split(',')
                if model_variable in p_list:
                    return True
            return False
        if len(r_node.body) == 0:
            return False
        # print(line,type(r_node.body[0]).__name__)
        if type(r_node.body[0]).__name__ == 'Assign':
            for target in r_node.body[0].targets:
                if type(target).__name__ == 'Name':
                    if target.id == model_variable:
                        return True

        return False

    def get_type(model_node, origin_code,line_num,add_line,ins_num):
        model_variable = ''
        if type(model_node).__name__ == 'Attribute':
            model_node = model_node.value
        if type(model_node).__name__ == 'Subscript':
            model_node = model_node.value
        if type(model_node).__name__ == 'Compare':
            model_node = model_node.left
        if type(model_node).__name__ == 'Name':
            if model_node.id[-1] == '\n':
                model_variable = model_node.id[0:-1]
            else:
                model_variable = model_node.id
        elif type(model_node).__name__ == 'List':
            # print('555')
            return 'unknown',origin_code,add_line
        elif type(model_node).__name__ == 'Call':
            call_func = model_node.func
            if type(call_func).__name__ == 'Attribute':
                if type(call_func.value).__name__ == 'Attribute':
                    call_func.value = call_func.value.value
                if type(call_func.value).__name__ == 'Name':
                    if call_func.value.id[-1] == '\n':
                        model_variable = call_func.value.id[0:-1]
                    else:
                        model_variable = call_func.value.id
                elif type(call_func.value).__name__ == 'Call':
                    if type(call_func.value.func).__name__ == 'Name':
                        return call_func.value.func.id,origin_code,add_line
                else:
                    # print('444')
                    return 'unknown',origin_code,add_line
            elif type(call_func).__name__ == 'Name':
                # print('333')
                if call_func.id == 'cross_val_predict':
                    args =model_node.args
                    kws = model_node.keywords
                    if len(args) >= 1:
                        mn = args[0]
                        if type(mn).__name__ == 'Name':
                            model_variable = mn.id
                        elif type(mn).__name__ == 'Call':
                            if type(mn.func).__name__ == 'Name':
                                return mn.func.id,origin_code,add_line
                            else:
                                return astunparse.unparse('mn.func'),origin_code,add_line
                return call_func.id,origin_code,add_line

        code_list = origin_code.split('\n')
        line_list = []
        new_line_num = 0


        # print('model_variable:', model_variable)
        # print('line_num:',line_num)
        for index,line in enumerate(code_list):
            if index > line_num-1:
                # print(line_num)
                break
            if check_line(model_variable,line) == True:
                if line not in line_list:
                    line_list.append(line)
                new_line_num = index
        # print(line_list)
        if len(line_list) == 0:
            # print('111')
            return 'unknown',origin_code,add_line
        else:
            need_parse_line = ''
            # print(line_list)
            for i in range(0, len(list(line_list))):
                # print('sL:',list(line_list)[len(list(line_list)) - 1 - i])
                if '.predict(' not in list(line_list)[len(list(line_list)) - 1 - i]:
                    continue
                else:
                    # print('linelist:',line_list)
                    need_parse_line = list(line_list)[len(list(line_list)) - 1 - i]
                    break
            # print('need:',need_parse_line)
            if need_parse_line == '':
                need_parse_line = list(line_list)[-1]
            # print('\033[0;36;40m' + need_parse_line + '\033[0m')
            # print('get_type:', need_parse_line)s
            # print('222')
            # print(need_parse_line)
            return get_type1(need_parse_line,origin_code,new_line_num,model_variable,add_line,ins_num)

        # print('\033[0;36;40mmodel_variable:'+model_variable+'\033[0m')




    # print('ctxt:', ctxt)
    r_node = ast.parse(ctxt)
    if type(r_node.body[0].value).__name__ == 'Call':
        call_func = r_node.body[0].value.func
        arg_list = r_node.body[0].value.args
        kw_list = r_node.body[0].value.keywords
        # print(type(call_func).__name__)
        if (type(call_func).__name__ == 'Attribute'): #xxx.score
            if call_func.attr == 'score':
                model_node = call_func.value
            elif call_func.attr == 'evaluate':
                model_node = call_func.value
            elif call_func.attr == 'classification_report':
                return 'unknown',origin_code,add_line
            else:
                func_name = call_func.attr
                if func_name == 'cross_val_score':
                    if len(arg_list) >= 1:
                        model_node = arg_list[0]
                    for kw_node in kw_list:
                        if 'estimator' == kw_node.arg:
                            model_node = kw_node.value
                elif func_name == 'auc':
                    is_in = False
                    if len(arg_list) >= 2:
                        model_node = arg_list[1]
                        is_in = True
                    for kw_node in kw_list:
                        if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                            model_node = kw_node.value
                            is_in = True
                            break
                    if is_in == False:
                        return 'unknown',origin_code,add_line

                else:
                    is_in = False
                    if len(arg_list) >= 2:
                        model_node = arg_list[1]
                        is_in = True
                    for kw_node in kw_list:
                        if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                            model_node = kw_node.value
                            is_in = True
                    if is_in == False:
                        return 'unknown',origin_code,add_line

        if (type(call_func).__name__ == 'Name'): # other
            func_name = call_func.id

            if func_name == 'cross_val_score':
                if len(arg_list) >= 1:
                    model_node = arg_list[0]
                for kw_node in kw_list:
                    if 'estimator' == kw_node.arg:
                        model_node = kw_node.value
            elif func_name == 'auc':
                is_in = False
                if len(arg_list) >= 2:
                    model_node = arg_list[1]
                    is_in = True
                for kw_node in kw_list:
                    if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                        model_node = kw_node.value
                        is_in=True
                        break
                if is_in == False:
                    return 'unknown',origin_code,add_line
            elif func_name == 'silhouette_score':
                is_in = False
                if len(arg_list) >= 2:
                    model_node = arg_list[1]
                    is_in = True
                for kw_node in kw_list:
                    if 'labels' == kw_node.arg:
                        model_node = kw_node.value
                        is_in=True
                        break
                if is_in == False:
                    return 'unknown',origin_code,add_line
            else:
                if len(arg_list) >= 2:
                    model_node = arg_list[1]
                for kw_node in kw_list:
                    if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                        model_node = kw_node.value

    elif type(r_node.body[0].value).__name__ == 'Attribute': #best_score_
        call_func = r_node.body[0].value.value
        attr = r_node.body[0].value.attr
        model_node = call_func
    # origin_code = insert_one_line_in_code(origin_code, line_num + 1,"print('model_type:',type(" + astunparse.unparse(model_node)[0:-1] + ').__name__)')
    result = get_type(model_node, origin_code, line_num,add_line,ins_num)
    # print(result)
    code_list = result[1].split('\n')

    # for index, line in enumerate(code_list):
    #     print(index, line)
    # print("\033[0;36;40m" + result+ "\033[0m")
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
    return result

def get_model_from_error_param_code(ctxt,origin_code,line_num,ins_num):
    add_line = 0
    # print('ctxt:',ctxt)
    def get_type1(line,origin_code,line_num,model_variable,add_line,ins_num):
        # print(line.strip())
        try:
            r_node = ast.parse(line.strip())
        except:
            if 'for' in line and 'in' in line and ':' in line:
                origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                                      "if type(" + model_variable + ').__name__ not in mdtypes_' + str(ins_num) +  ": mdtypes_" + str(ins_num) +".append(type(" + model_variable + ').__name__)')
                add_line += 1
                line_num += 1
                return 'unknown',origin_code,add_line
            elif 'def' in line and '(' in line and '):' in line:
                left = line.replace(' ','').index('(')
                right = line.replace(' ','').index(')')
                p_list = line.replace(' ','')[left+1:right].split(',')
                if model_variable in p_list:
                    origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                                           "if type(" + model_variable + ').__name__ not in mdtypes_' + str(ins_num) +  ": mdtypes_" + str(ins_num) +".append(type(" + model_variable + ').__name__)')
                    add_line += 1
                    line_num += 1
                    return 'unknown',origin_code,add_line
            return False
        if len(r_node.body) == 0:
            return False
        # print('/////')
        # print(line_num+1)
        # print(model_variable)
        origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                               "if type(" + model_variable + ').__name__ not in mdtypes_' + str(ins_num) +  ": mdtypes_" + str(ins_num) +".append(type(" + model_variable + ').__name__)')
        add_line += 1
        line_num += 1
        # print(type(r_node.body[0]).__name__)
        if type(r_node.body[0]).__name__ == 'Assign':
            model_node = r_node.body[0].value
            target_code = astunparse.unparse(r_node.body[0].targets)
            value_code = astunparse.unparse(r_node.body[0].value)
            if target_code[-1] == '\n':
                target_code = target_code[0:-1]
            if target_code in value_code and target_code != 'pred':
                return 'unknown', origin_code, add_line

            # for index,line in enumerate(code_list):
            #     print('mmm'+str(index),line)
            return get_type(model_node,origin_code,line_num,add_line,ins_num)
        # elif type(r_node.body[0]).__name__ == 'Assign':
        #     model_node = r_node.body[0].value
        #     code_list = origin_code.split('\n')
        #     for index,line in enumerate(code_list):
        #         print('mmm'+str(index),line)
        #     return get_type(model_node,origin_code,line_num)

    def check_line(model_variable,line):

        try:
            r_node = ast.parse(line.strip())
        except:
            if 'for' in line and 'in' in line and ':' in line:
                index_for = line.find('for')
                index_in = line.find('in')
                subline = line[index_for+4:index_in].strip()
                subline_list = subline.split(',')
                # print(subline_list)
                # print(model_variable)
                for code in subline_list:
                    if code.strip() == model_variable:
                        # print('true')
                        return True
                return False
            elif 'def' in line and '(' in line and '):' in line:
                left = line.replace(' ','').index('(')
                right = line.replace(' ','').index(')')
                p_list = line.replace(' ','')[left+1:right].split(',')
                if model_variable in p_list:
                    return True
            return False
        if len(r_node.body) == 0:
            return False
        # print(line,type(r_node.body[0]).__name__)
        if type(r_node.body[0]).__name__ == 'Assign':
            for target in r_node.body[0].targets:
                if type(target).__name__ == 'Name':
                    if target.id == model_variable:
                        return True

        return False

    def get_type(model_node, origin_code,line_num,add_line,ins_num):
        model_variable = ''
        if type(model_node).__name__ == 'Attribute':
            model_node = model_node.value
        if type(model_node).__name__ == 'Subscript':
            model_variable = astunparse.unparse(model_node)
        if type(model_node).__name__ == 'Compare':
            model_node = model_node.left
        if type(model_node).__name__ == 'Name':
            if model_node.id[-1] == '\n':
                model_variable = model_node.id[0:-1]
            else:
                model_variable = model_node.id
        elif type(model_node).__name__ == 'List':
            # print('555')
            return 'unknown',origin_code,add_line
        elif type(model_node).__name__ == 'Call':
            call_func = model_node.func
            if type(call_func).__name__ == 'Attribute':
                if type(call_func.value).__name__ == 'Attribute':
                    call_func.value = call_func.value.value
                if type(call_func.value).__name__ == 'Name':
                    if call_func.value.id[-1] == '\n':
                        model_variable = call_func.value.id[0:-1]
                    else:
                        model_variable = call_func.value.id
                elif type(call_func.value).__name__ == 'Call':
                    if type(call_func.value.func).__name__ == 'Name':
                        return call_func.value.func.id,origin_code,add_line
                else:
                    # print('444')
                    return 'unknown',origin_code,add_line
            elif type(call_func).__name__ == 'Name':
                # print('333')
                return call_func.id,origin_code,add_line

        code_list = origin_code.split('\n')
        line_list = []
        new_line_num = 0


        # print('model_variable:', model_variable)
        # print('line_num:',line_num)
        for index,line in enumerate(code_list):
            if index > line_num-1:
                # print('break:', line)
                break
            if check_line(model_variable,line) == True:
                if line not in line_list:
                    line_list.append(line)
                new_line_num = index
        # print(line_list)
        if len(line_list) == 0:
            # print('111')
            return 'unknown',origin_code,add_line
        else:
            need_parse_line = ''
            for i in range(0,len(list(line_list))):
                if '.predict' not in list(line_list)[len(list(line_list))-1-i]:
                    continue
                else:
                    need_parse_line = list(line_list)[len(list(line_list)) - 1 - i]
            if need_parse_line == '':
                need_parse_line = list(line_list)[-1]
            # print('\033[0;36;40m' + need_parse_line + '\033[0m')
            # print('get_type:', need_parse_line)s
            # print('222')
            return get_type1(need_parse_line,origin_code,new_line_num,model_variable,add_line,ins_num)

        # print('\033[0;36;40mmodel_variable:'+model_variable+'\033[0m')




    # print('ctxt:', ctxt)
    r_node = ast.parse(ctxt)
    if type(r_node.body[0].value).__name__ == 'Call':
        call_func = r_node.body[0].value.func
        arg_list = r_node.body[0].value.args
        kw_list = r_node.body[0].value.keywords
        # print(type(call_func).__name__)
        if (type(call_func).__name__ == 'Attribute'): #xxx.score
            if call_func.attr == 'score':
                model_node = call_func.value
            elif call_func.attr == 'evaluate':
                model_node = call_func.value
            elif call_func.attr == 'classification_report':
                return 'unknown',origin_code,add_line
            else:
                func_name = call_func.attr
                if func_name == 'cross_val_score':
                    if len(arg_list) >= 1:
                        model_node = arg_list[0]
                    for kw_node in kw_list:
                        if 'estimator' == kw_node.arg:
                            model_node = kw_node.value
                elif func_name == 'auc':
                    is_in = False
                    if len(arg_list) >= 2:
                        model_node = arg_list[1]
                        is_in = True
                    for kw_node in kw_list:
                        if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                            model_node = kw_node.value
                            is_in = True
                            break
                    if is_in == False:
                        return 'unknown',origin_code,add_line

                else:
                    is_in = False
                    if len(arg_list) >= 2:
                        model_node = arg_list[1]
                        is_in = True
                    for kw_node in kw_list:
                        if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                            model_node = kw_node.value
                            is_in = True
                    if is_in == False:
                        return 'unknown',origin_code,add_line

        if (type(call_func).__name__ == 'Name'): # other
            func_name = call_func.id

            if func_name == 'cross_val_score':
                if len(arg_list) >= 1:
                    model_node = arg_list[0]
                for kw_node in kw_list:
                    if 'estimator' == kw_node.arg:
                        model_node = kw_node.value
            elif func_name == 'auc':
                is_in = False
                if len(arg_list) >= 1:
                    model_node = arg_list[0]
                    is_in = True
                for kw_node in kw_list:
                    if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                        model_node = kw_node.value
                        is_in=True
                        break
                if is_in == False:
                    return 'unknown',origin_code,add_line

            else:
                if len(arg_list) >= 1:
                    model_node = arg_list[0]
                for kw_node in kw_list:
                    if 'y_pred' == kw_node.arg or 'y_score' == kw_node.arg:
                        model_node = kw_node.value

    elif type(r_node.body[0].value).__name__ == 'Attribute': #best_score_
        call_func = r_node.body[0].value.value
        attr = r_node.body[0].value.attr
        model_node = call_func

    # origin_code = insert_one_line_in_code(origin_code, line_num + 1,"print('model_type:',type(" + astunparse.unparse(model_node)[0:-1] + ').__name__)')
    # print("model_node",model_node)
    result = get_type(model_node, origin_code, line_num,add_line,ins_num)
    # print(result)
    code_list = result[1].split('\n')

    # for index, line in enumerate(code_list):
    #     print(index, line)
    # print("\033[0;36;40m" + result+ "\033[0m")
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
    return result
def get_finish_data(notebook_id, origin_code, rank, root_path = '../spider/notebook'):
    def find_train_variable(notebook_id,code_list,rank, model_type):
        model_variable = 'no_model_variable__'
        train_variable = []
        ind = 0
        rank_code = get_result_code(notebook_id, origin_code, rank, get_min_max=0)
        # print('rank_code:', rank_code)
        fit_variable = []
        # print('model_type:',model_type)
        for line in code_list:
            # print(line)
            if ind > rank_code[0][1]:
                break
            if 'cross_val_predict(' in line:
                start_index = line.find('cross_val_predict(')
                second_index = line[start_index:].find(',')
                end_index = line[start_index:][second_index:].find(',')
                train_var = line[start_index:][second_index:end_index].strip()
                train_variable.append((train_var, ind))

            elif model_type in line and '.fit(' in line:
                start_index = line.find('.fit(') + 5
                end_index = start_index
                while line[end_index] != ',':
                    end_index += 1
                train_variable.append((line[start_index: end_index].strip(), ind))
            elif model_type in line:
                try:
                    r_node = ast.parse(line.strip())
                    if type(r_node.body[0]).__name__ == 'Assign':
                        for target in r_node.body[0].targets:
                            if type(target).__name__ == 'Name':
                                model_variable = target.id
                except:
                    continue
                            # print('model_variable',model_variable)
            elif model_variable+'.fit(' in line:
                # print(ind)
                start_index = line.find('.fit(') + 5
                end_index = start_index
                while line[end_index] != ',':
                    end_index += 1
                train_variable.append((line[start_index:end_index].strip(), ind))
                # print('train_variable:',train_variable)
                fit_variable.append((line[start_index:end_index].strip(), ind))
            elif '.fit(' in line:
                start_index = line.find('.fit(') + 5
                end_index = start_index
                # print(line)
                while line[end_index] != ',':
                    end_index += 1
                    if end_index > len(line)-1:
                        break
                if end_index <= len(line) - 1:
                    fit_variable.append((line[start_index:end_index].strip(), ind))
            ind += 1
        # print(fit_variable)
        if len(train_variable) == 0:
            train_variable.append(fit_variable[-1])
        return train_variable[-1]

    code_list = origin_code.split('\n')
    cursor, db = create_connection()
    sql = 'select id,model_type,metric_type,code from result where notebook_id='+str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    count = 1
    result = 'no such rank'
    for row in sql_res:
        if count != rank:
            count += 1
            continue
        model_type = row[1]
        if row[2] == 'cross_val_score':
            rank_code = get_result_code(notebook_id, origin_code, rank, get_min_max=0)
            code = row[3]
            start_index = code.find('cross_val_score(') + 16
            end_index = start_index
            while code[end_index] != ',':
                end_index += 1
            start_index = end_index+1

            end_index = start_index
            while code[end_index] != ',':
                end_index += 1
            train_variable = code[start_index: end_index].strip()
            result = (train_variable,rank_code[0][-1])
            # print('result:', result)
        elif model_type == 'unknown':
            result = 'no such model'
        else:
            result = find_train_variable(notebook_id, code_list, rank, model_type)
            # print('result:',result)
        break
    return result

def get_dataframe_and_operator_list_from_one_result(notebook_id, origin_code, result_id, result_rank):
    code_list = origin_code.split('\n')
    ope_dic = eval(CONFIG.get('operators', 'operations'))
    cursor, db = create_connection()

    # result_code_and_line = get_result_code(notebook_id, origin_code, result_rank, get_min_max=0)
    # res_tup = result_code_and_line[0]

    sql = 'SELECT rank from operator where notebook_id=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    res_tup = get_finish_data(notebook_id, origin_code, result_rank)
    if res_tup == 'no such rank' or 'no such model':
        return 'data wrong'
    operator_list = []
    for row in sql_res:
        operator_code_and_line = get_operator_code(notebook_id, origin_code, row[0], ope_dic, get_min_max=0)
        print('ope:', operator_code_and_line)
        ope_tup = operator_code_and_line[0][0]
        ope_triple = (ope_tup[0], ope_tup[1], row[0])
        # print(ope_triple)
        # print(ope_triple[1],res_tup[1])
        if ope_triple[1] < res_tup[1]:
            operator_list.append(ope_triple)


    if not os.path.exists('../predf/'):
        os.mkdir('../predf/')
    if not os.path.exists('../predf/'+str(notebook_id)):
        os.mkdir('../predf/'+str(notebook_id))
    # if not os.path.exists('../predf/'+str(notebook_id)+'/' + str(result_id)):
    #     os.mkdir('../predf/'+str(notebook_id)+'/' + str(result_id))
    # new_code = insert_one_line_in_code(origin_code, res_tup[1] + 1,
    #                                    'print("tp:",type('+res_tup[0]+').__name__)')
    new_code = insert_one_line_in_code(origin_code, res_tup[1] + 1,
                                       'if type('+res_tup[0]+').__name__ == "Dataframe":')
    new_code = insert_one_line_in_code(new_code, res_tup[1] + 2,
                                       '    '+res_tup[0] + '.to_csv("../predf/' + str(notebook_id) + '/' + str(
                                           result_id) + '.csv",encoding="gbk")')
    new_code = insert_one_line_in_code(new_code, res_tup[1] + 3,
                                       'else:')
    # new_code = insert_one_line_in_code(new_code, res_tup[1] + 5, '    print("hhhhhhhhhh)"')
    new_code = insert_one_line_in_code(new_code,res_tup[1]+4,'    np.save("../predf/'+ str(notebook_id) + '/' + str(result_id) + '.npy", '+ res_tup[0] + ')')
    update_db('result', 'data_object_value', res_tup[0], 'id', '=', result_id)
    cl = new_code.split('\n')

    # print('operator_list:',operator_list)
    # print('result_line:',res)
    return new_code, operator_list,res_tup


def save_dataframe_and_update_reuslt(notebook_id,notebook_root="../spider/notebook/"):
    # try:
    origin_code = get_code_txt(notebook_root+'/' + str(notebook_id) + '.ipynb')
    code_list = origin_code.split('\n')
    # for index, line in enumerate(code_list):
    #     print(index, line)
    # # except:
    #     return 'read error'

    cursor, db = create_connection()
    sql = 'SELECT id from result where notebook_id=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    count = 1
    for row in sql_res:
        result = get_dataframe_and_operator_list_from_one_result(notebook_id, origin_code, row[0], count)
        if result == 'data wrong':
            break
        origin_code = result[0]
        operator_list = result[1]
        result_line = result[2]
        operator_list_string = ''
        for ope in operator_list:
            operator_list_string += str(ope[2])
            operator_list_string += ','
        operator_list_string = operator_list_string[0:-1]
        # print('xxxxxx')
        # print('operator_list:',operator_list)
        # print(count, operator_list_string)
        # print(result_line)
        count += 1
        update_db('result','sequence',operator_list_string,'id','=',row[0])
    code_list = origin_code.split('\n')
    # for index, line in enumerate(code_list):
    #     print(index, line)
    return origin_code
def save_runnings(notebook_id, dataset_name,notebook_root="../notebook", dataset_root="../dataset"):
    dataset_path_root = dataset_root + '/' + dataset_name + '.zip'
    origin_code = save_dataframe_and_update_reuslt(notebook_id, notebook_root)
    code_list = origin_code.split('\n')
    for index, line in enumerate(code_list):
        print(index, line)
    add_code = origin_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
    add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer',
                                'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.externals import joblib', 'import joblib')
    add_code = add_code.replace('pandas.tools', 'pandas')
    add_code = add_code.replace('from sklearn import cross_validation',
                                'from sklearn.model_selection import cross_val_score')
    add_code = add_code.replace('from sklearn.cross_validation import',
                                'from sklearn.model_selection import')
    add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')

    add_code = insert_one_line_in_code(add_code, 'import matplotlib.pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')

    can_run_code = running(notebook_id, add_code,dataset_path_root,0)
    if can_run_code == "compile fail":
        print("\033[0;31;40mcompile fail\033[0m")
        return "compile fail"
    elif can_run_code == 'False':
        return "False"
    else:
        return 'succeed'

def single_running_and_save(notebook_id, dataset_name ,notebook_root="../notebook",dataset_root="../dataset", need_add=False):
    dataset_path_root = dataset_root + '/' + dataset_name + '.zip'
    # try:
    #     origin_code = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
    #     code_list = origin_code.split('\n')
    # except Exception as e:
    #     print(e)
    #     return "read fail"
    print('save before')
    origin_code = save_dataframe_and_update_reuslt(notebook_id, notebook_root)
    print('save after')
    add_code, add, add_running = add_result(notebook_id, origin_code)
    add_code = add_params_miresult(notebook_id, add_code)

    add_code = add_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
    add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer', 'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.externals import joblib','import joblib')
    add_code = add_code.replace('pandas.tools', 'pandas')
    add_code = add_code.replace('from sklearn import cross_validation', 'from sklearn.model_selection import cross_val_score')
    add_code = add_code.replace('from sklearn.cross_validation import',
                                'from sklearn.model_selection import')
    add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')

    add_code = insert_one_line_in_code(add_code,'import matplotlib.pyplot as plt','matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')

    for index,i in enumerate(add_code.split('\n')):
        print(index, i)
    if add_running == False:
        return 'no result pass'
    if need_add == True:
        if add == True:
            can_run_code = running(notebook_id, add_code, dataset_path_root, 0)
    else:
        can_run_code = running(notebook_id, add_code,dataset_path_root,0)
    if can_run_code == "compile fail":
        print("\033[0;31;40mcompile fail\033[0m")
        return "compile fail"
    elif can_run_code == 'False':
        return "False"
    else:
        return 'succeed'


def single_runnings(notebook_id, dataset_name ,notebook_root="../notebook",dataset_root="../dataset", need_add=False):
    dataset_path_root = dataset_root + '/' + dataset_name + '.zip'
    try:
        origin_code = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
        code_list = origin_code.split('\n')
    except Exception as e:
        print(e)
        return "read fail"

    add_code, add, add_running = add_result(notebook_id, origin_code)
    add_code = add_params_miresult(notebook_id, add_code)

    add_code = add_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
    add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer', 'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.externals import joblib','import joblib')
    add_code = add_code.replace('pandas.tools', 'pandas')
    add_code = add_code.replace('from sklearn import cross_validation', 'from sklearn.model_selection import cross_val_score')
    add_code = add_code.replace('from sklearn.cross_validation import',
                                'from sklearn.model_selection import')
    add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')

    add_code = insert_one_line_in_code(add_code,'import matplotlib.pyplot as plt','matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')

    for index,i in enumerate(add_code.split('\n')):
        print(index, i)
    if add_running == False:
        return 'no result pass'
    if need_add == True:
        if add == True:
            can_run_code = running(notebook_id, add_code, dataset_path_root, 0)
    else:
        can_run_code = running(notebook_id, add_code,dataset_path_root,0)
    if can_run_code == "compile fail":
        print("\033[0;31;40mcompile fail\033[0m")
        return "compile fail"
    elif can_run_code == 'False':
        return "False"
    else:
        return 'succeed'

def batch_running_single(notebook_root="../notebook",dataset_root="../dataset",ip="39.99.150.216", pair_type=5):
    print(pair_type)
    notebook_id = 7786621
    pairs = get_pair(ip, type=pair_type, notebook_id=notebook_id)
    print(len(pairs))
    count = 0
    for pair in pairs:
        # try:
        # print(pair[0])
        # print(type(pair[0]).__name__)
        # if count == 1:
        # print(type(pair[0]))
        # if pair[0] == 1543334:
        notebook_id = pair[0]
        dataset_name = pair[1]
        # if notebook_name == "most-dangerous-departure-and-destination-cities.ipynb":
        print("\033[0;33;44m" + str(notebook_id) + "\033[0m")
        if len(pair) == 3:
            print(pair[2])
        res = single_running_and_save(notebook_id, dataset_name, notebook_root, dataset_root)
        # print(res)
        if res != 'False' and res != 'compile fail' and res != 'read fail':
            update_db("notebook", "add_run", '1', 'id', '=', notebook_id)
            update_db("notebook", "add_model", '1', 'id', "=", notebook_id)
            update_db("notebook", "add_seq_df", '1', 'id', '=', notebook_id)
        if res == 'compile fail':
            update_db("notebook", "add_run", '2', 'id', '=', notebook_id)
            update_db("notebook", "add_seq_df", '2', 'id', '=', notebook_id)
        if res == 'False':
            update_db("notebook", "add_run", '3', 'id', '=', notebook_id)
            update_db("notebook", "add_model", '4', 'id', "=", notebook_id)
            update_db("notebook", "add_seq_df", '3', 'id', '=', notebook_id)
        if res == 'read fail':
            update_db("notebook", "add_run", '4', 'id', '=', notebook_id)
            update_db("notebook", "add_seq_df", '4', 'id', '=', notebook_id)
        if res == 'no result pass':
            update_db("notebook", "add_run", '6', 'id', '=', notebook_id)
            update_db("notebook", "add_seq_df", '6', 'id', '=', notebook_id)

def batch_save_running(notebook_root="../notebook",dataset_root="../dataset",ip="39.99.150.216", pair_type=5):
    print(pair_type)
    notebook_id = 5078241
    pairs = get_save_pair(ip)
    print(len(pairs))
    count = 0
    for pair in pairs:
        # try:
        # print(pair[0])
        # print(type(pair[0]).__name__)
        # if count == 1:
        # print(type(pair[0]))
        # if pair[0] == '10856937':
            notebook_id = pair[0]
            dataset_name = pair[1]
            # if notebook_name == "most-dangerous-departure-and-destination-cities.ipynb":
            print("\033[0;33;44m" + str(notebook_id) + "\033[0m")
            if len(pair) == 3:
                print(pair[2])
            res = save_runnings(notebook_id, dataset_name, notebook_root, dataset_root)
            # print(res)
            if res != 'False' and res != 'compile fail' and res != 'read fail':
                update_db("notebook", "add_seq_df", '1', 'id', '=', notebook_id)
            if res == 'compile fail':
                update_db("notebook", "add_seq_df", '2', 'id', '=', notebook_id)
            if res == 'False':
                update_db("notebook", "add_seq_df", '3', 'id', '=', notebook_id)
            if res == 'read fail':
                update_db("notebook", "add_seq_df", '4', 'id', '=', notebook_id)
            if res == 'no result pass':
                update_db("notebook", "add_seq_df", '6', 'id', '=', notebook_id)
            # break

def check_random_result(notebook_id, dataset_name, notebook_root='../notebook',dataset_root='../dataset'):
    try:
        origin_code = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
        # code_list = origin_code.split('\n')
    except Exception as e:
        print(e)
        return "read fail"

    dataset_path_root = dataset_root + dataset_name + '.zip'
    add_code, add, add_running = add_changed_result(notebook_id, origin_code)
    # add_code = add_params_miresult(notebook_id, add_code)

    add_code = add_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
    add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer', 'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.externals import joblib','import joblib')
    add_code = add_code.replace('pandas.tools', 'pandas')
    add_code = add_code.replace('from sklearn import cross_validation', 'from sklearn.model_selection import cross_val_score')
    add_code = add_code.replace('from sklearn.cross_validation import',
                                'from sklearn.model_selection import')
    add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')

    add_code = insert_one_line_in_code(add_code,'import matplotlib.pyplot as plt','matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')
    code_list = add_code.split('\n')
    for index, line in enumerate(code_list):
        print("\033[0;36;40m" + str(index) + ':' + line + "\033[0m")

    res = running_temp_code(add_code, dataset_path_root, 0)
    print(res)
    if res[0:7] == 'error 8':
        return 'error 8'
    # terminal = True
    try:
        changed_result = np.load('./temp_result.npy', allow_pickle=True)
    except:
        return 'error 8'
    # os.system('rm -f ./temp_result.npy')
    origin_result = []
    cursor, db = create_connection()
    sql = 'select * from result where notebook_id=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        if row[2] != None:
            origin_result.append([row[1], row[3], row[2], row[5], row[4]])
        else:
            origin_result.append([row[1], row[3], row[6], row[5], row[4]])

    if len(changed_result) != len(origin_result):
        return False

    for index, item in enumerate(changed_result):
        print('changed_result:', changed_result[index])
        print('origin_result:', origin_result[index])
        if type(item[2]).__name__ == 'list' and item[4] == 'evaluate':
            item[2] = item[2][-1]
            print(item[2])
        elif type(item[2]).__name__ == 'ndarray' and item[4] == 'cross_val_score':
            sum = 0
            for score in item[2]:
                sum += score
            if len(item[2]) == 0:
                item[2] = 0
            else:
                item[2] = sum / len(item[2])
        elif item[4] == 'confusion_matrix' or item[4] == 'classification_report' or item[
            4] == 'mean_absolute_error' or item[4] == 'mean_squared_error':
            continue

        if type(origin_result[index][2]).__name__ == 'str' and origin_result[index][4] == 'evaluate':
            origin_result[index][2] = origin_result[index][2][1:-1].split(',')[-1].strip()
            print(origin_result[index][2])
        elif type(origin_result[index][2]).__name__ == 'str' and origin_result[index][4] == 'cross_val_score':
            continue
        elif origin_result[index][4] == 'confusion_matrix' or origin_result[index][4] == 'classification_report' or origin_result[index][4] == 'mean_absolute_error' or origin_result[index][4] == 'mean_squared_error':
            continue
        # print(item[2])
        now_result = float(item[2])
        pre_result = float(origin_result[index][2])
        # print(item[1])
        if abs(now_result - pre_result) > 0.00001:
            return False

    return True


def batch_checking(notebook_root="../notebook",dataset_root="../unzip_dataset/",ip="39.99.150.216"):
    cursor, db = create_connection()
    in_result = []
    sql = 'select distinct notebook_id from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(int(row[0]))

    sql = 'select pair.nid,dataset.dataSourceUrl from pair,dataset where pair.did=dataset.id and dataset.isdownload=1 and dataset.server_ip=\'' + ip + "'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    continued = True
    for row in sql_res:
        # notebook_id = 3383747
        notebook_id = int(row[0])
        # print(str(notebook_id)[0:1])
            # continue
        # if continued == True:
        #     continue
        # if notebook_id != 3383747:
        #     continue
        print('notebook_id:',notebook_id)
        dataset_name  = row[1].split('/')[-1]
        if notebook_id not in in_result:
            continue
        sql = 'select isRandom from notebook where id='+str(notebook_id)
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        is_continue = False
        for row1 in sql_res:
            print(row1[0])

            if row1[0] != 0:
                is_continue = True
        if is_continue:
            continue
        check_result = check_random_result(notebook_id, dataset_name, notebook_root=notebook_root, dataset_root=dataset_root)
        if check_result == 'error 8':
            update_db('notebook', 'isRandom', -1, 'id', '=', notebook_id)
            continue
        print(check_result)
        if check_result == True:
            update_db('notebook','isRandom',1,'id','=',notebook_id)
        if check_result == False:
            update_db('notebook','isRandom',-1,'id','=',notebook_id)
        break

def batch_running(notebook_root="../notebook",dataset_root="../dataset",ip="39.99.150.216", pair_type=5):
    print(pair_type)
    notebook_id = 5078241
    pairs = get_pair(ip,type=pair_type,notebook_id=notebook_id)
    print(len(pairs))
    count = 0
    for pair in pairs:
        # try:
        # print(pair[0])
        # print(type(pair[0]).__name__)
        # if count == 1:
        # print(type(pair[0]))
        # if pair[0] == 1543334:
            notebook_id = pair[0]
            dataset_name = pair[1]
            if dataset_name == 'None':
                continue
            # if notebook_name == "most-dangerous-departure-and-destination-cities.ipynb":
            print("\033[0;33;44m" + str(notebook_id) + "\033[0m")
            if len(pair) == 3:
                print(pair[2])
            res = single_runnings(notebook_id, dataset_name, notebook_root, dataset_root)
            # print(res)
            if res != 'False' and res != 'compile fail' and res != 'read fail':
                update_db("notebook", "add_run", '1', 'id', '=', notebook_id)
                update_db("notebook", "add_model", '1', 'id', "=", notebook_id)
            if res == 'compile fail':
                update_db("notebook", "add_run", '2', 'id', '=', notebook_id)
            if res == 'False':
                update_db("notebook", "add_run", '3', 'id', '=', notebook_id)
                update_db("notebook", "add_model", '4', 'id', "=", notebook_id)
            if res == 'read fail':
                update_db("notebook", "add_run", '4', 'id', '=', notebook_id)
            if res == 'no result pass':
                update_db("notebook", "add_run", '6', 'id', '=', notebook_id)
        # count += 1
        #
    # print(count)

def add_finish_data(notebook_id, root_path = '../spider/notebook'):
    def find_train_variable(notebook_id, model_type, root_path):
        try:
            origin_code = get_code_txt(root_path + '/' + str(notebook_id) + '.ipynb')
        except Exception as e:
            print(e)
            return "read fail"
        code_list = origin_code.split('\n')
        model_variable = 'no_model_variable__'
        train_variable = []
        for line in code_list:
            if model_type in line and '.fit(' in line:
                start_index = line.find('.fit(') + 5
                end_index = start_index
                while line[end_index] != ',':
                    end_index += 1
                train_variable.append(line[start_index, end_index].strip())
            elif model_type in line:
                r_node = ast.parse(line)
                if type(r_node.body[0]).__name__ == 'Assign':
                    for target in r_node.body[0].targets:
                        if type(target).__name__ == 'Name':
                            model_variable = target.id
                            print('model_variable',model_variable)
            elif model_variable+'.fit(' in line:
                start_index = line.find('.fit(') + 5
                end_index = start_index
                while line[end_index] != ',':
                    end_index += 1
                train_variable.append(line[start_index, end_index].strip())
        print('train_variable',train_variable)
    cursor, db = create_connection()
    sql = 'select id,model_type from result where notebook_id='+str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        model_type = row[1]
        line,train_variable = find_train_variable(notebook_id, model_type,root_path)


if __name__ == '__main__':
    # print('input this ip:')
    ip = get_host_ip()
    print('input this pair: 1: add_run=1 in aliyun; 2: add_run=2 in aliyun,..... 5: add_run=0 in other')
    ptype = input()
    if ptype == '-1':
        batch_checking(ip=ip)
        # if ip == '39.99.150.216':
        #     batch_save_running(notebook_root="../spider/notebook", dataset_root="../spider/unzip_dataset", ip=ip,
        #                        pair_type=int(ptype))
        # else:
        #     batch_save_running(notebook_root="../notebook", dataset_root="../unzip_dataset", ip=ip,
        #                        pair_type=int(ptype))

    # elif ptype == '5':
    #     print('///////')
    #     if ip == '39.99.150.216':
    #         batch_running_single(notebook_root="../spider/notebook", dataset_root="../spider/unzip_dataset", ip=ip, pair_type=int(ptype))
    #     else:
    #         batch_running_single(notebook_root="../notebook",dataset_root="../unzip_dataset",ip=ip,pair_type=int(ptype))
    else:
        if ip == '39.99.150.216':
            batch_running(notebook_root="../spider/notebook", dataset_root="../spider/unzip_dataset", ip=ip, pair_type=int(ptype))
        else:
            batch_running(notebook_root="../notebook",dataset_root="../unzip_dataset",ip=ip,pair_type=int(ptype))
    # check_no_model()
