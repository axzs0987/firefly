
from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run
from notebook2sequence import single_running
from utils import get_params_code_by_id
from utils import CONFIG
from utils import find_special_notebook
from utils import get_pair
from utils import get_compile_fail_pair
from utils import update_db
import numpy as np
import os
import re
import traceback

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
        if line[count:count+3] == 'if ' and ':' in line or line[count:count+4] == 'for ' and ':' or line[count:count+6] == 'while ' and ':' or line[count:count+4] == 'def ' and ':':
            count += 4
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
                target = converse_target(code_list[index_list[index]-1] , target_line)
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
            target = converse_target(code_list[under_line-1], target_line)
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


def running(func_def, new_path,count, found=False):
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
    except Exception as e:
        traceback.print_exc()
        error_str = str(e)
        new_code = func_def
        foun = 0
        # print("\033[0;31;40merror_str\033[0m", error_str)
        if "[Errno 2] No such file or directory: " in error_str:
            error_path = error_str.replace("[Errno 2] No such file or directory: " , "")
            error_path = error_path[1:-1]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            foun=1
            # running(new_code)
        elif "does not exist:" in error_str and '[Errno 2] File ' in error_str:
            error_path = error_str.split(':')[-1].strip()
            error_path = error_path[1:-1]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            # print('new_code:', new_code)
            foun=1
        elif "No module named " in error_str and '_tkinter' not in error_str:
            package = error_str.replace("No module named ", "")
            package = package[1:-1]
            command = ' pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0]
            os.system(command)
        elif  ": No such file or directory" in error_str:
            index1 = error_str.find("'")
            index2 = error_str.find("'", index1+1)
            error_path = error_str[index1+1:index2]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
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
        else:
            # print("?")
            traceback.print_exc()
            print("\033[0;31;40merror_str\033[0m", error_str)
            return "False"
        if count < 2 and can_run==False and found==False:
            # print(new_code)
            if foun ==1:
                found = True
            res = running(new_code, new_path, count + 1,found)
            return res

    return func_def

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
    print("weewrwL:", origin_code)
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
    func_def += "def insert_result(type, content, code, model_type):\n"
    func_def += "    notebook_id = " + str(notebook_id) + '\n'
    func_def += "    add_result(notebook_id, type, content, code, model_type)\n"


    this_walk_logs = np.load(walk_logs_path + '/' + str(notebook_id) + '.npy',allow_pickle=True).item()
    origin_code = func_def + origin_code

    model_dic  = eval(CONFIG.get('models', 'model_dic'))
    model_result_log = {}

    line = 0
    model_pred = this_walk_logs['models_pred']

    while(line < len(origin_code.split('\n'))):
        code_list = origin_code.split('\n')
        code = code_list[line]
        # print(code)
        if '.score(' in code:
            index = code.find('.score(')
            head = index - 1
            # print("index:",index)
            while code[head].isalpha() or code[head] == '_' \
                    or code[head] == '[' or code[head] == ']' \
                    or code[head] == '\'' or code[head] == '\"' \
                    or code[head].isalnum():
                if code[head] == ']':
                    while code[head] != '[':
                        head-=1
                else:
                    head -= 1
            # print("head:", head)
            left_index =  index + 6
            left_num = 1
            # print("left_index:", left_index)
            right_index = 0
            for ind in range(left_index + 1, len(code)):
                if code[ind] == '(':
                    left_num += 1
                elif code[ind] == ')':
                    left_num -= 1
                if left_num == 0:
                    right_index = ind
                    break
            # print("right_index:", right_index)
            model_id = -1
            for item in model_pred:
                if model_pred[item] in code:
                    model_id = model_dic[item]
                elif item in code:
                    model_id = model_dic[item]
            # print(code[head+1:right_index + 1])
            temp_code = code
            temp_code = temp_code.replace(' ','')
            temp_code = temp_code.replace('\t', '')
            temp_code = temp_code.replace('\'', '\\\'')
            temp_code = temp_code.replace('"', '\\"')
            need2print = "insert_result(" + str(model_id) + "," + code[head+1:right_index + 1] +',"' + temp_code + '", "score")'
            # print(need2print)
            line += 1
            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            # param = code[left_index:right_index+1]

            # if 'score' not in model_result_log:
            #     model_result_log['score'] = []
            #
            # if param not in model_result_log['score']:
            #     line += 1
            #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
            #     model_result_log['score'].append(param)
        elif 'best_score_' in code:
            index = code.find('best_score_')
            head = index - 2
            print("index:",index)
            print(code)
            while code[head].isalpha() or code[head] == '_' \
                    or code[head] == '[' or code[head] == ']' \
                    or code[head] == '\'' or code[head] == '\"' \
                    or code[head].isalnum():
                if code[head] == ']':
                    while code[head] != '[':
                        head-=1
                else:
                    head -= 1
            print('head:', head)
            right_index = index + 10
            # left_index
            # print("head:", head)
            # left_index =  index + 6
            # left_num = 1
            # # print("left_index:", left_index)
            # right_index = 0
            # for ind in range(left_index + 1, len(code)):
            #     if code[ind] == '(':
            #         left_num += 1
            #     elif code[ind] == ')':
            #         left_num -= 1
            #     if left_num == 0:
            #         right_index = ind
            #         break
            # print("right_index:", right_index)
            model_id = -1
            for item in model_pred:
                if model_pred[item] in code:
                    model_id = model_dic[item]
                elif item in code:
                    model_id = model_dic[item]

            # print(code[head+1:right_index + 1])
            temp_code = code
            temp_code = temp_code.replace(' ','')
            temp_code = temp_code.replace('\t', '')
            temp_code = temp_code.replace('\'', '\\\'')
            temp_code = temp_code.replace('"', '\\"')
            need2print = "insert_result(" + str(model_id) + "," + code[head+1:right_index + 1] +',"' + temp_code + '", "best_score_")'

            line += 1
            origin_code = insert_one_line_in_code(origin_code, line, need2print)
        elif 'accuracy_score(' in code:
            index = code.find('accuracy_score(')
            if index == 0:
                head = index
                left_index = index  + 14
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
                need2print = "insert_result(" + str(model_id) + "," + code[head:right_index + 1] + ',"' + temp_code + '", "accuracy_score")'
                param = code[left_index :right_index + 1]
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # if 'accuracy_score' not in model_result_log:
                #     model_result_log['accuracy_score'] = []
                #
                # if param not in model_result_log['accuracy_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['accuracy_score'].append(param)
            elif code[index-1] != '.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index
                left_index = index + 14
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head:right_index + 1] + ',"' + temp_code + '", "accuracy_score")'

                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)

            else:
                head = index - 2
                # print("index:",index)
                while code[head].isalpha() or code[head] == '_' \
                        or code[head] == '[' or code[head] == ']' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum():
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    else:
                        head -= 1
                left_index = index + 14
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head+1:right_index + 1] + ',"' + temp_code + '", "accuracy_score")'
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)


        elif 'auc(' in code:
            index = code.find('auc(')
            if index == 0:
                head = index
                left_index = index + 3
                # print("left_index:", left_index)
                left_num = 1


                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[head:right_index + 1] + ',\'' + temp_code + '\', "auc")'
                # print(need2print)
                param = code[left_index :right_index + 1]
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # if 'auc' not in model_result_log:
                #     model_result_log['auc'] = []
                #
                # if param not in model_result_log['auc']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['auc'].append(param)
            elif code[index-1]!='.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index - 2
                left_index = index + 3
                # print("left_index:", left_index)
                left_num = 1

                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head + 1:right_index + 1] + ',\'' + temp_code + '\', "auc")'
                # print(need2print)
                param = code[left_index:right_index + 1]
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
            else:
                head = index - 2
                while code[head].isalpha() or code[head] == '_' \
                        or code[head] == '[' or code[head] == ']' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum():
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    else:
                        head -= 1

                left_index = index + 3
                # print("left_index:", left_index)
                left_num = 1

                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head + 1:right_index + 1] + ',\'' + temp_code + '\', "auc")'
                # print(need2print)
                param = code[left_index:right_index + 1]
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
        elif 'f1_score(' in code:
            index = code.find('f1_score(')
            if index == 0:
                head = index
                # print("index:",index)
                left_index = index + 8
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[head:right_index + 1] + ',\'' + temp_code + '\', "auc")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index :right_index + 1]
                # if 'f1_score' not in model_result_log:
                #     model_result_log['f1_score'] = []
                #
                # if param not in model_result_log['f1_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['f1_score'].append(param)
            elif code[index-1] != '.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index
                # print("index:",index)
                left_index = index + 8
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head :right_index + 1] + ',\'' + temp_code + '\', "auc")'
                # print(need2print)
                # param = code[left_index:right_index + 1]
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # if 'f1_score' not in model_result_log:
                #     model_result_log['f1_score'] = []
                #
                # if param not in model_result_log['f1_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['f1_score'].append(param)
            else:
                head = index - 2
                # print("index:",index)
                while code[head].isalpha() or code[head] == '_' \
                        or code[head] == '[' or code[head] == ']' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum():
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    else:
                        head -= 1
                # print("index:",index)
                left_index = index + 8
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head + 1:right_index + 1] + ',\'' + temp_code + '\', "f1_score")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index:right_index + 1]
                # if 'f1_score' not in model_result_log:
                #     model_result_log['f1_score'] = []
                #
                # if param not in model_result_log['f1_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['f1_score'].append(param)
        elif 'r2_score(' in code:
            index = code.find('r2_score(')
            if index == 0:
                head = index
                # print("index:",index)
                left_index = index + 8
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head:right_index + 1] + ',\'' + temp_code + '\', "r2_score")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index:right_index + 1]
                # if 'r2_score' not in model_result_log:
                #     model_result_log['r2_score'] = []
                #
                # if param not in model_result_log['r2_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['r2_score'].append(param)
            elif code[index - 1] != '.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index
                # print("index:",index)
                left_index = index + 8
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head:right_index + 1] + ',\'' + temp_code + '\', "r2_score")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index:right_index + 1]
                # if 'r2_score' not in model_result_log:
                #     model_result_log['r2_score'] = []
                #
                # if param not in model_result_log['r2_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['r2_score'].append(param)
            else:
                head = index - 2
                # print("index:",index)
                left_index = index + 8
                # print("left_index:", left_index)
                left_num = 1
                while code[head].isalpha() or code[head] == '_' \
                        or code[head] == '[' or code[head] == ']' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum():
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    else:
                        head -= 1
                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head + 1:right_index + 1] + ',\'' + temp_code + '\', "r2_score")'
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # print(need2print)
                # param = code[left_index:right_index + 1]
                # if 'r2_score' not in model_result_log:
                #     model_result_log['r2_score'] = []
                #
                # if param not in model_result_log['r2_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['r2_score'].append(param)
        elif 'cross_val_score(' in code:
            index = code.find('cross_val_score(')
            if index == 0:
                head = index
                # print("index:",index)
                left_index = index + 15
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head:right_index + 1]+'.mean()' + ',\'' + temp_code + '\', "cross_val_score")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index:right_index + 1]
                # if 'cross_val_score' not in model_result_log:
                #     model_result_log['cross_val_score'] = []
                #
                # if param not in model_result_log['cross_val_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['cross_val_score'].append(param)
            elif code[index - 1] != '.':
                if code[index -1].isalpha() or code[index-1] == '_' \
                        or code[index-1].isalnum():
                    line += 1
                    continue
                head = index
                # print("index:",index)
                left_index = index + 15
                # print("left_index:", left_index)
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
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[head:right_index + 1]+'.mean()' + ',\'' + temp_code + '\', "cross_val_score")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index:right_index + 1]
                # if 'cross_val_score' not in model_result_log:
                #     model_result_log['cross_val_score'] = []
                #
                # if param not in model_result_log['cross_val_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['cross_val_score'].append(param)
            else:
                head = index -2
                # print("index:",index)
                left_index = index + 15
                # print("left_index:", left_index)
                left_num = 1
                while code[head].isalpha() or code[head] == '_' \
                        or code[head] == '[' or code[head] == ']' \
                        or code[head] == '\'' or code[head] == '\"' \
                        or code[head].isalnum():
                    if code[head] == ']':
                        while code[head] != '[':
                            head -= 1
                    else:
                        head -= 1
                right_index = 0
                for ind in range(left_index + 1, len(code)):
                    if code[ind] == '(':
                        left_num += 1
                    elif code[ind] == ')':
                        left_num -= 1
                    if left_num == 0:
                        right_index = ind
                        break
                # print("right_index:", right_index)
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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head + 1:right_index + 1]+'.mean()' + ',\'' + temp_code + '\', "cross_val_score")'
                # print(need2print)
                line += 1
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                # param = code[left_index:right_index + 1]
                # if 'cross_val_score' not in model_result_log:
                #     model_result_log['cross_val_score'] = []
                #
                # if param not in model_result_log['cross_val_score']:
                #     line += 1
                #     origin_code = insert_one_line_in_code(origin_code, line, need2print)
                #     model_result_log['cross_val_score'].append(param)
        line += 1
    # print(origin_code)
    return origin_code

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
    model_dic = model_dic = eval(CONFIG.get('models', 'model_dic'))
    for id in no_model_notebook_id_list:
        origin_code = get_code_txt(root_path + '/' + str(id) + '.ipynb')
        for model_name in model_dic:
            if model_name + '(' in origin_code:
                print(str(id)+ ":found model!", model_name)

def test_running(notebook_root, dataset_root, notebook_name, dataset_name):
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
    can_run_code = running(func_def,dataset_path_root,0)
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
    can_run_code = running(add_varible_code_content, dataset_path_root, 0)

def single_runnings(notebook_id, dataset_name ,notebook_root="../notebook",dataset_root="../dataset"):
    dataset_path_root = dataset_root + '/' + dataset_name + '.zip'
    try:
        origin_code = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
    except:
        print("\033[0;31;read fail\033[0m")
        return "read fail"

    add_code = add_result(notebook_id, origin_code)
    add_code = add_params_miresult(notebook_id, add_code)

    add_code = add_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer', 'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.externals import joblib','import joblib')
    add_code = add_code.replace('pandas.tools', 'pandas')
    add_code = insert_one_line_in_code(add_code,'import matplotlib.pyplot as plt','matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')
    for index,i in enumerate(add_code.split('\n')):
        print(index, i)

    can_run_code = running(add_code,dataset_path_root,0)
    if can_run_code == "compile fail":
        print("\033[0;31;40mcompile fail\033[0m")
        return "compile fail"
    if can_run_code == 'False':
        return "False"
    return 'succeed'



def batch_running(notebook_root="../notebook",dataset_root="../dataset",ip="39.99.150.216"):
    pairs = get_compile_fail_pair(ip)
    print(len(pairs))
    count = 0
    for pair in pairs:
        # try:
        # print(pair[0])
        # print(type(pair[0]).__name__)
        if count == 8:
        # if pair[0] == '114886':
        #     print("///")
            notebook_id = pair[0]
            dataset_name = pair[1]

            # if notebook_name == "most-dangerous-departure-and-destination-cities.ipynb":

            print("\033[0;33;44m" + str(notebook_id) + "\033[0m")
            res = single_runnings(notebook_id, dataset_name, notebook_root, dataset_root)

            if res != 'False' and res != 'compile fail' and res != 'read fail':
                update_db("notebook", "add_run", '1', 'id', '=', notebook_id)
        #     if res == 'compile fail':
        #         update_db("notebook", "add_run", '2', 'id', '=', notebook_id)
        #     if res == 'False':
        #         update_db("notebook", "add_run", '3', 'id', '=', notebook_id)
        #     if res == 'read fail':
        #         update_db("notebook", "add_run", '0', 'id', '=', notebook_id)
        count += 1
        #




if __name__ == '__main__':
    batch_running(notebook_root="../spider/notebook",dataset_root="../spider/unzip_dataset")
    # check_no_model()
