from utils import get_code_txt
from utils import CONFIG
from utils import create_connection
from utils import check_iloc
import numpy as np
import pymysql
import pandas as pd
import pprint
import os
import ast

# from testRunning import running
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
def running_temp_code(func_def, new_path,count, found=False):
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
        namespace = {}
        exec(cm,namespace)
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
            print("\033[0;31;40merror_str\033[0m", error_str)
            print('error 2')
            foun=1
        # elif "No module named " in error_str and '_tkinter' not in error_str:
        #     package = error_str.replace("No module named ", "")
        #     package = package[1:-1]
        #     # command = ' pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0]
        #     # os.system(command)
        #     command = ' pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0] + ' --trusted-host pypi.tuna.tsinghua.edu.cn'
        #     # command = ' pip install ' + package.split('.')[0]
        #     os.system(command)
        #     print('error 3')
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
            # print("\033[0;31;40merror_str\033[0m", error_str)
            foun = 1
            print('error 5')
        elif "File b" in error_str:
            index1 = error_str.find("'")
            index2 = error_str.find("'", index1 + 1)
            error_path = error_str[index1 + 1:index2]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            # print('error_path:', error_path)
            print('error 10')
            foun = 1
            print('error 5')
        elif "'DataFrame' object has no attribute 'ix'" in error_str or "'Series' object has no attribute 'ix'" in error_str:
            new_code = func_def.replace('.ix', '.iloc')
            print('error 6')
        elif "'DataFrame' object has no attribute 'sort'" in error_str:
            new_code = func_def.replace('.sort(', '.sort_values(')
            print('error 7')
        elif "dlopen: cannot load any more object with static TLS" in error_str:
            print("\033[0;31;40merror_str\033[0m", error_str)
            return 'break'
        else:
            # print("?")
            # traceback.print_exc()

            print("\033[0;31;40merror_str\033[0m", error_str)
            print('error 8')
            return 'error 8' + error_str
        if count < 7:
            # print(new_code)
            if foun ==1:
                found = True
            code_list = new_code.split('\n')
            res = running_temp_code(new_code, new_path, count + 1,found)
            if res == 'compile fail' or res== 'False':
                return res
            if res[0:7] == 'error 8':
                return res
            # return res
        else:
            print('error 9')
            return "error 8"
    return func_def



def get_operator_code(notebook_id, notebook_code, change_rank, ope_dic, get_min_max=0):
    code_list = notebook_code.split('\n')
    # for index, line in enumerate(code_list):
    #     print("\033[0;39;41m" + str(index) + ':' + line + "\033[0m")
    cursor, db = create_connection()
    try:
        walk_logs = np.load('../walklogs/' + str(notebook_id) + '.npy', allow_pickle=True).item()
    except:
        walk_logs = []
    sql = "select operator,data_object_value,data_object from operator where rank=" + str(change_rank) + " and notebook_id=" + str(
        notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    operation = ''
    data_object_value = ''
    est_value = ''
    for row in sql_res:
        operation = row[0]
        data_object_value = row[1]
        dj = row[2]
    # print('operation:', operation)
    # print(len(operation))
    check_result = 0
    if 'iloc' in data_object_value and operation == 'iloc':
        check_result = check_iloc(data_object_value)

    # print(check_result)
    if check_result != 2 and check_result != 0:
        return [[[0,-1]]]
    if operation == '':
        return 'no such operator'
    if data_object_value[0] == '(' and data_object_value[-1] == ')':
        data_object_value = data_object_value[1:-1]

    candidate = []
    if data_object_value ==  "raw_data[(raw_data.region == 46)].sort_values(by='Date').set_index('Date').drop(columns=['AveragePrice'])":
        data_object_value = "raw_data[raw_data.region==46].sort_values(by='Date').set_index('Date').drop(columns=['AveragePrice'])"
    elif data_object_value ==  "raw_data[(raw_data.region == 46)].sort_values(by='Date').set_index('Date').drop":
        data_object_value = "raw_data[raw_data.region==46].sort_values(by='Date').set_index('Date').drop"
    # print('data_object_value:', data_object_value)
    temp_data_object_value = data_object_value.replace(' ', '')
    temp_data_object_value1 = data_object_value.replace('(- 1)','-1')
    temp_data_object_value2 = temp_data_object_value1.replace(' ','')
    # print('temp_data_object_value1:',temp_data_object_value1)
    # print('temp_data_object_value2:',temp_data_object_value2)
    if ope_dic[operation]['call_type'] == 0 or ope_dic[operation]['call_type'] == 2 or ope_dic[operation][
        'call_type'] == 4:
        count = 0
        for i in code_list:
            if len(i) > 0:
                if i[0] == '#':
                    count += 1
                    continue
            # print(operation,i)
            temp_code = i.replace('"','\'')
            temp_code = temp_code.replace(' ','')
            if (data_object_value in i or temp_data_object_value in i
                or data_object_value in i.replace('"','\'') or temp_data_object_value in i.replace('"','\'')
                or data_object_value in temp_code or temp_data_object_value in temp_code or temp_data_object_value1 in i or temp_data_object_value2 in i) and operation in i:
                # print('i:',i)
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                if temp_data_object_value1 in i:
                    data_object_value = temp_data_object_value1
                if temp_data_object_value2 in i:
                    data_object_value = temp_data_object_value2
                candidate.append((i, count))
            count += 1
    elif ope_dic[operation]['call_type'] == 3:
        # print(walk_logs["estiminator_values"])
        if "estiminator_values" in walk_logs:
            if operation in walk_logs['estiminator_values']:
                est_value = walk_logs["estiminator_values"][operation]
            else:
                est_value = ''
        else:
            est_value = ''
        # print('est_value', est_value)
        count = 0
        for i in code_list:
            if len(i) > 0:
                if i[0] == '#':
                    count += 1
                    continue
            temp_code = i.replace('"', '\'')
            temp_code = temp_code.replace(' ', '')

            if est_value in i and (data_object_value in i or temp_data_object_value in i
                or data_object_value in i.replace('"','\'') or temp_data_object_value in i.replace('"','\'')
                or data_object_value in temp_code or temp_data_object_value in temp_code or temp_data_object_value1 in i or temp_data_object_value2 in i) and (
                    'fit_transform' in i or 'transform' in i):
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                if temp_data_object_value1 in i:
                    data_object_value = temp_data_object_value1
                if temp_data_object_value2 in i:
                    data_object_value = temp_data_object_value2
                candidate.append((i, count))
                # print(operation,count)
            elif operation in i and (data_object_value in i or temp_data_object_value in i
                or data_object_value in i.replace('"','\'') or temp_data_object_value in i.replace('"','\'')
                or data_object_value in temp_code or temp_data_object_value in temp_code or temp_data_object_value1 in i or temp_data_object_value2 in i) and (
                    'fit_transform' in i or 'transform' in i):
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                if temp_data_object_value1 in i:
                    data_object_value = temp_data_object_value1
                if temp_data_object_value2 in i:
                    data_object_value = temp_data_object_value2
                candidate.append((i, count))
                # print(operation, count)
            if candidate == []:
                if i and (data_object_value in i or temp_data_object_value in i
                                       or data_object_value in i.replace('"',
                                                                         '\'') or temp_data_object_value in i.replace(
                            '"', '\'')
                                       or data_object_value in temp_code or temp_data_object_value in temp_code or temp_data_object_value1 in i or temp_data_object_value2 in i) and (
                        'fit_transform' in i or 'transform' in i):
                    if temp_data_object_value in i:
                        data_object_value = temp_data_object_value
                    if temp_data_object_value1 in i:
                        data_object_value = temp_data_object_value1
                    if temp_data_object_value2 in i:
                        data_object_value = temp_data_object_value2
                    candidate.append((i, count))
                    # print(operation,count)
            count += 1
    elif ope_dic[operation]['call_type'] == 5:
        # print(walk_logs["estiminator_values"])
        count = 0
        for i in code_list:
            if len(i) > 0:
                if i[0] == '#':
                    count += 1
                    continue
            # print(operation,i)
            temp_code = i.replace('"', '\'')
            temp_code = temp_code.replace(' ', '')
            if (data_object_value in i or temp_data_object_value in i or data_object_value in i.replace('"',
                                                                                                        '\'') or temp_data_object_value in i.replace(
                    '"',
                    '\'') or data_object_value in temp_code or temp_data_object_value in temp_code or temp_data_object_value1 in i or temp_data_object_value2 in i) and operation in i:
                # print('i:',i)
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                if temp_data_object_value1 in i:
                    data_object_value = temp_data_object_value1
                if temp_data_object_value2 in i:
                    data_object_value = temp_data_object_value2
                candidate.append((i, count))
            count += 1
    # print('???')
    # print(candidate)
    # print('min_max:',get_min_max)
    if get_min_max == 0:
        if len(candidate) > 1:
            if change_rank > 1:
                last_get = get_operator_code(notebook_id, notebook_code, change_rank - 1, ope_dic, get_min_max=1)
                if type(last_get).__name__ != 'str':
                    min = last_get[0]
                    if last_get[0][0][1] == -1:
                        min = last_get[0]
                else:
                    min = [(0, 0)]
                print('last_get:', last_get)
            else:
                min = [(0, 0)]
            next_get = get_operator_code(notebook_id, notebook_code, change_rank + 1, ope_dic, get_min_max=2)
            # print(next_get)
            if type(next_get).__name__ == 'str':
                max = [(0, 1000)]
            elif next_get[0][0][1] == -1:
                max = [(0, 1000)]
            else:
                max = next_get[0]

            print('max:', max)
            print('min:', min)
            print('candidate:', candidate)
            temp_candicate = []
            for i in candidate:
                if i[1] >= min[0][1] and i[1] <= max[0][1]:
                    temp_candicate.append(i)
            if len(temp_candicate) == 0:
                for i in candidate:
                    if i[1] <= max[0][1]:
                        temp_candicate.append(i)
            candidate = [temp_candicate[-1]]
        elif len(candidate) == 0:
            return 'no such operator'
        return candidate, operation, data_object_value, est_value,dj
    elif get_min_max == 1:
        # print('1_candidate:', candidate)
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if change_rank > 1:
                last_get = get_operator_code(notebook_id, notebook_code, change_rank - 1, ope_dic, get_min_max=1)
                print('last_get:', last_get)
                if type(last_get).__name__ != 'str':
                    min = last_get[0]
                    if last_get[0][0][1] == -1:
                        min = last_get[0]
                else:
                    min = [(0, 0)]
            else:
                min = [(0, 0)]
            temp_candicate = []
            count = 0
            for i in candidate:
                count += 1
                # print('count:', count)
                # print('min:',min)
                # print('type:min:', type(min))
                # print('i[1]:',i[1])
                if i[1] > min[0][1]:
                    temp_candicate.append(i)
            # print('len(:',len(temp_candicate))
            if len(temp_candicate) == 0:
                temp_candicate = min
            candidate = [temp_candicate[0]]
            # print('return:', candidate)
        elif len(candidate) == 0:
            return 'no such operator'
        return candidate, operation, data_object_value, est_value,dj
    elif get_min_max == 2:
        # print('2_candidate:', candidate)
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if change_rank > 1:
                last_get = get_operator_code(notebook_id, notebook_code, change_rank + 1, ope_dic, get_min_max=1)
                print('last_get:', last_get)
                if type(last_get).__name__ != 'str':
                    max = last_get[0]
                    if last_get[0][0][1] == -1:
                        max = last_get[0]
                else:
                    max = [(0, 1000)]
            else:
                max = [(0, 1000)]
            temp_candicate = []
            for i in candidate:
                if i[1] < max[0][1]:
                    temp_candicate.append(i)
            if len(temp_candicate) == 0:
                temp_candicate = max
            candidate = [temp_candicate[-1]]
        elif len(candidate) == 0:
            return 'no such operator'
        return candidate, operation, data_object_value, est_value,dj

def get_result_code(notebook_id, notebook_code, result_rank, get_min_max=0):
    def delete_error_tuple():
        cursor, db = create_connection()
        sql = "select id from result where notebook_id=" + str(notebook_id)
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        count = 1
        need_delete_id = -1
        for row in sql_res:
            if count != result_rank:
                count += 1
                continue
            need_delete_id = row[0]
            break
        if need_delete_id != -1:
            sql = 'delete from result where id=' + str(need_delete_id)
        cursor.execute(sql)
        db.commit()
    code_list = notebook_code.split('\n')
    # for index, line in enumerate(code_list):
    #     print("\033[0;35;40m" + str(index) + ':' + line + "\033[0m")
    cursor, db = create_connection()
    sql = "select code from result where notebook_id=" + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    data_object_value = ''
    est_value = ''
    count = 1

    code = ''
    for row in sql_res:
        if count != result_rank:
            count += 1
            continue
        code = row[0]
        break
    if code == '':
        return 'no such result'

    candidate = []

    count = 0
    # print(code)
    for i in code_list:
        if len(i) > 0:
            if i[0] == '#':
                count += 1
                continue
        if code in i.replace(' ',''):
            candidate.append((i, count))
        count += 1
    # print('cadidate:',candidate)
    if candidate == []:
        return 'no such result'
    if get_min_max == 0:
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if result_rank > 1:
                last_get = get_result_code(notebook_id, notebook_code, result_rank - 1, get_min_max=1)
                if last_get != 'no such result':
                    min = last_get
                else:
                    min = [(0, 0)]
            else:
                min = [(0, 0)]
            next_get = get_result_code(notebook_id, notebook_code, result_rank + 1, get_min_max=2)
            if next_get == 'no such result':
                max = [(0, 1000)]
            else:
                max = next_get

            # print('min:',min)
            # print('max:',max)
            if max[0][1] < min[0][1]:
                delete_error_tuple()
                temp_candicate = []
                for i in candidate:
                    if i[1] >= min[0][1]:
                        temp_candicate.append(i)
            else:
                temp_candicate = []
                for i in candidate:
                    # print(i)
                    if i[1] >= min[0][1] and i[1] <= max[0][1]:
                        temp_candicate.append(i)

            candidate = [temp_candicate[0]]
        return candidate
    elif get_min_max == 1:
        # print('1_candidate:', candidate)
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if result_rank > 1:
                last_get = get_result_code(notebook_id, notebook_code, result_rank - 1, get_min_max=1)
                if last_get != 'no such result':
                    min = last_get
                else:
                    min = [(0, 0)]
            else:
                min = [(0, 0)]
            temp_candicate = []
            for i in candidate:
                if i[1] > min[0][1]:
                    temp_candicate.append(i)

            if temp_candicate == []:
                return 'no such result'
            candidate = [temp_candicate[0]]
        return candidate
    elif get_min_max == 2:
        # print('2_candidate:', candidate)
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if result_rank > 1:
                last_get = get_result_code(notebook_id, notebook_code, result_rank + 1, get_min_max=1)
                if last_get != 'no such result':
                    max = last_get
                else:
                    max = [(0, 1000)]
            else:
                max = [(0, 1000)]
            temp_candicate = []
            for i in candidate:
                if i[1] < max[0][1]:
                    temp_candicate.append(i)

            if temp_candicate == []:
                return 'no such result'
            candidate = [temp_candicate[-1]]
        return candidate

def changeOperator(notebook_id, change_rank, target_content, notebook_root_path='../notebook/'):
    """

    :param notebook_id:
    :param notebook_root_path:
    :param change_rank:
    :param target_content: {
        operation: '',
        ope_type: 1,
        parameters: [],
        }
    :return:
    """
    ope_dic = eval(CONFIG.get('operators', 'operations'))
    notebook_path = notebook_root_path + str(notebook_id) + '.ipynb'
    notebook_code = get_code_txt(notebook_path)
    # print(notebook_code)
    res = get_operator_code(notebook_id,notebook_code,change_rank,ope_dic)
    if res == 'no such operator':
        return res

    candidate_code_list =res[0]
    operation =res[1]
    data_object_value =res[2]
    est_value =res[3]
    print(candidate_code_list)

    # if len(candidate_code_list) == 1:
    candidate_code = candidate_code_list[0][0]
    line_number = candidate_code_list[0][1]
    print(candidate_code)
    print(line_number)
    need_replace = ''
    data_object = ''

    call_type = ope_dic[operation]['call_type']
    if call_type == 0:
        data_object = data_object_value[0:data_object_value.find(operation)-1]
        # print(candidate_code.find(data_object_value))
        need_code =  candidate_code[candidate_code.find(data_object_value):]
        print('need_code:',need_code)
        operation_index = need_code.find(operation)
        code1 = need_code[0:operation_index]
        print("code1:",code1)
        need_code = need_code[operation_index:]
        left_index = need_code.find('(')
        ind = left_index+1
        left_count = 1
        while left_count!=0:
            if need_code[ind] == '(':
                left_count += 1
            elif need_code[ind] == ')':
                left_count -= 1
            ind += 1
        print("need_code:", need_code[0:ind])
        need_replace = code1 + need_code[0:ind]
    elif call_type == 2 or call_type == 4:
        data_object = data_object_value
        need_code_index = candidate_code.find(operation)
        head = need_code_index
        prefix = ''
        if need_code_index > 1:
            if candidate_code[need_code_index-1] == '.':
                head = need_code_index -2
                while candidate_code[head].isalnum():
                    head -= 1
                prefix = candidate_code[head+1:need_code_index]
        need_code = candidate_code[need_code_index:]
        left_index = need_code.find('(')
        ind = left_index + 1
        left_count = 1
        while left_count != 0:
            if need_code[ind] == '(':
                left_count += 1
            elif need_code[ind] == ')':
                left_count -= 1
            ind += 1
        need_replace = prefix + need_code[0:ind]
    elif call_type == 3:
        if operation in candidate_code:
            head = candidate_code.find(operation)
        elif est_value in candidate_code:
            head = candidate_code.find(operation)
        else:
            return 'no estiminator'
        need_code = candidate_code[head:]
        if 'fit_transform' in candidate_code:
            fit_index = need_code.find('fit_transform')
        elif 'transform' in candidate_code:
            fit_index = need_code.find('transform')
        else:
            return 'no transform function'
        prefix = need_code[0:fit_index]
        need_code = need_code[fit_index:]
        left_index = need_code.find('(')
        ind = left_index + 1
        left_count = 1
        while left_count != 0:
            if need_code[ind] == '(':
                left_count += 1
            elif need_code[ind] == ')':
                left_count -= 1
            ind += 1
        need_replace = prefix + need_code[0:ind]
        data_object = data_object_value

    if 'data_object' in target_content.keys():
        if target_content['data_object'] !=  '':
            data_object = target_content['data_object']

    if ('+' in data_object or '-' in data_object or '*' in data_object or '/' in data_object) \
            and not (data_object[0] == '(' and data_object[-1] == ')'):
        data_object = '(' + data_object + ')'
    if need_replace != '' and data_object != '':
        param_code = ''
        for index,param in enumerate(target_content['parameters']):
            param_code += str(param)
            if index != len(target_content['parameters'])-1:
                param_code += ','

        if target_content['ope_type'] == 0:
            new_code_line = data_object + '.' + target_content['operation'] + '(' + param_code + ')'
            package_code = 'import pandas as pd\n'
        elif target_content['ope_type'] == 2:
            if param_code != '':
                new_code_line = 'pd.' + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
            else:
                new_code_line = 'pd.' + target_content['operation'] + '(' + data_object + ')'
            package_code = 'import pandas as pd\n'
        elif target_content['ope_type'] == 3:
            new_code_line = target_content['operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object +')'
            if target_content['operation'] == 'SimpleImputer':
                package_code = 'from sklearn.impute import SimpleImputer\n'
            elif target_content['operation'] == 'PCA':
                package_code = 'from sklearn.decomposition import PCA\n'
            else:
                package_code = 'from sklearn.preprocessing import ' + target_content['operation'] + '\n'
            # param_code += 'from sklearn.preprocessing import OneHotEncoder\n'
            # param_code += 'from sklearn.preprocessing import LabelEncoder\n'
            # param_code += 'from sklearn.preprocessing import LabelBinarizer\n'
            # param_code += 'from sklearn.preprocessing import StandardScaler\n'
            # param_code += 'from sklearn.preprocessing import MinMaxScaler\n'
            # param_code += 'from sklearn.preprocessing import RobustScaler\n'
            # param_code += 'from sklearn.preprocessing import Normalizer\n'
            #

        elif target_content['ope_type'] == 4:
            if target_content['operation']  == 'boxcox' or target_content['operation']  == 'boxcox1p':
                package_code = 'from scipy.stats import boxcox\n'
                package_code += 'from scipy.special import boxcox1p\n'
                if param_code != '':
                    new_code_line =target_content['operation'] + '(' + data_object + ',' + param_code + ')'
                else:
                    new_code_line =target_content['operation'] + '(' + data_object + ')'
            elif target_content['operation'] == 'l2_normalize':
                prefix = 'tf.nn.'
                if param_code != '':
                    new_code_line =prefix + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
                else:
                    new_code_line =prefix + target_content['operation'] + '(' + data_object + ')'
                package_code = 'import tensorflow as tf'
            else:
                package_code = 'import numpy as np\n'
                alias = 'np'
                if param_code != '':
                    new_code_line = alias + '.' + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
                else:
                    new_code_line = alias + '.' + target_content[
                        'operation'] + '(' + data_object + ')'


        new_code = ''
        code_list = notebook_code.split('\n')
        replaced_line = candidate_code.replace(need_replace,new_code_line)
        for index,line in enumerate(code_list):
            if index != line_number:
                new_code += line
                new_code += '\n'
            else:
                new_code += replaced_line
                new_code += '\n'

        new_code = package_code + new_code
        print('need_replace:', need_replace)
        print('new_code:', new_code_line)
        return new_code
    else:
        return notebook_code

    # else:
    #     return notebook_code

def deleteOperator(notebook_id, change_rank, notebook_root_path='../notebook/'):
    ope_dic = eval(CONFIG.get('operators', 'operations'))
    notebook_path = notebook_root_path + str(notebook_id) + '.ipynb'
    notebook_code = get_code_txt(notebook_path)
    code_list = notebook_code.split('\n')
    for index, line in enumerate(code_list):
        print("\033[0;35;40m" + str(index) + ':' + line + "\033[0m")

    res = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
    if res == 'no such operator':
        return res
    candidate_code_list =res[0]
    print(candidate_code_list)
    line_number = candidate_code_list[0][1]

    new_code = ''
    code_list = notebook_code.split('\n')
    for index, line in enumerate(code_list):
        if index != line_number:
            new_code += line
            new_code += '\n'
    return new_code

def get_seq_from_rank(seq, notebook_id, padding=50):
    list = seq.split(',')
    seq_list = []

    ope_dic = eval(CONFIG.get('operators', 'operations'))
    for rank in list:
        sql = 'select operator from operator where notebook_id='+str(notebook_id) + ' and rank='+str(rank)
        cursor, db = create_connection()
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        operator =''
        for row in sql_res:
            operator=row[0]
            break
        one_hot_list = list(np.zeros((27,)))
        one_hot_list[ope_dic[operator]['index']-1] = 1
        seq_list.append(one_hot_list)

    len_seq = len(seq_list)
    for i in range(len_seq,padding):
        seq_list.append(list(np.zeros((27,))))
    seq_list=np.array(seq_list)
    return seq_list

def get_origin_data(notebook_id,notebook_root='../spider/notebook',dataset_root_path='../spider/unzip_dataset'):
    cursor, db = create_connection()
    sql = 'select dataset.dataSourceUrl from dataset,notebook,pair where dataset.id=pair.did and notebook.id=pair.nid and notebook.id=' + str(
        notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    file_list = []
    for row in sql_res:
        temp = "/" + row[0].split('/')[-1] + '.zip'
        file_list.append(temp)
        # break

    try:
        ct = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
    except:
        return 'no such notebook'
    code_list = ct.split('\n')

    find_fail = True
    print(file_list)

    for dataset_p in file_list:
        dataset_root_path += dataset_p
        dataset_root_path += '/'
        if not os.path.exists(dataset_root_path):
            return 'no such dataset'
        for code_txt in code_list:
            # print(code_txt)
            if 'read_csv(' in code_txt:
                r_node = ast.parse(code_txt.strip())
                print(code_txt)
                try:
                    print(type(r_node.body[0].value.args[0]))
                    if type(r_node.body[0].value.args[0]).__name__ == 'Str':
                        file_path = r_node.body[0].value.args[0].s
                        file_name = file_path.split('/')[-1]
                    elif type(r_node.body[0].value.args[0]).__name__ == 'Name':
                        file_path = r_node.body[0].value.args[0].id
                        file_name = file_path.split('/')[-1]
                    else:
                        fl = os.listdir(dataset_root_path)
                        file_name = fl[0]
                except:
                    fl = os.listdir(dataset_root_path)
                    file_name = fl[0]
                file_path = dataset_root_path + file_name

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_pickle(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_pickle(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_fwf(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_fwf(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_clipboard(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_clipboard(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            # elif 'read_json(' in code_txt:
            #     r_node = ast.parse(code_txt)
            #     for arg in r_node.body[0].value.args:
            #     file_path = r_node.body[0].value.args[0].s
            #     file_name = file_path.split('/')[-1]
            #     file_path = dataset_root_path + file_name
            #     origin_df = pd.read_json(file_path)
            #
            #     try:
            #         origin_df = pd.read_csv(file_path)
            #     except Exception as e:
            #         print(e)
            #         find_fail = False
            #
            #     if find_fail == True:
            #         break
            #     else:
            #         continue
            elif 'json_normalize(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.json_normalize(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_html(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_html(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_hdf(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_hdf(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_feather(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_feather(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_parquet(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_parquet(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_orc(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_orc(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_sas(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_sas(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_spss(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_spss(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_sql_table(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_sql_table(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            # elif 'read_sql_query(' in code_txt:
            #     r_node = ast.parse(code_txt)
            #     file_path = r_node.body[0].value.args[0].s
            #     file_name = file_path.split('/')[-1]
            #     file_path = dataset_root_path + file_name
            #     origin_df = pd.read_sql_query(file_path)
            #
            #     try:
            #         origin_df = pd.read_csv(file_path)
            #     except Exception as e:
            #         print(e)
            #         find_fail = False
            #
            #     if find_fail == True:
            #         break
            #     else:
            #         continue
            elif 'read_gbq(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_gbq(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'read_stata(' in code_txt:
                r_node = ast.parse(code_txt)
                file_path = r_node.body[0].value.args[0].s
                file_name = file_path.split('/')[-1]
                file_path = dataset_root_path + file_name
                origin_df = pd.read_stata(file_path)

                try:
                    origin_df = pd.read_csv(file_path)
                except Exception as e:
                    print(e)
                    find_fail = False

                if find_fail == True:
                    break
                else:
                    continue
            elif 'open(' in code_txt:
                index = code_txt.find('open(')
                if index != 0:
                    if code_txt[index-1] == '.':
                        continue
                try:
                    r_node = ast.parse(code_txt.strip())
                except:
                    continue
                print(code_txt)
                try:
                    print(type(r_node.body[0].value.args[0]))
                    if type(r_node.body[0].value.args[0]).__name__ == 'Str':
                        file_path = r_node.body[0].value.args[0].s
                        file_name = file_path.split('/')[-1]
                    elif type(r_node.body[0].value.args[0]).__name__ == 'Name':
                        file_path = r_node.body[0].value.args[0].id
                        file_name = file_path.split('/')[-1]
                    else:
                        fl = os.listdir(dataset_root_path)
                        file_name = fl[0]
                except:
                    fl = os.listdir(dataset_root_path)
                    file_name = fl[0]
                file_path = dataset_root_path + file_name
                if '.csv' in file_name:
                    try:
                        origin_df = pd.read_csv(file_path)
                    except Exception as e:
                        print(e)
                        find_fail = False

                    if find_fail == True:
                        break
                    else:
                        continue
            else:
                # print('no such df')
                origin_df = 'no such df'

        if type(origin_df).__name__ == 'str':
            print('no origin df')
            return 'no origin df'

        else:
            dtypes = origin_df.dtypes
            origin_num_df_list = []
            origin_cat_df_list = []
            origin_column_info = {}
            for i in range(len(dtypes)):
                if str(dtypes.values[i]) == 'int64' or str(dtypes.values[i]) == 'float64' or str(
                        dtypes.values[i]) == 'int32' \
                        or str(dtypes.values[i]) == 'float32' or str(dtypes.values[i]) == 'int' or str(
                    dtypes.values[i]) == 'float':
                    origin_num_df_list.append(dtypes.index[i])
                elif str(dtypes.values[i]) == 'str' or str(dtypes.values[i]) == 'Category':
                    origin_cat_df_list.append(dtypes.index[i])
                origin_column_info[i] = {}
                origin_column_info[i]['col_name'] = dtypes.index[i]
                origin_column_info[i]['dtype'] = str(dtypes.values[i])
                origin_column_info[i]['content'] = origin_df[dtypes.index[i]].values
                origin_column_info[i]['length'] = len(origin_df[dtypes.index[i]].values)
                origin_column_info[i]['null_ratio'] = origin_df[dtypes.index[i]].isnull().sum() / len(
                    origin_df[dtypes.index[i]].values)
                origin_column_info[i]['ctype'] = 1 if str(dtypes.values[i]) == 'int64' or str(
                    dtypes.values[i]) == 'float64' or str(dtypes.values[i]) == 'int32' \
                                                      or str(dtypes.values[i]) == 'float32' or str(
                    dtypes.values[i]) == 'int' or str(dtypes.values[i]) == 'float' else 2
                origin_column_info[i]['nunique'] = origin_df[dtypes.index[i]].nunique()
                origin_column_info[i]['nunique_ratio'] = origin_df[dtypes.index[i]].nunique() / len(
                    origin_df[dtypes.index[i]].values)
            # pprint.pprint(column_info[0])

            for column in origin_column_info:
                if origin_column_info[column]['ctype'] == 1:  # 如果是数字列
                    origin_column_info[column]['mean'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        'mean']
                    origin_column_info[column]['std'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        'std']
                    origin_column_info[column]['min'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        'min']
                    origin_column_info[column]['25%'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        '25%']
                    origin_column_info[column]['50%'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        '50%']
                    origin_column_info[column]['75%'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        '75%']
                    origin_column_info[column]['max'] = origin_df[origin_column_info[column]['col_name']].describe()[
                        'max']
                    origin_column_info[column]['median'] = origin_df[origin_column_info[column]['col_name']].median()
                    if len(origin_df[origin_column_info[column]['col_name']].mode()) == 0:
                        origin_column_info[column]['mode'] = 'NAN'
                    else:
                        origin_column_info[column]['mode'] = origin_df[origin_column_info[column]['col_name']].mode().iloc[0]
                    origin_column_info[column]['mode_ratio'] = \
                        origin_df[origin_column_info[column]['col_name']].astype('category').describe().iloc[3] / \
                        origin_column_info[column][
                            'length']
                    origin_column_info[column]['sum'] = origin_df[origin_column_info[column]['col_name']].sum()
                    origin_column_info[column]['skew'] = origin_df[origin_column_info[column]['col_name']].skew()
                    origin_column_info[column]['kurt'] = origin_df[origin_column_info[column]['col_name']].kurt()

                elif origin_column_info[column]['ctype'] == 2:  # category列
                    origin_column_info[column]['nunique'] = origin_df[origin_column_info[column]['col_name']].nunique()
                    origin_column_info[column]['unique'] = origin_df[origin_column_info[column]['col_name']].unique()
                    for item in origin_df[origin_column_info[column]['col_name']].unique():
                        # print(item)
                        temp = 0
                        for va in origin_df[origin_column_info[column]['col_name']].values:
                            if va == item:
                                temp += 1
                        origin_column_info[column][item] = temp

        # print('origin_column_info')
        # pprint.pprint(origin_column_info)
        break
    return origin_column_info

def sampling(action, notebook_id, result_id, notebook_root='../spider/notebook',dataset_root='../unzip_dataset',T=True):
    """
    :param s: s[0] = dataframe input to model, s[1] = sequence tensor, s[2] = model_id
    :param action: action = operator_name
    :return: r = [-1:1], s1 = new state
    """
    cursor, db = create_connection()
    # walk_logs = np.load('../walklogs/' + str(notebook_id) + '.npy', allow_pickle=True).item()
    sql = 'select content,sequence,model_type from result where id='+str(result_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    model_id_dic=np.load('./model_dic.npy',allow_pickle=True).item()
    seq = ''
    score = 0
    model_type = ''
    for row in sql_res:
        seq = row[1]
        score = row[0]
        model_type = row[2]
    print(model_type)
    if model_type not in model_id_dic:
        print('useless result')
        return 'useless result'

    #####get input data of model #######
    num_df_list = []
    cat_df_list = []
    column_info = {}
    file_list = os.listdir('../predf/'+ str(notebook_id) + '/')
    if str(result_id) + '.csv' in file_list:
        s_df = pd.read_csv('../predf/'+ str(notebook_id) + '/' + str(result_id) + '.csv')
        dtypes = s_df.dtypes
        for i in range(len(dtypes)):
            if str(dtypes.values[i]) == 'int64' or str(dtypes.values[i]) == 'float64' or str(dtypes.values[i]) == 'int32' \
                    or str(dtypes.values[i]) == 'float32' or str(dtypes.values[i]) == 'int' or str(dtypes.values[i]) == 'float':
                num_df_list.append(dtypes.index[i])
            elif dtypes.values[i] == 'str' or dtypes.values[i] == 'Category':
                cat_df_list.append(dtypes.index[i])
            column_info[i] = {}
            column_info[i]['col_name'] = dtypes.index[i]
            column_info[i]['dtype'] = str(dtypes.values[i])
            column_info[i]['content'] = s_df[dtypes.index[i]].values
            column_info[i]['length'] = len(s_df[dtypes.index[i]].values)
            column_info[i]['null_ratio'] = s_df[dtypes.index[i]].isnull().sum()/len(s_df[dtypes.index[i]].values)
            column_info[i]['ctype'] = 1 if str(dtypes.values[i]) == 'int64' or str(dtypes.values[i]) == 'float64' or str(dtypes.values[i]) == 'int32' \
                    or str(dtypes.values[i]) == 'float32' or str(dtypes.values[i]) == 'int' or str(dtypes.values[i]) == 'float' else 2
            column_info[i]['nunique'] = s_df[dtypes.index[i]].nunique()
            column_info[i]['nunique_ratio'] = s_df[dtypes.index[i]].nunique()/len(s_df[dtypes.index[i]].values)
        # pprint.pprint(column_info[0])

        for column in column_info:
            column_feature = []
            if column_info[column]['ctype'] == 1: #如果是数字列
                column_info[column]['mean'] = s_df[column_info[column]['col_name']].describe()['mean']
                column_info[column]['std'] = s_df[column_info[column]['col_name']].describe()['std']
                column_info[column]['min'] = s_df[column_info[column]['col_name']].describe()['min']
                column_info[column]['25%'] = s_df[column_info[column]['col_name']].describe()['25%']
                column_info[column]['50%'] = s_df[column_info[column]['col_name']].describe()['50%']
                column_info[column]['75%'] = s_df[column_info[column]['col_name']].describe()['75%']
                column_info[column]['max'] = s_df[column_info[column]['col_name']].describe()['max']
                column_info[column]['median'] = s_df[column_info[column]['col_name']].median()
                column_info[column]['mode'] = s_df[column_info[column]['col_name']].mode().iloc[0]
                column_info[column]['mode_ratio'] = s_df[column_info[column]['col_name']].astype('category').describe().iloc[3]/column_info[column]['length']
                column_info[column]['sum'] = s_df[column_info[column]['col_name']].sum()
                column_info[column]['skew'] = s_df[column_info[column]['col_name']].skew()
                column_info[column]['kurt'] = s_df[column_info[column]['col_name']].kurt()

            elif column_info[column]['ctype']==2: #category列
                column_info[i]['mean'] = 0
                column_info[i]['std'] = 0
                column_info[i]['min'] = 0
                column_info[i]['25%'] = 0
                column_info[i]['50%'] = 0
                column_info[i]['75%'] = 0
                column_info[i]['max'] = 0
                column_info[i]['median'] = 0
                column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0
                column_info[i]['sum'] = 0
                column_info[i]['skew'] = 0
                column_info[i]['kurt'] = 0
                # column_info[column]['unique'] = s_df[column_info[column]['col_name']].unique()
                # for item in s_df[column_info[column]['col_name']].unique():
                #     temp1 = [x for i, x in enumerate(s_df[column_info[column]['col_name']]) if
                #              s_df[column_info[column]['col_name']].iat[0, i] == item]
                #     column_info[column][item] = len(temp1)
            for key in column_info[column]:
                if key != 'col_name':
                    column_feature[key].append(column_info[column][key])
            # break
    elif str(result_id) + '.npy' in file_list:
        inp_data = np.load('../predf/' + str(notebook_id) + '/' + str(result_id) + '.npy').T.tolist()
        for i,col in enumerate(inp_data):
            s_s = pd.Series(col)
            if str(s_s.dtypes) == 'int64' or str(s_s.dtypes) == 'float64' or str(s_s.dtypes) == 'int32' \
                    or str(s_s.dtypes) == 'float32' or str(s_s.dtypes) == 'int' or str(s_s.dtypes) == 'float':
                num_df_list.append('unknown_'+str(i))
            elif str(s_s.dtypes) == 'int64' == 'str' or str(s_s.dtypes)  == 'Category':
                cat_df_list.append('unknown_'+str(i))
            column_info[i] = {}
            column_info[i]['col_name'] = 'unknown_'+str(i)
            column_info[i]['dtype'] = str(s_s.dtypes)
            column_info[i]['content'] = s_s.values
            column_info[i]['length'] = len(s_s.values)
            column_info[i]['null_ratio'] = s_s.isnull().sum()/len(s_s.values)
            column_info[i]['ctype'] = 1 if str(s_s.dtypes) == 'int64' or str(s_s.dtypes) == 'float64' or str(s_s.dtypes) == 'int32' \
                    or str(s_s.dtypes) == 'float32' or str(s_s.dtypes) == 'int' or str(s_s.dtypes) == 'float' else 2
            column_info[i]['nunique'] = s_s.nunique()
            column_info[i]['nunique_ratio'] = s_s.nunique()/len(s_s.values)

            if column_info[i]['ctype'] == 1: #如果是数字列
                column_info[i]['mean'] = s_s.describe()['mean']
                column_info[i]['std'] = s_s.describe()['std']
                column_info[i]['min'] = s_s.describe()['min']
                column_info[i]['25%'] = s_s.describe()['25%']
                column_info[i]['50%'] = s_s.describe()['50%']
                column_info[i]['75%'] = s_s.describe()['75%']
                column_info[i]['max'] = s_s.describe()['max']
                column_info[i]['median'] = s_s.median()
                column_info[i]['mode'] = s_s.mode().iloc[0]
                column_info[i]['mode_ratio'] = s_s.astype('category').describe().iloc[3]/column_info[i]['length']
                column_info[i]['sum'] = s_s.sum()
                column_info[i]['skew'] = s_s.skew()
                column_info[i]['kurt'] = s_s.kurt()

            elif column_info[i]['ctype']==2: #category列
                column_info[i]['mean'] = 0
                column_info[i]['std'] = 0
                column_info[i]['min'] = 0
                column_info[i]['25%'] = 0
                column_info[i]['50%'] = 0
                column_info[i]['75%'] = 0
                column_info[i]['max'] = 0
                column_info[i]['median'] = 0
                column_info[i]['mode'] = 0
                column_info[i]['mode_ratio'] = 0
                column_info[i]['sum'] = 0
                column_info[i]['skew'] = 0
                column_info[i]['kurt'] = 0
                # for item in s_s.unique():
                #     temp1 = [x for i, x in enumerate(s_s) if s_s.iat[0, i] == item]
                #     column_info[i][item] = len(temp1)

    ####load origin dataset#####

    # print(dtypes.index)
    # origin_code = get_code_txt(notebook_root + '/' + notebook_id + '.ipynb')

    # if action[0] == 'Add':
    #     data_object = get_data_object(result_id, action[1])
    #     target_content = {
    #         'operation': action[2],
    #         'ope_type': 1,
    #         'parameters': [],
    #         'data_object': 'train',
    #     }
    #     new_code = addOperator(notebook_id, action[1], target_content)
    # elif action[0] == 'Update':
    #     data_object = get_data_object(result_id, action[1])
    #     target_content = {
    #         'operation': action[2],
    #         'ope_type': 1,
    #         'parameters': [],
    #         'data_object': 'train',
    #     }
    #     new_code = changeOperator(notebook_id, action[1], target_content)
    # elif action[0] == 'Delete':
    #     new_code = deleteOperator(notebook_id, action[1])
    #
    # run_result = changed_running()
    #

def stat_colnum_and_uniques(ip,notebook_root='../spider/notebook',dataset_root_path='../spider/unzip_dataset'):
    in_result = []
    cursor, db = create_connection()
    sql = 'select distinct notebook_id from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(int(row[0]))

    sql = "select distinct pair.nid,pair.did from pair,dataset where pair.did=dataset.id and dataset.server_ip='" + ip +"' and dataset.isdownload=1"
    print(sql)
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    count = 0
    col_num_sum = 0
    cat_sum = 0
    has_print = []
    max_col_num = 0
    max_unique_num = 0
    sum_unique_ratio=0
    max_unique_ratio = 0
    max_length = 0
    sum_length = 0
    for row in sql_res:

        # print(count)
        if row[1] not in has_print:
            # print('id:',row[1])
            has_print.append(row[1])
        else:
            continue
        # notebook_id=int(row[0])
        file_list = os.listdir('../origindf/')
        #
        # if str(row[1])+'.npy' in file_list:
        #     # print("already in")
        #     continue
        # if notebook_id not in in_result:
        #     continue
        if str(row[1])+'.npy' in file_list:
            print(count)
            count += 1
            origin_column_info = np.load('../origindf/' + str(row[1])+'.npy',allow_pickle=True).item()
        else:
            continue
        # if origin_column_info == 'no origin df':
        #     continue
        # np.save('../origindf/' + str(row[1])+'.npy', origin_column_info)
        # print(origin_column_info)
        if len(origin_column_info) > max_col_num:
            max_col_num = len(origin_column_info)
        for col in origin_column_info:
            if origin_column_info[col]['ctype'] == 2:
                print('nunique:',origin_column_info[col]['nunique'])
                cat_sum += origin_column_info[col]['nunique']
                if origin_column_info[col]['nunique'] > max_unique_num:
                    max_unique_num = origin_column_info[col]['nunique']

        col_num_sum += len(origin_column_info)

        sum_unique_ratio += origin_column_info[col]['nunique_ratio']
        if origin_column_info[col]['nunique_ratio'] > max_unique_ratio:
            max_unique_ratio = origin_column_info[col]['nunique_ratio']

        sum_length += origin_column_info[col]['length']
        if origin_column_info[col]['length'] > max_length:
            max_length = origin_column_info[col]['length']
        # cat_num_sum += col_num_sum

    # print('count:', count)
    if count == 0:
        return
    else:
        print('mean_col_num:',col_num_sum/count)
        print('max_col_num:', max_col_num)
        print('mean_uniques:', cat_sum/col_num_sum)
        print('max_unique_num:', max_unique_num)
        print('mean_uniques_ratio:', sum_unique_ratio / col_num_sum)
        print('max_unique_ratio:', max_unique_ratio)
        print('mean_length:', sum_length / col_num_sum)
        print('max_length:', max_length)
    print(has_print)

def save_origin_df(ip,notebook_root='../spider/notebook',dataset_root_path='../spider/unzip_dataset'):
    in_result = []
    cursor, db = create_connection()
    sql = 'select distinct notebook_id from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(int(row[0]))

    sql = "select distinct pair.nid,pair.did from pair,dataset where pair.did=dataset.id and dataset.server_ip='" + ip + "' and dataset.isdownload=1"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    has_checked = []
    nod = 0
    non = 0
    nsd = 0
    for row in sql_res:
        file_list = os.listdir('../origindf/')

        if row[1] in has_checked:
            continue
        has_checked.append(row[1])
        if str(row[1])+'.npy' in file_list:
            # print("already in")
            continue

        print('dataset_id:', row[1])
        notebook_id= row[0]
        origin_column_info = get_origin_data(notebook_id,notebook_root,dataset_root_path)

        if origin_column_info == 'no origin df':
            print('no origin df')
            nod += 1
            continue
        if origin_column_info == 'no such notebook':
            non += 1
            print('no such notebook')
            continue
        if origin_column_info == 'no such dataset':
            nsd += 1
            print('no such dataset')
            continue

        np.save('../origindf/' + str(row[1])+'.npy', origin_column_info)
    print('nod:',nod)
    print('non:',non)

def get_model_dic():
    cursor, db = create_connection()
    sql = 'select distinct model_type, count(distinct notebook_id) from result group by model_type'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    model_dic = {}
    id=1
    for row in sql_res:
        if row[1]<10 or row[0]=='str' or row[0]=='unknown' or row[0]=='list' or row[0]=='Pipeline' or row[0]=='cross_val_predict':
            continue
        model_dic[row[0]] = id
        id+=1
    pprint.pprint(model_dic)
    np.save('./model_dic', model_dic)

if __name__ == '__main__':

    dataset_name = ''
    cursor, db = create_connection()
    sql = 'select dataSourceUrl from pair,dataset where pair.did=dataset.id and pair.nid=7835272'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        if row[0] == 'None' or row[0] == None:
            continue
        dataset_name = row[0].split('/')[-1]
        break
    print(dataset_name)
    # get_model_dic()
    # sampling('Add',103681,173671,notebook_root='../notebook')
    #
    # ip = '10.77.70.128'
    # if ip != '39.99.150.216':
    #     notebook_root = '../notebook'
    #     dataset_root = '../unzip_dataset'
    #     save_origin_df(ip,notebook_root=notebook_root,dataset_root_path=dataset_root)
    # else:
    #     save_origin_df(ip)


    # notebook_id = 16869
    # target_content = {
    #     'operation': 'boxcox1p',
    #     'ope_type': 4,
    #     'parameters': [],
    #     'data_object': 'y_test[\'floor\']'
    # }
    # # new_code = changeOperator(notebook_id,4,target_content)
    # new_code = deleteOperator(notebook_id, 12,notebook_root_path='../spider/notebook/')
    # code_list = new_code.split('\n')
    # for index,line in enumerate(code_list):
    #     print("\033[0;33;40m" + str(index)+':' + line + "\033[0m")
    # # print(new_code)