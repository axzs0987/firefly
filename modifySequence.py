from utils import get_code_txt
from utils import CONFIG
from utils import create_connection
import numpy as np
import pymysql

def get_operator_code(notebook_id, notebook_code, change_rank, ope_dic, get_min_max=0):
    code_list = notebook_code.split('\n')
    # for index, line in enumerate(code_list):
    #     print("\033[0;35;40m" + str(index) + ':' + line + "\033[0m")
    cursor, db = create_connection()
    walk_logs = np.load('../walklogs/' + str(notebook_id) + '.npy', allow_pickle=True).item()
    sql = "select operator,data_object_value from operator where rank=" + str(change_rank) + " and notebook_id=" + str(
        notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    operation = ''
    data_object_value = ''
    est_value = ''
    for row in sql_res:
        operation = row[0]
        data_object_value = row[1]
    print('operation:', operation)
    if operation == '':
        return 'no such operator'
    if data_object_value[0] == '(' and data_object_value[-1] == ')':
        data_object_value = data_object_value[1:-1]
    print('data_object_value:', data_object_value)
    temp_data_object_value = data_object_value.replace(' ', '')
    candidate = []
    if ope_dic[operation]['call_type'] == 0 or ope_dic[operation]['call_type'] == 2 or ope_dic[operation][
        'call_type'] == 4:
        count = 0
        for i in code_list:
            if len(i) > 0:
                if i[0] == '#':
                    count += 1
                    continue
            if (data_object_value in i or temp_data_object_value in i) and operation in i:
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                candidate.append((i, count))
            count += 1
    elif ope_dic[operation]['call_type'] == 3:
        # print(walk_logs["estiminator_values"])
        est_value = walk_logs["estiminator_values"][operation]
        count = 0
        for i in code_list:
            if len(i) > 0:
                if i[0] == '#':
                    count += 1
                    continue
            if est_value in i and (data_object_value in i or temp_data_object_value in i) and (
                    'fit_transform' in i or 'transform' in i):
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                candidate.append((i, count))
                # print(operation,count)
            elif operation in i and (data_object_value in i or temp_data_object_value in i) and (
                    'fit_transform' in i or 'transform' in i):
                if temp_data_object_value in i:
                    data_object_value = temp_data_object_value
                candidate.append((i, count))
                # print(operation, count)
            count += 1

    if get_min_max == 0:
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if change_rank > 1:
                last_get = get_operator_code(notebook_id, notebook_code, change_rank - 1, ope_dic, get_min_max=1)
                if last_get != 'no such operator':
                    min = last_get[0]
                else:
                    min = [(0, 0)]
            else:
                min = [(0, 0)]
            next_get = get_operator_code(notebook_id, notebook_code, change_rank + 1, ope_dic, get_min_max=2)
            max = next_get[0]

            temp_candicate = []
            for i in candidate:
                if i[1] > min[0][1] and i[1] < max[0][1]:
                    temp_candicate.append(i)
            candidate = [temp_candicate[0]]
        return candidate, operation, data_object_value, est_value
    elif get_min_max == 1:
        # print('1_candidate:', candidate)
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if change_rank > 1:
                last_get = get_operator_code(notebook_id, notebook_code, change_rank - 1, ope_dic, get_min_max=1)
                if last_get != 'no such operator':
                    min = last_get[0]
                else:
                    min = [(0, 0)]
            else:
                min = [(0, 0)]
            temp_candicate = []
            for i in candidate:
                if i[1] > min[0][1]:
                    temp_candicate.append(i)

            candidate = [temp_candicate[0]]
        return candidate, operation, data_object_value, est_value
    elif get_min_max == 2:
        # print('2_candidate:', candidate)
        if len(candidate) > 1:
            # print('candidate:', candidate)
            if change_rank > 1:
                last_get = get_operator_code(notebook_id, notebook_code, change_rank + 1, ope_dic, get_min_max=1)
                if last_get != 'no such operator':
                    max = last_get[0]
                else:
                    max = [(0, 0)]
            else:
                max = [(0, 0)]
            temp_candicate = []
            for i in candidate:
                if i[1] < max[0][1]:
                    temp_candicate.append(i)

            candidate = [temp_candicate[-1]]
        return candidate, operation, data_object_value, est_value

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

def addOperator(notebook_id, change_rank, target_content, notebook_root_path='../notebook/'):
    """
    :param notebook_id:
    :param change_rank:
    :param target_content: {
        operation: '',
        ope_type: 1,
        parameters: [],
        data_object: 'train',
        }
    :param notebook_root_path:
    :return:
    """
    ope_dic = eval(CONFIG.get('operators', 'operations'))
    notebook_path = notebook_root_path + str(notebook_id) + '.ipynb'
    notebook_code = get_code_txt(notebook_path)
    code_list = notebook_code.split('\n')
    for index, line in enumerate(code_list):
        print("\033[0;35;40m" + str(index) + ':' + line + "\033[0m")
    # print(notebook_code)
    walklogs = np.load('../walklogs/' + str(notebook_id) + '.npy', allow_pickle=True).item()
    res = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
    if res == 'no such operator':
        return res
    #如果新增加的位置不是起始位置，那么数据对象是上一个操作的对象，否则是读入数据的对象


    res1 = get_operator_code(notebook_id, notebook_code, change_rank - 1, ope_dic)
    if res1 != 'no such operator':
        operation = res1[1]
        data_object_value = res1[2]
        call_type = ope_dic[operation]['call_type']
        if call_type == 0:
            data_object = data_object_value[0:data_object_value.find(operation) - 1]
        elif call_type == 2 or call_type == 4:
            data_object = data_object_value
        elif call_type == 3:
            data_object = data_object_value
    else:
        data_object = walklogs['data_values'][0]

    if 'data_object' in target_content.keys():
        if target_content['data_object'] != '':
            data_object = target_content['data_object']

    candidate_code_list =res[0]
    print(candidate_code_list)
    line_number = candidate_code_list[0][1]

    #生成新操作的代码
    param_code = ''
    for index, param in enumerate(target_content['parameters']):
        param_code += str(param)
        if index != len(target_content['parameters']) - 1:
            param_code += ','

    if target_content['ope_type'] == 0:
        new_code_line = data_object + '=' + data_object + '.' + target_content['operation'] + '(' + param_code + ')'
        package_code = 'import pandas as pd\n'
    elif target_content['ope_type'] == 2:
        if param_code != '':
            new_code_line = data_object + '=' + 'pd.' + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
        else:
            new_code_line = data_object + '=' + 'pd.' + target_content['operation'] + '(' + data_object + ')'
        package_code = 'import pandas as pd\n'
    elif target_content['ope_type'] == 3:
        new_code_line = data_object + '=' + target_content[
                            'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object + ')'
        if target_content['operation'] == 'SimpleImputer':
            package_code = 'from sklearn.impute import SimpleImputer\n'
        elif target_content['operation'] == 'PCA':
            package_code = 'from sklearn.decomposition import PCA\n'
        else:
            package_code = 'from sklearn.preprocessing import ' + target_content['operation'] + '\n'

    elif target_content['ope_type'] == 4:
        if target_content['operation'] == 'boxcox' or target_content['operation'] == 'boxcox1p':
            package_code = 'from scipy.stats import boxcox\n'
            package_code += 'from scipy.special import boxcox1p\n'
            if param_code != '':
                new_code_line = data_object + '=' +  target_content['operation'] + '(' + data_object + ',' + param_code + ')'
            else:
                new_code_line = data_object + '=' + target_content['operation'] + '(' + data_object + ')'
        elif target_content['operation'] == 'l2_normalize':
            prefix = 'tf.nn.'
            if param_code != '':
                new_code_line = data_object + '=' + prefix + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
            else:
                new_code_line = data_object + '=' + prefix + target_content['operation'] + '(' + data_object + ')'
            package_code = 'import tensorflow as tf'
        else:
            package_code = 'import numpy as np\n'
            alias = 'np'
            if param_code != '':
                new_code_line = data_object + '=' +  alias + '.' + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
            else:
                new_code_line = data_object + '=' +  alias + '.' + target_content[
                    'operation'] + '(' + data_object + ')'

    new_code = ''
    code_list = notebook_code.split('\n')
    for index, line in enumerate(code_list):
        if index == line_number:
            new_code += new_code_line
            new_code += '\n'
        new_code += line
        new_code += '\n'

    new_code = package_code + new_code
    return new_code

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

if __name__ == '__main__':
    notebook_id = 259277
    target_content = {
        'operation': 'boxcox1p',
        'ope_type': 4,
        'parameters': [],
        'data_object': 'y_test[\'floor\']'
    }
    # new_code = changeOperator(notebook_id,4,target_content)
    new_code = deleteOperator(notebook_id, 12)
    code_list = new_code.split('\n')
    for index,line in enumerate(code_list):
        print("\033[0;33;40m" + str(index)+':' + line + "\033[0m")
    # print(new_code)