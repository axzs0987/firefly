import numpy as np
import pandas as pd
import traceback
import ast
import astunparse
import os

from utils import CONFIG
from utils import create_connection

from modifySequence import get_operator_code
from modifySequence import get_result_code

from testRunning import get_model_from_code
from testRunning import get_model_from_error_param_code
from testRunning import found_dataset
from testRunning import insert_one_line_in_code
from testRunning import get_code_txt
from modifySequence import running_temp_code
from testRunning import add_changed_result

from utils import get_host_ip
import grpc
import message_pb2
import message_pb2_grpc
from concurrent import futures
import time

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

if not os.path.exists('./error_str_dic.npy'):
    error_str_dic = {}
else:
    error_str_dic = np.load('./error_str_dic.npy',allow_pickle=True).item()

def get_read_code(notebook_code):
    code_list = notebook_code.split('\n')
    target_value = ''
    for index,code_txt in enumerate(code_list):
        if 'read_csv' in code_txt or 'read_pickle' in code_txt or 'read_table' in code_txt or 'read_fwf' in code_txt or \
                'read_clipboard' in code_txt or 'read_excel' in code_txt or 'ExcelFile.parse' in code_txt or 'ExcelWriter' in code_txt or \
                'read_json' in code_txt or 'json_normalize' in code_txt or 'build_table_schema' in code_txt or 'read_html' in code_txt or \
                'read_hdf' in code_txt or 'read_feather' in code_txt or 'read_parquet' in code_txt or 'read_orc' in code_txt or \
                'read_sas' in code_txt or 'read_spss' in code_txt or 'read_sql_table' in code_txt or 'read_sql_query' in code_txt or \
                'read_sql' in code_txt or 'read_gbq' in code_txt or 'read_stata' in code_txt:
            r_node = ast.parse(code_txt.strip())
            if type(r_node.body[0]).__name__ == 'Assign':
                if type(r_node.body[0].targets[0]).__name__ == 'Name':
                    target_value = r_node.body[0].targets[0].id
                    print(target_value,index)
                    return target_value,index
    return 'no read data'



def addOperator(notebook_id,notebook_code,target_content, mid_line_number, notebook_root_path='../notebook/',dataset_root='../unzip_dataset/'):
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
    cursor, db = create_connection()
    sql = 'select operator,rank from operator where notebook_id=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    change_rank=0

    result_code_line = get_result_code(notebook_id, notebook_code, 1, get_min_max=0)
    if type(result_code_line).__name__ == 'str':
        result_code_line = 10000
    else:
        result_code_line = result_code_line[0][1]

    # print('result_code_line:',result_code_line)
    for row in sql_res:
        change_rank=row[1]

    ope_dic = eval(CONFIG.get('operators', 'operations'))
    # print('type(operator_code_line).__name__:zzzz')
    operator_code_line = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
    # print('type(operator_code_line).__name__:xxxxx')
    # print('type(operator_code_line).__name__:',type(operator_code_line).__name__)
    if type(operator_code_line).__name__ == 'str':
        change_rank = 0
    else:
        print('operator_code_line:', operator_code_line)
        # print('operator_code_line:', operator_code_line[0][0][1])
        operator_code_line = operator_code_line[0][0][1]
        # print(operator_code_line)
        while change_rank > 0 and (operator_code_line > result_code_line or operator_code_line == -1):
            # print(operator_code_line)
            change_rank -= 1
            operator_code_line = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
            if operator_code_line == 'no such operator':
                continue
            operator_code_line=operator_code_line[0][0][1]
        if operator_code_line == 'no such operator':
            change_rank = 0

    code_list = notebook_code.split('\n')
    for index, line in enumerate(code_list):
        print("\033[0;35;40m" + str(index) + ':' + line + "\033[0m")


    if mid_line_number == -1:
        if change_rank != 0:
            res = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
            # 如果新增加的位置不是起始位置，那么数据对象是上一个操作的对象，否则是读入数据的对象
            # res1 = get_operator_code(notebook_id, notebook_code, change_rank - 1, ope_dic)
            # print('get_operator_code:', res)
            code = res[0][0][0]
            r_node = ast.parse(code.strip())
            operation = res[1]
            call_type = ope_dic[operation]['call_type']
            # print('call_type:', call_type)
            # print('type(r_node.body[0]).__name__', type(r_node.body[0]).__name__)
            if type(r_node.body[0]).__name__ == 'Assign':
                for target in r_node.body[0].targets:
                    if type(target).__name__ == 'Name':
                        data_object = target.id
                        break
                    elif type(target).__name__ == 'Subscript':
                        # print(type(r_node.body[0].targets[0].value).__name__)
                        if type(target.value).__name__ == 'Name':
                            data_object = target.value.id
                        else:
                            data_object = astunparse.unparse(target.value)
                            if data_object[-1] == '\n':
                                data_object = data_object[0:-1]
                    else:
                        data_object = astunparse.unparse(target)
                        if data_object[-1] == '\n':
                            data_object = data_object[0:-1]
                        break
            else:
                data_object_value = res[2]
                if call_type == 0:
                    data_object = data_object_value[0:data_object_value.find(operation) - 1]
                elif call_type == 2 or call_type == 4:
                    data_object = data_object_value
                elif call_type == 3:
                    data_object = data_object_value

            candidate_code_list = res[0]
            # print(candidate_code_list)
            line_number = candidate_code_list[0][1] + 1
        else:
            res = get_read_code(notebook_code)
            if res == 'no read data':
                return notebook_code,notebook_code,notebook_code,-2
            data_object = res[0]
            line_number = res[1] + 1
    else:
        # print('mid_line_number',mid_line_number-1)
        # print(code_list[mid_line_number-1])
        r_node = ast.parse(code_list[mid_line_number-1].strip())
        # print(type(r_node.body[0].targets[0]).__name__)
        if type(r_node.body[0].targets[0]).__name__ == 'Name':
            data_object = r_node.body[0].targets[0].id
        elif type(r_node.body[0].targets[0]).__name__ == 'Subscript':
            # print(type(r_node.body[0].targets[0].value).__name__)
            if type(r_node.body[0].targets[0].value).__name__ == 'Name':
                data_object = r_node.body[0].targets[0].value.id
            else:
                data_object = astunparse.unparse(r_node.body[0].targets[0].value)
                if data_object[-1] == '\n':
                    data_object = data_object[0:-1]
        else:
            data_object = astunparse.unparse(r_node.body[0].targets[0])
            if data_object[-1] == '\n':
                data_object = data_object[0:-1]
        line_number = mid_line_number


    if 'data_object' in target_content.keys():
        data_object1 = data_object
        data_object2 = data_object
        if target_content['data_object'] != -1:
            data_object2 = data_object + '[' + str(target_content['data_object']) + ']'
            data_object = data_object + '.iloc[:, ' + str(target_content['data_object'])+ ']'


    #生成新操作的代码
    param_code = ''
    for index, param in enumerate(target_content['parameters']):
        if type(param).__name__ == 'str' and param != 'np.nan':
            param_code += '"'
            param_code += str(param)
            param_code += '"'
        else:
            param_code += str(param)
        if index != len(target_content['parameters']) - 1:
            param_code += ','
    new_line_0 = ''
    new_line_1 = ''
    new_line_2 = ''
    new_line_0_0 = ''
    new_line_0_1 = ''
    new_line_3 = ''
    new_line_3_1 = ''
    new_line_4 = ''
    new_line_5 = ''

    package_line_number = 0
    if target_content['ope_type'] == 0:
        if target_content['operation'] == 'drop':
            new_code_line = data_object1 + '=' + data_object1 + '.drop(index=' + str(target_content['data_object']) + ')'
        elif target_content['operation'] == 'dropna':
            new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
            new_code_line = '    ' + data_object1 + '=' + data_object + '.' + target_content['operation'] + '(' + param_code + ')'
            new_line_0_0 = '    np.save("type.npy", 1)\n'
            new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
            new_line_2 = '    ' + data_object1 + '=' + data_object2 + '.' + target_content[
                'operation'] + '(' + param_code + ')'
            new_line_0_1 = '    np.save("type.npy", 2)\n'
            new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
            new_line_3_1 = '    np.save("./s.npy",'+ data_object1 +'.A)'
            new_line_4 = '    ' + data_object1 + '=' + data_object2 + '.' + target_content[
                'operation'] + '(' + param_code + ')'
            new_line_5 = '    np.save("type.npy", 3)\n'
        else:
            new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
            new_code_line = '    '+data_object + '=' + data_object + '.' + target_content['operation'] + '(' + param_code + ')'
            new_line_0_0 = '    np.save("type.npy", 1)\n'
            new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
            new_line_2 = '    ' + data_object2 + '=' + data_object2 + '.' + target_content[
                'operation'] + '(' + param_code + ')'
            new_line_0_1 = '    np.save("type.npy", 2)\n'
            new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
            new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
            new_line_4 = '    ' + data_object2 + '=' + data_object2 + '.' + target_content[
                'operation'] + '(' + param_code + ')'
            new_line_5 = '    np.save("type.npy", 3)\n'
        package_code = 'import pandas as pd\nimport numpy as np\n'
        package_line_number += 2
    elif target_content['ope_type'] == 2:
        if param_code != '':
            new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
            new_code_line = '    '+data_object + '=' + 'pd.' + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
            new_line_0_0 = '    np.save("type.npy", 1)\n'
            new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
            new_line_2 = '    ' + data_object2 + '=' + 'pd.' + target_content[
                'operation'] + '(' + data_object2 + ',' + param_code + ')'
            new_line_0_1 = '    np.save("type.npy", 2)\n'
            new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
            new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
            new_line_4 = '    ' + data_object2 + '=' + 'pd.' + target_content[
                'operation'] + '(' + data_object2 + ',' + param_code + ')'
            new_line_5 = '    np.save("type.npy", 3)\n'
        else:
            new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
            new_code_line = '    '+data_object + '=' + 'pd.' + target_content['operation'] + '(' + data_object + ')'
            new_line_0_0 = '    np.save("type.npy", 1)\n'
            new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
            new_line_2 = '    '+data_object2 + '=' + 'pd.' + target_content['operation'] + '(' + data_object2 + ')'
            new_line_0_1 = '    np.save("type.npy", 2)\n'
            new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
            new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
            new_line_4 = '    '+data_object2 + '=' + 'pd.' + target_content['operation'] + '(' + data_object2 + ')'
            new_line_5 = '    np.save("type.npy", 3)\n'
        package_code = 'import pandas as pd\nimport numpy as np\n'
        package_line_number += 2
    elif target_content['ope_type'] == 3:
        if target_content['operation'] == 'OneHotEncoder':
            new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
            new_code_line = '    '+data_object + '=' + target_content[
                'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object + ').toarray()'
            new_line_0_0 = '    np.save("type.npy", 1)\n'
            new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
            new_line_2 = '    '+data_object2 + '=' + target_content[
                'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object2 + ').toarray()'
            new_line_0_1 = '    np.save("type.npy", 2)\n'
            new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
            new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
            new_line_4 = '    '+data_object2 + '=' + target_content[
                'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object2 + ').toarray()'
            new_line_5 = '    np.save("type.npy", 3)\n'
        else:
            new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
            new_code_line = '    '+data_object + '=' + target_content[
                                'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object + ')'
            new_line_0_0 = '    np.save("type.npy", 1)\n'
            new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
            new_line_2 = '    '+data_object2 + '=' + target_content[
                'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object2 + ')'
            new_line_0_1 = '    np.save("type.npy", 2)\n'
            new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
            new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
            new_line_4 = '    '+data_object2 + '=' + target_content[
                'operation'] + '(' + param_code + ')' + '.' + 'fit_transform(' + data_object2 + ')'
            new_line_5 = '    np.save("type.npy", 3)\n'
        if target_content['operation'] == 'SimpleImputer':
            package_code = 'from sklearn.impute import SimpleImputer\nimport numpy as np\n'
        elif target_content['operation'] == 'PCA':
            package_code = 'from sklearn.decomposition import PCA\nimport numpy as np\n'
        else:
            package_code = 'from sklearn.preprocessing import ' + target_content['operation'] + '\nimport numpy as np\n'
        package_line_number += 2

    elif target_content['ope_type'] == 4:
        if target_content['operation'] == 'boxcox' or target_content['operation'] == 'boxcox1p':
            package_code = 'from scipy.stats import boxcox\n'
            package_code += 'from scipy.special import boxcox1p\nimport numpy as np\n'
            package_line_number += 3
            if param_code != '':
                new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
                new_code_line = '    '+data_object + '=' +  target_content['operation'] + '(' + data_object + ',' + param_code + ')'
                new_line_0_0 = '    np.save("type.npy", 1)\n'
                new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
                new_line_2 = '    '+data_object2 + '=' + target_content[
                    'operation'] + '(' + data_object2 + ',' + param_code + ')'
                new_line_0_1 = '    np.save("type.npy", 2)\n'
                new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
                new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
                new_line_4 = '    '+data_object2 + '=' + target_content[
                    'operation'] + '(' + data_object2 + ',' + param_code + ')'
                new_line_5 = '    np.save("type.npy", 3)\n'
            else:
                new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
                new_code_line = '    '+data_object + '=' + target_content['operation'] + '(' + data_object + ')'
                new_line_0_0 = '    np.save("type.npy", 1)\n'
                new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
                new_line_2 = '    ' + data_object2 + '=' + target_content['operation'] + '(' + data_object2 + ')'
                new_line_0_1 = '    np.save("type.npy", 2)\n'
                new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
                new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
                new_line_4 = '    ' + data_object2 + '=' + target_content['operation'] + '(' + data_object2 + ')'
                new_line_5 = '    np.save("type.npy", 3)\n'
        elif target_content['operation'] == 'l2_normalize':
            prefix = 'tf.nn.'
            if param_code != '':
                new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
                new_code_line =  '    '+data_object + '=' + prefix + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
                new_line_0_0 = '    np.save("type.npy", 1)\n'
                new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
                new_line_2 =  '    '+data_object2 + '=' + prefix + target_content[
                    'operation'] + '(' + data_object2 + ',' + param_code + ')'
                new_line_0_1 = '    np.save("type.npy", 2)\n'
                new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
                new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
                new_line_4 = '    '+data_object2 + '=' + prefix + target_content[
                    'operation'] + '(' + data_object2 + ',' + param_code + ')'
                new_line_5 = '    np.save("type.npy", 3)\n'
            else:
                new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
                new_code_line = '    '+data_object + '=' + prefix + target_content['operation'] + '(' + data_object + ')'
                new_line_0_0 = '    np.save("type.npy", 1)\n'
                new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
                new_line_2 = '    '+data_object2 + '=' + prefix + target_content['operation'] + '(' + data_object2 + ')'
                new_line_0_1 = '    np.save("type.npy", 2)\n'
                new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
                new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
                new_line_4 = '    '+data_object2 + '=' + prefix + target_content['operation'] + '(' + data_object2 + ')'
                new_line_5 = '    np.save("type.npy", 3)\n'
            package_code = 'import tensorflow as tf\nimport numpy as np\n'
            package_line_number += 2
        else:
            package_code = 'import numpy as np\n'
            package_line_number += 1
            alias = 'np'
            if param_code != '':
                new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
                new_code_line = '    '+data_object + '=' +  alias + '.' + target_content['operation'] + '(' + data_object + ',' + param_code + ')'
                new_line_0_0 = '    np.save("type.npy", 1)\n'
                new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
                new_line_2 = '    '+data_object2 + '=' + alias + '.' + target_content[
                    'operation'] + '(' + data_object2 + ',' + param_code + ')'
                new_line_0_1 = '    np.save("type.npy", 2)\n'
                new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
                new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
                new_line_4 = '    '+data_object2 + '=' + alias + '.' + target_content[
                    'operation'] + '(' + data_object2 + ',' + param_code + ')'
                new_line_5 = '    np.save("type.npy", 3)\n'
            else:
                new_line_0 = 'if type(' + data_object1 + ').__name__ == "DataFrame":\n'
                new_code_line = '    '+data_object + '=' +  alias + '.' + target_content[
                    'operation'] + '(' + data_object + ')'
                new_line_0_0 = '    np.save("type.npy", 1)\n'
                new_line_1 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
                new_line_2 = '    '+data_object2 + '=' + alias + '.' + target_content[
                    'operation'] + '(' + data_object2 + ')'
                new_line_0_1 = '    np.save("type.npy", 2)\n'
                new_line_3 = 'elif type(' + data_object1 + ').__name__ == "csr_matrix":\n'
                new_line_3_1 = '    np.save("./s.npy",' + data_object1 + '.A)'
                new_line_4 = '    '+data_object2 + '=' + alias + '.' + target_content[
                    'operation'] + '(' + data_object2 + ')'
                new_line_5 = '    np.save("type.npy", 3)\n'

    new_code_line += '\n'
    temp_code_line_0 = 'temp_save_value=' + data_object1 + '\n'
    temp_code_line_1 = 'np.save("./s.npy",temp_save_value)\n'
    temp_code_line_2 = 'print(type(' + data_object1 + ').__name__)\n'
    temp_code_line_2_0 = 'print(temp_save_value)\n'
    temp_code_line_3 = new_line_0
    temp_code_line_3_1 = new_code_line
    temp_code_line_3_1_0 = new_line_0_0
    temp_code_line_3_2 = new_line_1
    temp_code_line_3_3 = new_line_2
    temp_code_line_3_3_0 = new_line_0_1
    temp_code_line_3_3_1 = new_line_3
    temp_code_line_3_3_2 = new_line_3_1
    temp_code_line_3_3_3 = new_line_4
    temp_code_line_3_3_4 = new_line_5
    temp_code_line_4 = 'temp_save_value1 = ' + data_object1 + '\n'
    temp_code_line_5_0 = 'if type('+ data_object1 + ').__name__ == "csr_matrix":\n'
    temp_code_line_5 = '    np.save("./s+1.npy",'+ data_object1 + '.A)\n'
    temp_code_line_5_0_1 = '    np.save("type_1.npy", 3)\n'
    temp_code_line_5_1 = 'elif type('+ data_object1 + ').__name__ == "DataFrame":\n'
    temp_code_line_5_2 = '    np.save("./s+1.npy",'+ data_object1 + ')\n'
    temp_code_line_5_2_1 = '    np.save("type_1.npy", 1)\n'
    temp_code_line_5_3 = 'elif type(' + data_object1 + ').__name__ == "ndarray":\n'
    temp_code_line_5_4 = '    np.save("./s+1.npy",' + data_object1 + ')\n'
    temp_code_line_5_4_1 = '    np.save("type_1.npy", 2)\n'
    temp_code_line_5_4_2 = 'print('+data_object1+')\n'
    # temp_code_line_5 = 'print(' + data_object1 + ')\n'

    # new_code = notebook_code
    # temp_code = notebook_code
    # code_list = notebook_code.split('\n')

    n_count = 0
    new_code = notebook_code
    new_code_1 = notebook_code
    temp_code = notebook_code

    if new_line_0 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_0)
        n_count += 1

    if new_line_0_0 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_0_0)
        n_count += 1
    new_code = insert_one_line_in_code(new_code,line_number+n_count,new_code_line)
    n_count += 1

    if new_line_1 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_1)
        n_count += 1
    if new_line_0_1 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_0_1)
        n_count += 1

    if new_line_2 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_2)
        n_count += 1
    if new_line_3 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_3)
        n_count += 1
    if new_line_3_1 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_3_1)
        n_count += 1
    if new_line_5 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_5)
        n_count += 1
    if new_line_4 != '':
        new_code = insert_one_line_in_code(new_code, line_number + n_count, new_line_4)
        n_count += 1

    # print('n_count', n_count)
    # print('line_number', line_number)
    # print('package_count', package_line_number)


    t_count = 0
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_0)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1,line_number+t_count,temp_code_line_1)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_2)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_2_0)
    t_count += 1
    if temp_code_line_3 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3)
        t_count += 1
    if temp_code_line_3_1_0 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_1_0)
        t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_1)
    t_count += 1

    if temp_code_line_3_2 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_2)
        t_count += 1
    if temp_code_line_3_3_0 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_3_0)
        t_count += 1
    if temp_code_line_3_3 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_3)
        t_count += 1
    if temp_code_line_3_3_1 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_3_1)
        t_count += 1
    if temp_code_line_3_3_2 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_3_2)
        t_count += 1
    if temp_code_line_3_3_4 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_3_4)
        t_count += 1
    if temp_code_line_3_3_3 != '':
        new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_3_3_3)
        t_count += 1

    # new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_4)
    # t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_0)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_0_1)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_1)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_2)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_2_1)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_2)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_3)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_4)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_4_1)
    t_count += 1
    new_code_1 = insert_one_line_in_code(new_code_1, line_number + t_count, temp_code_line_5_4_2)


    t_count = 0
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_0)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_1)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_2)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_2_0)
    t_count += 1
    if temp_code_line_3 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3)
        t_count += 1
    if temp_code_line_3_1_0 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_1_0)
        t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_1)
    t_count += 1

    if temp_code_line_3_2 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_2)
        t_count += 1
    if temp_code_line_3_3_0 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_3_0)
        t_count += 1
    if temp_code_line_3_3 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_3)
        t_count += 1
    if temp_code_line_3_3_1 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_3_1)
        t_count += 1
    if temp_code_line_3_3_2 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_3_2)
        t_count += 1
    if temp_code_line_3_3_4 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_3_4)
        t_count += 1
    if temp_code_line_3_3_3 != '':
        temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_3_3_3)
        t_count += 1


    # temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_4)
    # t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_0)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_0_1)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_1)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_2)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_2_1)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_3)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_4)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_4_1)
    t_count += 1
    temp_code = insert_one_line_in_code(temp_code, line_number + t_count, temp_code_line_5_4_2)

    code_list = temp_code.split('\n')
    temp_code = ''
    for index, line in enumerate(code_list):
        # if index == line_number:
        #     temp1 = 'np.save("./s.npy",' + data_object + ')'
        #     temp1_1 = 'print(' + data_object + ')'
        #     temp2 = 'np.save("./s+1.npy",' + data_object + ')'
        #     temp_code_line = temp1 + '\n' + temp1_1 + '\n' + new_code_line + '\n' + temp2 + '\n'
        #     temp_code += temp_code_line
        #
        #     new_code += new_code_line
        #     new_code += '\n'
        if index > line_number+t_count:
            temp_code += '# '
            temp_code += line
            temp_code += '\n'
        else:
            temp_code += line
            temp_code += '\n'
        # else:
        #     temp_code += line
        #     temp_code += '\n'
        # new_code += line
        # new_code += '\n'

    new_code = package_code + new_code
    temp_code = package_code + temp_code
    new_code_1 = package_code + new_code_1
    return new_code,temp_code,new_code_1,line_number+n_count+package_line_number

def get_feature_from_data(inp_data, column_num=100):
    column_info = {}
    dtype_dic = eval(CONFIG.get('dtype', 'dtype'))
    # print('inp_data:',inp_data)
    # print('len(inp_data[0]):')
    print('len(inp_data[0]):',len(inp_data[0]))
    try:
        inp_data_temp = inp_data[0]
        # print(inp_data[0])
    except:
        inp_data = np.array([inp_data[0]])

    for i in range(0, len(inp_data[0])):
        # print(i)
        if i >= column_num:
            break
        col = inp_data[:,i]
        # print('col:',col)
        s_s = pd.Series(col)
        # print('s_s:', s_s)
        column_info[i] = {}
        column_info[i]['col_name'] = 'unknown_' + str(i)
        column_info[i]['dtype'] = str(s_s.dtypes) # 1
        # column_info[i]['content'] = s_s.values
        column_info[i]['length'] = len(s_s.values) # 2
        column_info[i]['null_ratio'] = s_s.isnull().sum() / len(s_s.values) # 3
        column_info[i]['ctype'] = 1 if dtype_dic[str(s_s.dtypes)] == 1 or dtype_dic[str(s_s.dtypes)] == 2 else 2 # 4
        column_info[i]['nunique'] = s_s.nunique() # 5
        column_info[i]['nunique_ratio'] = s_s.nunique() / len(s_s.values) # 6
        # print(s_s)
        # print(s_s.describe())
        # print(s_s.mode())
        if column_info[i]['ctype'] == 1:  # 如果是数字列
            column_info[i]['mean'] = 0 if np.isnan(s_s.describe()['mean']) or abs(s_s.describe()['mean'])==np.inf else s_s.describe()['mean'] # 7
            column_info[i]['std'] = 0 if np.isnan(s_s.describe()['std']) or abs(s_s.describe()['std'])==np.inf else s_s.describe()['std'] # 8

            column_info[i]['min'] = 0 if np.isnan(s_s.describe()['min']) or abs(s_s.describe()['min'])==np.inf else s_s.describe()['min'] # 9
            column_info[i]['25%'] = 0 if np.isnan(s_s.describe()['25%']) or abs(s_s.describe()['25%'])==np.inf else s_s.describe()['25%']
            column_info[i]['50%'] = 0 if np.isnan(s_s.describe()['50%']) or abs(s_s.describe()['50%'])==np.inf else s_s.describe()['50%']
            column_info[i]['75%'] = 0 if np.isnan(s_s.describe()['75%']) or abs(s_s.describe()['75%'])==np.inf else s_s.describe()['75%']
            column_info[i]['max'] = 0 if np.isnan(s_s.describe()['max']) or abs(s_s.describe()['max'])==np.inf else s_s.describe()['max']
            column_info[i]['median'] = 0 if np.isnan(s_s.median()) or abs(s_s.median())==np.inf else s_s.median()
            if len(s_s.mode()) != 0:
                column_info[i]['mode'] = 0 if np.isnan(s_s.mode().iloc[0]) or abs(s_s.mode().iloc[0])==np.inf else s_s.mode().iloc[0]
            else:
                column_info[i]['mode'] = 0
            column_info[i]['mode_ratio'] = 0 if np.isnan(s_s.astype('category').describe().iloc[3] / column_info[i]['length']) or abs(s_s.astype('category').describe().iloc[3] / column_info[i]['length'])==np.inf else s_s.astype('category').describe().iloc[3] / column_info[i]['length']
            column_info[i]['sum'] = 0 if np.isnan(s_s.sum()) or abs(s_s.sum())==np.inf else s_s.sum()
            column_info[i]['skew'] = 0 if np.isnan(s_s.skew()) or abs(s_s.skew())==np.inf else s_s.skew()
            column_info[i]['kurt'] = 0 if np.isnan(s_s.kurt()) or abs(s_s.kurt())==np.inf else s_s.kurt()

        elif column_info[i]['ctype'] == 2:  # category列
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
    print('###########')
    # print(column_info)
    result = []
    for index in column_info.keys():
        one_column_feature = []
        column_dic = column_info[index]
        for kw in column_dic.keys():
            if kw == 'col_name' or kw == 'content':
                continue
            elif kw == 'dtype':
                content = dtype_dic[column_dic[kw]]
            else:
                content = column_dic[kw]
            one_column_feature.append(content)
        result.append(one_column_feature)
    if len(column_info) < column_num:
        for index in range(len(column_info),column_num):
            one_column_feature = np.zeros(19)
            result.append(one_column_feature)
    result = np.array(result)
    # print(result)
    # for i in result:
    #     for j in i:
    #         if np.isnan(j):
    #             print('has_nan:',result)
    del inp_data
    del column_info
    return result


def do_an_action(notebook_id,notebook_code,target_content,column_num,mid_line_number=-1,dataset_root='../unzip_dataset/', sat_ter=False, terminal=False):
    print('notebook_id:', notebook_id)
    operator_dic = eval(CONFIG.get('operators1', 'operations1'))
    action = (target_content['operation'], target_content['data_object'])
    for key in operator_dic:
        if target_content['operation'] == operator_dic[key]['index']:
            if target_content['operation'] == 1 or target_content['operation'] == 2 or target_content['operation'] == 5 or target_content['operation'] == 6:
                target_content['operation'] = key[3:]
                target_content['parameters'] = operator_dic[key]['default_param_numeric']
                target_content['ope_type'] = operator_dic[key]['call_type']
            else:
                target_content['operation'] = key
                target_content['parameters'] = operator_dic[target_content['operation']]['default_param_numeric']
                target_content['ope_type'] = operator_dic[target_content['operation']]['call_type']
            break

    if target_content['operation'] != 22:
        new_code, temp_code, new_code_1, res_line_number = addOperator(notebook_id,notebook_code,target_content,mid_line_number)
        if sat_ter == False:
            terminal = False if np.random.choice([0, 1], p = [0.8,0.2])==0 else True
    else:
        new_code_1 = notebook_code
        new_code = notebook_code
        temp_code = notebook_code
        res_line_number = -1
        terminal = True
    if res_line_number == -2:
        return [], action, -2, False, [], new_code, res_line_number
    ###################################################next get st+1, add feature_get and delete after code
    dataset_name = ''
    cursor, db = create_connection()
    sql = 'select dataSourceUrl from pair,dataset where pair.did=dataset.id and pair.nid=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        if row[0] == 'None' or row[0] == None:
            continue
        dataset_name = row[0].split('/')[-1]
        break

    dataset_path_root = dataset_root  + dataset_name + '.zip'

    code_list = temp_code.split('\n')
    # action = operator_dic[target_content['operation']]
    # print('action:',action)
    # if action[0] == 28:
    #     add_code, add, add_running = add_changed_result(notebook_id,new_code)
    #     code_list = add_code.split('\n')
    #     for index, line in enumerate(code_list):
    #         print("\033[0;36;40m" + str(index) + ':' + line + "\033[0m")
    #
    #     add_code = add_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
    #     add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer',
    #                                 'from sklearn.impute import SimpleImputer')
    #     add_code = add_code.replace('from sklearn.externals import joblib', 'import joblib')
    #     add_code = add_code.replace('pandas.tools', 'pandas')
    #     add_code = add_code.replace('from sklearn import cross_validation',
    #                                 'from sklearn.model_selection import cross_val_score')
    #     add_code = add_code.replace('from sklearn.cross_validation import',
    #                                 'from sklearn.model_selection import')
    #     add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')
    #
    #     add_code = insert_one_line_in_code(add_code, 'import matplotlib.pyplot as plt', 'matplotlib.use("agg")\n')
    #     add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    #     add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    #     add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    #     add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')

    if terminal == True:
        add_code, add, add_running = add_changed_result(notebook_id, new_code_1)
        code_list = add_code.split('\n')
        add_code = add_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
        add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer',
                                    'from sklearn.impute import SimpleImputer')
        add_code = add_code.replace('from sklearn.externals import joblib', 'import joblib')
        add_code = add_code.replace('pandas.tools', 'pandas')
        add_code = add_code.replace('from sklearn import cross_validation',
                                    'from sklearn.model_selection import cross_val_score')
        add_code = add_code.replace('from sklearn.cross_validation import',
                                    'from sklearn.model_selection import')
        add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')
        add_code = add_code.replace('.as_matrix()', '.values')

        add_code = insert_one_line_in_code(add_code, 'import matplotlib.pyplot as plt', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')
        for index, line in enumerate(code_list):
            print("\033[0;36;40m" + str(index) + ':' + line + "\033[0m")

        res = running_temp_code(add_code, dataset_path_root,0)
        # terminal = True
        if res == 'break':
            return [], action, -2, terminal, [], new_code, res_line_number
        elif res[0:7] == 'error 8':
            global error_str_dic
            if res[7:] not in error_str_dic:
                error_str_dic[res[7:]] = []
            error_str_dic[res[7:]].append(action[0])
            reward = -1
            np.save('./error_str_dic.npy', error_str_dic)
        else:
            changed_result = np.load('./temp_result.npy',allow_pickle=True)
            os.system('rm -f ./temp_result.npy')
            origin_result = []
            cursor, db = create_connection()
            sql = 'select * from result where notebook_id=' + str(notebook_id)
            cursor.execute(sql)
            sql_res = cursor.fetchall()
            for row in sql_res:
                if row[2] != None:
                    origin_result.append([row[1],row[3],row[2],row[5],row[4]])
                else:
                    origin_result.append([row[1], row[3], row[6], row[5], row[4]])
            changed_reward = 0
            reward = 0
            origin_max = {}
            changed_max = {}
            # print('changed_result:',changed_result)

            origin_count = 0
            model_dic = eval(CONFIG.get('models', 'model_dic'))
            for index,item in enumerate(changed_result):
                if origin_count >= len(origin_result):
                    break
                if len(changed_result) != len(origin_result) and (origin_result[origin_count][4] != item[4] or origin_result[origin_count][1] != item[1] or origin_result[origin_count][3] != item[3]):
                    continue
                if item[1] not in model_dic.keys():
                    continue
                # print("origin_result", origin_result[origin_count][2])
                # print("changed_result", item[2])

                # print(type(item[2]).__name__)
                if type(item[2]).__name__ == 'list' and item[4] == 'evaluate':
                    item[2] = item[2][-1]
                    print(item[2])
                elif type(item[2]).__name__ == 'ndarray' and item[4] == 'cross_val_score':
                    continue
                elif item[4] == 'confusion_matrix' or item[4] == 'classification_report' or item[4] == 'mean_absolute_error' or item[4] == 'mean_squared_error':
                    continue
                # print(item[2])
                item[2] = float(item[2])
                # print(item[1])
                if item[1] not in changed_max:
                    changed_max[item[1]] = float(item[2])
                else:
                    if float(item[2]) > changed_max[item[1]]:
                        changed_max[item[1]] = float(item[2])

                if type(origin_result[origin_count][2]).__name__ == 'str' and origin_result[origin_count][4] == 'evaluate':
                    origin_result[origin_count][2] = float(origin_result[origin_count][2][1:-1].split(',')[-1].strip())
                elif type(origin_result[origin_count][2]).__name__ == 'str' and origin_result[origin_count][4] == 'cross_val_score':
                    score_list = origin_result[origin_count][2][1:-1].split(' ')
                    sum = 0
                    count = 0
                    for score in score_list:
                        if score == '':
                            continue
                        # print(score.strip())
                        count += 1
                        sum += float(score.strip())
                    if count == 0:
                        origin_result[origin_count][2] = 0
                    else:
                        origin_result[origin_count][2] = sum / count
                if origin_result[origin_count][4] == 'cross_val_score' or origin_result[origin_count][4] == 'confusion_matrix' or origin_result[origin_count][4] == 'classification_report' or origin_result[origin_count][4] ==  'mean_absolute_error' or origin_result[origin_count][4] == 'mean_squared_error':
                    continue
                # print("origin:",origin_result[origin_count][2])
                if origin_result[origin_count][1] not in origin_max:
                    origin_max[origin_result[origin_count][1]] = float(origin_result[origin_count][2])
                else:
                    if float(origin_result[origin_count][2]) > origin_max[origin_result[origin_count][1]]:
                        origin_max[origin_result[origin_count][1]] = float(origin_result[origin_count][2])
                origin_count += 1
                    # if len(str(origin_result[origin_count][2]).split('.')) != 1:
                    #     round_num = len(str(origin_result[origin_count][2]).split('.')[-1])
                    #     item[2] = round(float(item[2]),round_num)
                    # if float(item[2]) == float(origin_result[origin_count][2]):
                    #     continue
                    # else:
                    #     print('origin:',float(origin_result[origin_count][2]))
                    #     print('changed:', float(item[2]))
                    #     reward += float(item[2]) - float(origin_result[origin_count][2])
                    #     changed_reward += 1

            reward = 0
            # print('origin_max:',origin_max)
            # print('changed_max', changed_max)
            for item in origin_max:
                if item not in changed_max:
                    continue
                if len(str(origin_max[item]).split('.')) != 1:
                    round_num = len(str(origin_max[item]).split('.')[-1])
                    changed_max[item] = round(float(changed_max[item]), round_num)
                temp = changed_max[item] - origin_max[item]
                if abs(temp) > abs(reward):
                    reward = temp
            if abs(reward) < 0.00001:
                reward = 0
        # if changed_reward != 0:
        #     reward /= changed_reward
        # else:
        #     reward = 0

    else:
        add_code = temp_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
        add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer',
                                    'from sklearn.impute import SimpleImputer')
        add_code = add_code.replace('from sklearn.externals import joblib', 'import joblib')
        add_code = add_code.replace('pandas.tools', 'pandas')
        add_code = add_code.replace('from sklearn import cross_validation',
                                    'from sklearn.model_selection import cross_val_score')
        add_code = add_code.replace('from sklearn.cross_validation import',
                                    'from sklearn.model_selection import')
        add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')
        add_code = add_code.replace('.as_matrix()', '.values')

        add_code = insert_one_line_in_code(add_code, 'import matplotlib.pyplot as plt', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
        add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')
        for index, line in enumerate(code_list):
            print("\033[0;33;40m" + str(index) + ':' + line + "\033[0m")
        res = running_temp_code(add_code, dataset_path_root, 0)
        # print("res:",res)
        if res[0:7] == 'error 8':
            # global error_str_dic
            if res[7:] not in error_str_dic:
                error_str_dic[res[7:]] = []
            error_str_dic[res[7:]].append(action[0])
            np.save('./error_str_dic.npy',error_str_dic)
        reward = -1 if res[0:7] == 'error 8' else 0
        terminal = True if res[0:7] == 'error 8' else False
    print('running end')    # if res[0:7] != 'error 8':

    len_data_plus_1 = 0
    try:
        data_t = np.load("./s.npy", allow_pickle=True)
        # data_type = int(np.load("type.npy", allow_pickle=True))
        os.system('rm -f ./s.npy')
        s_t = get_feature_from_data(data_t, column_num)
    except Exception as e:
        # data_t_plus_1 = []
        print(e)
        s_t = []
    try:
        data_t_plus_1 = np.load("./s+1.npy", allow_pickle=True)
        os.system('rm -f ./s+1.npy')
        s_t_plus_1 = get_feature_from_data(data_t_plus_1, column_num)
        len_data_plus_1 = len(data_t_plus_1[0])
    except Exception as e:
        # data_t_plus_1 = []
        print(e)
        s_t_plus_1 = []
        # data_type = int(np.load("type.npy", allow_pickle=True))

    print('returned')
    # print(data_t)
    # print(data_t_plus_1)

    if terminal == False:
        return s_t,action,reward,terminal,s_t_plus_1,new_code,res_line_number,len_data_plus_1
    if terminal == True:
        return s_t,action,reward,terminal,s_t_plus_1,new_code,res_line_number,len_data_plus_1

def get_origin_state(notebook_id,notebook_code,column_num,dataset_root= '../unzip_dataset/'):
    dataset_name = ''
    cursor, db = create_connection()
    sql = 'select dataSourceUrl from pair,dataset where pair.did=dataset.id and pair.nid=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        dataset_name = row[0].split('/')[-1]
        break

    sql = 'select operator,rank from operator where notebook_id=' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    change_rank=0

    result_code_line = get_result_code(notebook_id, notebook_code, 1, get_min_max=0)
    # print(result_code_line)
    if type(result_code_line).__name__ == 'str':
        result_code_line = 10000
    else:
        result_code_line = result_code_line[0][1]
    # print('result_code_line:',result_code_line)
    for row in sql_res:
        change_rank=row[1]

    ope_dic = eval(CONFIG.get('operators', 'operations'))
    operator_code_line = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
    if type(operator_code_line).__name__ == 'str':
        change_rank = 0
    else:
        operator_code_line = operator_code_line[0][0][1]
        # print(operator_code_line)
        while change_rank > 0 and (operator_code_line > result_code_line or operator_code_line == -1):
            change_rank -= 1
            operator_code_line = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
            if operator_code_line == 'no such operator':
                continue
            operator_code_line = operator_code_line[0][0][1]
        if operator_code_line == 'no such operator':
            change_rank = 0
    code_list = notebook_code.split('\n')

    # print(notebook_code)
    # walklogs = np.load('../walklogs/' + str(notebook_id) + '.npy', allow_pickle=True).item()
    # print('change_rank',change_rank)
    if change_rank != 0:
        res = get_operator_code(notebook_id, notebook_code, change_rank, ope_dic)
        # print('get_ope_cod_res:',res)
        if res == 'no such operator':
            return res
        #如果新增加的位置不是起始位置，那么数据对象是上一个操作的对象，否则是读入数据的对象

        # print('res1:',res1)
        code = res[0][0][0]
        r_node = ast.parse(code.strip())
        operation = res[1]
        call_type = ope_dic[operation]['call_type']

        # print('data_object_type:',type(r_node.body[0]).__name__)
        if type(r_node.body[0]).__name__ == 'Assign':
            # print(astunparse.unparse(r_node))
            for target in r_node.body[0].targets:
                if type(target).__name__ == 'Name':
                    data_object = target.id
                    break
                else:
                    data_object = astunparse.unparse(target)
                    if data_object[-1] == '\n':
                        data_object = data_object[0:-1]
                    break
        else:
            # print('call_type:', call_type)
            data_object_value = res[2]
            if call_type == 0:
                data_object = data_object_value[0:data_object_value.find(operation) - 1]
            elif call_type == 2 or call_type == 4:
                data_object = data_object_value
            elif call_type == 5:
                data_object = res[3]
            elif call_type == 3:
                data_object = res[2]

        candidate_code_list = res[0]
        # print(candidate_code_list)
        line_number = candidate_code_list[0][1]
    else:
        res = get_read_code(notebook_code)
        if res == 'no read data':
            return [],0
        data_object = res[0]
        line_number = res[1]
    # notebook_code = 'print(tf.constant(0))\n' + notebook_code
    code_list = notebook_code.split('\n')
    temp_code = ''
    # print('line_number:',line_number)

    # print('get res:',res)
    for index, line in enumerate(code_list):
        # if index == line_number:
        #     temp1 = 'np.save("./s.npy",' + data_object + ')'
        #     temp1_1 = 'print(' + data_object + ')'
        #     temp2 = 'np.save("./s+1.npy",' + data_object + ')'
        #     temp_code_line = temp1 + '\n' + temp1_1 + '\n' + new_code_line + '\n' + temp2 + '\n'
        #     temp_code += temp_code_line
        #
        #     new_code += new_code_line
        #     new_code += '\n'
        if index > line_number:
            temp_code += '# '
            temp_code += line
            temp_code += '\n'
        else:
            temp_code += line
            temp_code += '\n'

    temp_code_line = 'import numpy as np\n'
    notebook_code = insert_one_line_in_code(temp_code, line_number + 1, temp_code_line)
    temp_code_line = 'np.save("./s.npy",' + data_object + ')\n'
    notebook_code = insert_one_line_in_code(notebook_code, line_number + 2, temp_code_line)
    temp_code_line_0 = 'np.save("./type.npy", 1 if type(' + data_object + ').__name__ == "DataFrame" else 2)\n'
    notebook_code = insert_one_line_in_code(notebook_code, line_number + 3, temp_code_line_0)
    add_code = notebook_code.replace('from sklearn.preprocessing import Imputer', 'from sklearn.impute import Imputer')
    add_code = add_code.replace('from sklearn.preprocessing import SimpleImputer',
                                'from sklearn.impute import SimpleImputer')
    add_code = add_code.replace('from sklearn.externals import joblib', 'import joblib')
    add_code = add_code.replace('pandas.tools', 'pandas')
    add_code = add_code.replace('from sklearn import cross_validation',
                                'from sklearn.model_selection import cross_val_score')
    add_code = add_code.replace('from sklearn.cross_validation import',
                                'from sklearn.model_selection import')
    add_code = add_code.replace('import plotly.plotly as py', 'import chart_studio.plotly as py')
    add_code = add_code.replace('.as_matrix()', '.values')

    add_code = insert_one_line_in_code(add_code, 'import matplotlib.pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from matplotlib import pyplot as plt', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'import seaborn', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'from seaborn import', 'matplotlib.use("agg")\n')
    add_code = insert_one_line_in_code(add_code, 'matplotlib.use("agg")', 'import matplotlib\n')
    dataset_path_root = dataset_root + dataset_name + '.zip'
    code_list = add_code.split('\n')
    # for index, line in enumerate(code_list):
    #     print("\033[0;38;40m" + str(index) + ':' + line + "\033[0m")
    res = running_temp_code(add_code, dataset_path_root, 0)
    if res[0:7] == 'error 8':
        return [],0


    # print(type(data_t))
    # if data_t != None:
    # os.system('rm -f ./s.npy')
    len_data = 0
    try:
        data_t = np.load("./s.npy", allow_pickle=True)
        len_data = len(data_t[0])
        s_t = get_feature_from_data(data_t, column_num)
    except:
        s_t = [],0
    return s_t,len_data


############## 下面是通信部分

class MsgServicer(message_pb2_grpc.MsgServiceServicer):
    def rpc_do_an_action(self,request, context):
        def parse_do_an_action_result(s_t, s_t_plus_1, action_1,action_2, terminal, reward, new_code, res_line_number,len_data_plus_1):
            result = message_pb2.do_an_action_result(action_1=action_1,action_2=action_2, terminal=terminal, reward=reward, new_code=new_code,
                                                    res_line_number=res_line_number,len_data_plus_1=len_data_plus_1)
            for i in range(len(s_t)):
                d = result.s_t.add()
                d.row.extend(s_t[i])
            for i in range(len(s_t_plus_1)):
                d = result.s_t_plus_1.add()
                d.row.extend(s_t_plus_1[i])
            return result
        target_content = {
            'operation': request.target_content_operation,
            'data_object': request.target_content_data_object
        }
        print('?>>???????')
        s_t, action, reward, terminal, s_t_plus_1, new_code, res_line_number,len_data_plus_1 = do_an_action(request.notebook_id,request.notebook_code,target_content,request.column_num,request.res_line_number,dataset_root='../unzip_dataset/')
        # print('?>>???????')
        response = parse_do_an_action_result(s_t,s_t_plus_1,action[0],action[1],terminal,reward,new_code,res_line_number,len_data_plus_1)
        return response
    def rpc_get_origin_state(self,request, context):
        def parse_get_origin_state_result(s_t,len_data):
            result = message_pb2.get_origin_state_result(len_data=len_data)
            for i in range(len(s_t)):
                d = result.s_t.add()
                d.row.extend(s_t[i])
            return result

        s_t,len_data = get_origin_state(request.notebook_id, request.notebook_code, column_num=request.column_num, dataset_root='../unzip_dataset/')
        if type(s_t).__name__ == 'str':
            s_t = []
        response = parse_get_origin_state_result(s_t,len_data)
        return response

def serve():
    dataset_root = '../unzip_dataset/'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_MsgServiceServicer_to_server(MsgServicer(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    print("server_start")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

def check_model(notebook_id):
    model_dic = eval(CONFIG.get('models', 'model_dic'))
    cursor, db = create_connection()
    sql = 'select model_type from result where notebook_id = ' + str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    model_list = np.zeros([len(model_dic.keys())])
    check = False
    for row in sql_res:
        if row[0] in model_dic.keys():
            model_id = model_dic[row[0]]-1
            model_list[model_id] = 1
            check = True
    return check,model_list
if __name__ == '__main__':
    serve()
    # np.set_printoptions(threshold=np.inf)
    # notebook_id=1502124
    # ip = get_host_ip()
    # # notebook_id = 8281152
    # server_dic = eval(CONFIG.get('server', 'server'))
    # notebook_root_path = server_dic[ip]['npath']
    # dataset_root = server_dic[ip]['dpath']
    # notebook_path = notebook_root_path + str(notebook_id) + '.ipynb'
    # notebook_code = get_code_txt(notebook_path)
    # s_t ,len_data =get_origin_state(notebook_id,notebook_code,101)
    # print(s_t)
    #
    # check_result, model_list = check_model(notebook_id)
    # target_content = {
    #     'operation': 18,
    #     'data_object': 0,
    # }
    # s_t, action, reward, terminal, s_t_plus_1, new_code, res_line_number,len_data = do_an_action(notebook_id, notebook_code,
    #                                                                                  target_content,column_num=100,sat_ter=True,terminal=False)
    #
    # target_content = {
    #     'operation': 9,
    #     'data_object': 3,
    # }
    # s_t, action, reward, terminal, s_t_plus_1, new_code, res_line_number, len_data = do_an_action(notebook_id,
    #                                                                                               notebook_code,
    #                                                                                               target_content,
    #                                                                                               column_num=100,
    #                                                                                               sat_ter=True,
    #                                                                                               terminal=False)
    # target_content = {
    #     'operation': 4,
    #     'data_object': -1,
    # }
    # s_t, action, reward, terminal, s_t_plus_1, new_code, res_line_number, len_data = do_an_action(notebook_id,
    #                                                                                               notebook_code,
    #                                                                                               target_content,
    #                                                                                               column_num=100,
    #                                                                                               sat_ter=True,
    #                                                                                               terminal=False)
    # target_content = {
    #     'operation': 9,
    #     'data_object': 20,
    # }
    # s_t, action, reward, terminal, s_t_plus_1, new_code, res_line_number, len_data = do_an_action(notebook_id,
    #                                                                                               notebook_code,
    #                                                                                               target_content,
    #                                                                                               column_num=100,
    #                                                                                               sat_ter=True,
    #                                                                                               terminal=False)

    # s_t = np.ravel(s_t)
    # type_1 = np.array([int(np.load('type.npy', allow_pickle=True))])
    # if int(np.load('type.npy', allow_pickle=True)) != 1:
    #     terminal = True
    # if len(s_t) == 1900:
    #     s_t = np.concatenate((type_1, s_t), axis=0)
    # if len(s_t) == 1901:
    #     s_t = np.concatenate((s_t, model_list), axis=0)
    # print(reward)
    # s_t_p = s_t_plus_1
    # s_t_plus_1 = np.ravel(s_t_plus_1)
    # type_1 = np.array([int(np.load('type_1.npy', allow_pickle=True))])
    # if int(np.load('type_1.npy', allow_pickle=True)) != 1:
    #     terminal = True
    # if len(s_t_plus_1) == 1900:
    #     s_t_plus_1 = np.concatenate((type_1, s_t_plus_1), axis=0)
    # if len(s_t_plus_1) == 1901:
    #     s_t_plus_1 = np.concatenate((s_t_plus_1, model_list), axis=0)
    #
    # # s_t = s_t_plus_1
    # count = 0
    # for i in s_t:
    #     print('s_t_' + str(count), i)
    #     count += 1
    # count = 0
    # for i in s_t_plus_1:
    #     print('s_t_plus_1_' + str(count), i)
    #     count += 1
    # print((s_t==s_t_plus_1).all())
    # print((np.ravel(s_t)==np.ravel(s_t_plus_1)).all())
    # print((np.ravel(s_t) == np.ravel(s_t_plus_1)).all())
    # print((np.ravel(s_t) == np.ravel(s_t_plus_1)).all())
    # target_content = {
    #     'operation': 6,
    #     'data_object': 7,
    # }
    #
    # s_t, action, reward, terminal, s_t_plus_1,new_code,res_line_number,len_data = do_an_action(notebook_id,new_code,target_content,mid_line_number=res_line_number,column_num=100)
    # print(reward)
    # # print(s_t)
    # # print(s_t_plus_1)
    # print((s_t==s_t_plus_1).all())