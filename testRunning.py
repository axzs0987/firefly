
from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run
from notebook2sequence import single_running
from utils import get_params_code_by_id
import os



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


def get_pairs(path):
    """
    :param path:
    :return:
    暂时使用的函数，pair数据库建起来之前，使用pair.txt找对应关系
    """
    notebook_name_list = []
    dataset_name_list = []
    with open(path) as f:
        for line in f:
            line_list = line.split(']')
            notebook_name = line_list[0].split('/')[-1]
            dataset_name = line_list[1].split('/')[-1][0:-1]
            notebook_name_list.append(notebook_name)
            dataset_name_list.append(dataset_name)

    return notebook_name_list, dataset_name_list

def insert_one_line_in_code(origin_code, under_line, target_line, is_same = True):
    """
    :param origin_code: 原本代码
    :param under_line: 目标插入的位置，int，就是第几行（从0开始），string就是在匹配到包含此字符串的行的下面都插入
    :param target_line: 目标插入代码
    :return: 转换后的代码
    """
    def get_space_num(line):
        count = 0
        for char in line:
            if char == ' ':
                count += 1
            else:
                break
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
            print(index, line)
        for index in range(0, len(index_list)):
            if is_same == True:
                target = converse_target(code_list[index_list[index]-1] , target_line)
            code_list.insert(index_list[index], target)
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


def running(func_def, new_path,count):
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
    print("\033[0;33;40m" + str(count) +"\033[0m")
    can_run = False
    try:
        print(func_def)
        ns = {}
        exec(cm,ns)
        print("\033[0;32;40msucceed\033[0m")
        can_run = True
    except Exception as e:
        error_str = str(e)
        print("\033[0;31;40merror_str\033[0m", error_str)
        if "[Errno 2] No such file or directory: " in error_str:
            error_path = error_str.replace("[Errno 2] No such file or directory: " , "")
            error_path = error_path[1:-1]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            print('error_path:', error_path)
            # running(new_code)
        elif "does not exist:" in error_str and '[Errno 2] File ' in error_str:
            error_path = error_str.split(':')[-1].strip()
            error_path = error_path[1:-1]
            new_code = found_dataset(error_path, 1, new_path, func_def)
            print('error_path:', error_path)
            print('new_code:', new_code)
        elif "No module named " in error_str:
            package = error_str.replace("No module named ", "")
            package = package[1:-1]
            command = ' pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ' + package.split('.')[0]
            os.system(command)
        elif  ": No such file or directory" in error_str:
            index1 = error_str.find("'")
            index2 = error_str.find("'", index1+1)
            error_path = error_str.substr(index1+1,index2)
            new_code = found_dataset(error_path, 1, new_path, func_def)
            print('error_path:', error_path)
        if count < 4 and can_run==False:
            # print(new_code)
            res = running(new_code, new_path, count + 1)
            return res
    return func_def

def add_variable_code(origin_code, variable_list,notebook_name, save_root="../strcol"):
    function_def = \
        """
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
            new_df[column].to_csv(save_path + '/' + col_path + '/' + str(new_count[column]) + '.csv')
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

def add_params(id, notebook_name,notebook_root, dataset_name, dataset_root):

    walk_logs = single_running(1, notebook_name.split('.')[0], notebook_root)
    param_code_list = get_params_code_by_id(id)

    notebook_path = notebook_root + '/' + notebook_name
    dataset_path_root = dataset_root + '/' + dataset_name
    func_def = get_code_txt(notebook_path)

    code_list = func_def.split('\n')

    code_index = 0
    now_insert_num = 0
    for index, i in range(param_code_list):
        if i['name'] + '(' + i['p1'] in code_list[code_index] or i['name'] + '( ' + i['p1'] in code_list[code_index]:
            #找到operaotr对应的行了 A=LabelEncoder()
            parameter_code = code_list[code_index].substr[code_list[code_index].find(i['name']+'(') + len(i['name']):]
            parameter_lis_str = parameter_code[0:parameter_code.find(')')]
            parameter_value_list = parameter_lis_str.split(',')
            for ind,param_value in parameter_value_list:
                code_list.insert(code_index + ind + 1, "insert_param_db(notebook_id, " + str(ind+1) + "," + param_value + ")")






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

def batch_running(notebook_root,dataset_root,pair_path):
    notebook_name_list, dataset_name_list = get_pairs(pair_path)
    for i in range(0,len(notebook_name_list)):
        try:
            notebook_name = notebook_name_list[i]
            dataset_name =  dataset_name_list[i]

            # if notebook_name == "most-dangerous-departure-and-destination-cities.ipynb":
            print("\033[0;31;44m" + notebook_name + "\033[0m")
            test_running(notebook_root, dataset_root, notebook_name, dataset_name)
            # print("\033[0;32;40m\tsucceed\033[0m")
        except:
            continue
            print("\033[0;31;40m\terror\033[0m")



if __name__ == '__main__':
    batch_running("../spider/notebook", '../spider/unzip_dataset', '../spider/pair.txt')


