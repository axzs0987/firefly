
from compile_notebook.read_ipynb import read_ipynb
from compile_notebook.LR_matching import Feeding
from compile_notebook.LR_matching import LR_run
from notebook2sequence import single_running
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
    filename = old_path.split('/')[-1]
    print('filename', filename)
    if '.' not in filename:
        result = root_path
    else:
        result = root_path + '/' + filename
    print("result", result)
    return origin_code.replace(old_path, result)


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

def insert_one_line_in_code(origin_code, under_line, target_line):
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
            target = converse_target(code_list[index_list[index]] , target_line)
            code_list.insert(index_list[index], target)
            if index != len(index_list)-1:
                for after_index in range(index + 1, len(index_list)):
                    index_list[after_index] += 1

    #如果underline是数字，则在该行都下一行插入目标
    elif type(under_line).__name__ == 'int':
        target = converse_target(code_list[under_line], target_line)
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
        target_line = 'print("origin_data ' + str(i) + '" , ' + csv_varible[i] + ')'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 1, target_line)
        target_line = 'for col in ' + csv_varible[i] + ':'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 2, target_line)
        target_line = '    if str(' + csv_varible[i] + '[col].dtype) == "object":'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 3, target_line)
        target_line = '        column_name.append(str(col))'
        origin_code = insert_one_line_in_code(origin_code, csv_index[i] + 4, target_line)
        if i != len(csv_index) - 1:
            for after_index in range(i + 1, len(csv_index)):
                csv_index[after_index] += 4

    code_list = origin_code.split('\n')
    code_list.append("print(column_name)")
    result = ''
    for i in code_list:
        result = result + i + '\n'
    return result


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
    except:
        return "compile fail"
    print("\033[0;33;40m" + str(count) +"\033[0m")
    can_run = False
    try:
        exec(cm)
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
        if count < 1 and can_run==False:
            res = running(new_code, new_path, count + 1)
            return res
    return func_def

def add_variable_code(origin_code, variable_list,notebook_name):
    """
    :param origin_code: 原始代码
    :param variable_list: 数据流的变量名
    :param notebook_name: notebook名字
    :return:
    该函数是为了得到字符串列的变化流。属于experiment任务。
    """
    code_list = origin_code.split('\n')
    line = "def add_pair(series1, series2,save_path_file):"
    code_list.append(line)
    line = "    if len(series1) != len(series2):"
    code_list.append(line)
    line = "        return series1"
    code_list.append(line)
    line = "    if (series1.index != series2.index).any():"
    code_list.append(line)
    line = "        return series1"
    code_list.append(line)
    line = "    if (series1.values != series2.values).any():"
    code_list.append(line)
    line = "        result = pd.DataFrame([series1, series2])"
    code_list.append(line)
    line = "        result.to_csv(save_path_file)"
    code_list.append(line)
    line = "        return series2"
    code_list.append(line)
    line = "    return series1"
    code_list.append(line)



    line = "def compare_series(series1, series2):"
    code_list.append(line)
    line = "    if len(series1) != len(series2):"
    code_list.append(line)
    line = "        return False"
    code_list.append(line)
    line = "    if (series1.index != series2.index).any():"
    code_list.append(line)
    line = "        return False"
    code_list.append(line)
    line = "    if (series1.values != series2.values).any():"
    code_list.append(line)
    line = "        print(series1.values != series2.values)"
    code_list.append(line)
    line = "        return False"
    code_list.append(line)
    line = "    return True"
    code_list.append(line)

    line = "for column in column_name:"
    code_list.append(line)
    line = "    new_column_df = []"
    code_list.append(line)
    line = "    pair_column_df = []"
    code_list.append(line)
    line = "    count = 0"
    code_list.append(line)
    line = "    pair_count = 0"
    code_list.append(line)
    line = "    save_path = '../strcol/" + notebook_name + "'"
    code_list.append(line)
    line = "    if os.path.exists(save_path) == False:"
    code_list.append(line)
    line = "        os.mkdir(save_path)"
    code_list.append(line)
    line = "    save_path = '../strcol/" + notebook_name + "/' + column"
    code_list.append(line)
    line = "    if os.path.exists(save_path) == False:"
    code_list.append(line)
    line = "        os.mkdir(save_path)"
    code_list.append(line)
    line = "    pair_path = '../strcol/" + notebook_name + "/' + column + '/pair'"
    code_list.append(line)
    line = "    if os.path.exists(pair_path) == False:"
    code_list.append(line)
    line = "        os.mkdir(pair_path)"
    code_list.append(line)
    for i in variable_list:
        line = "    if type(" + i + ").__name__ == 'DataFrame':"
        code_list.append(line)
        line = "        if column in [column for column in " +  i + "]:"
        code_list.append(line)
        line = "            if str(" + i + "[column].dtype) == 'object':"
        code_list.append(line)
        line = "                save_pair_file = pair_path + '/' + str(pair_count) + '.csv'"
        code_list.append(line)
        line = "                pair_column_df = add_pair(pair_column_df, " + i + "[column], save_pair_file)"
        code_list.append(line)
        line = "                if compare_series(" + i + "[column], new_column_df) == False:"
        code_list.append(line)
        line = "                    new_column_df = " + i + "[column]"
        code_list.append(line)
        line = "                    save_path_file = save_path + '/' + str(count) + '.csv'"
        code_list.append(line)
        line = "                    " + i + "[column].to_csv(save_path_file)"
        code_list.append(line)
        line = "                    count += 1"
        code_list.append(line)
    result = ''
    for i in code_list:
        result = result + i + '\n'
    return result

# def get_result(origin_code, notebook_id)

def test_running(notebook_root, dataset_root, notebook_name, dataset_name):
    notebook_path = notebook_root + '/' + notebook_name
    dataset_path_root = dataset_root + '/' + dataset_name
    func_def = get_code_txt(notebook_path)
    can_run_code = running(func_def,dataset_path_root,0)
    if can_run_code == "compile fail":
        return
    # print("can_run_code",can_run_code)
    # add_csv_code = print_readcsv(can_run_code)
    # can_run_code = running(add_csv_code, dataset_path_root, 0)
    # print("can_run_code", can_run_code)

    walk_logs = single_running(1, notebook_name.split('.')[0], notebook_root)
    print(walk_logs['data_values'])
    print(walk_logs['data_types'])

    add_csv_code = print_readcsv(can_run_code)
    print("add_varible_code:", add_csv_code)
    can_run_code = running(add_csv_code, dataset_path_root, 0)

    add_varible_code_content = add_variable_code(can_run_code, walk_logs['data_values'],notebook_name.split('.')[0])
    print("add_varible_code_content:", add_varible_code_content)
    can_run_code = running(add_varible_code_content, dataset_path_root, 0)

    # print("can_run_code", can_run_code)

def batch_running(notebook_root,dataset_root,pair_path):
    notebook_name_list, dataset_name_list = get_pairs(pair_path)
    for i in range(1, 2):
        try:
            notebook_name = notebook_name_list[i]
            dataset_name =  dataset_name_list[i]
            test_running(notebook_root, dataset_root, notebook_name, dataset_name)
            print("\033[0;32;40m\tsucceed\033[0m")
        except:
            print("\033[0;31;40m\terror\033[0m")



if __name__ == '__main__':
    batch_running("../spider/notebook", '../spider/unzip_dataset', '../spider/pair.txt')



#    func_def = get_code_txt('../notebook/bike-sharing-exploratory-analysis.ipynb')
#    cm = compile(func_def, '<string>', 'exec')
#    try:
#        exec(cm)
#    except Exception as e:        
#        print(e)

    # r_node = ast.parse(fillna_example)
    # print(astunparse.dump(r_node))