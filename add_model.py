from utils import CONFIG
from utils import create_connection
from utils import get_code_txt
from utils import update_db
import ast
from modifySequence import get_operator_code
from modifySequence import get_result_code
import astunparse
import os

import numpy as np

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
                target = converse_target(code_list[index_list[index]] , target_line)
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

            target = converse_target(code_list[under_line], target_line)
        code_list.insert(under_line, target)

    # 把list转文本
    result = ''
    for i in code_list:
        result = result + i+'\n'
    return result

def get_model_from_code(ctxt,origin_code,line_num,ins_num):
    add_line = 0
    print('ctxt:',ctxt)
    def get_type1(line,origin_code,line_num,model_variable,add_line,ins_num):
        print(line.strip())
        try:
            r_node = ast.parse(line.strip())
        except:
            if 'for' in line and 'in' in line and ':' in line:
                origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                                      "mdtypes_" + str(ins_num) +".add(type(" + model_variable + ').__name__)')
                add_line += 1
                return type(model_variable).__name__,origin_code,add_line
            elif 'def' in line and '(' in line and '):' in line:
                left = line.index('(')
                right = line.index(')')
                p_list = line.replace(' ','')[left+1:right].split(',')
                if model_variable in p_list:
                    origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                                          "mdtypes_" + str(ins_num) +".add(type(" + model_variable + ').__name__)')
                    add_line += 1
                    return type(model_variable).__name__,origin_code,add_line
            return False
        if len(r_node.body) == 0:
            return False
        print('/////')
        print(line_num+1)
        print(model_variable)
        add_line += 1
        origin_code = insert_one_line_in_code(origin_code, line_num + 1,
                                               "mdtypes_" + str(ins_num) +".add(type(" + model_variable + ').__name__)')
        print(type(r_node.body[0]).__name__)
        if type(r_node.body[0]).__name__ == 'Assign':
            model_node = r_node.body[0].value
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
                subline = line[index_for:index_in].strip()
                subline_list = subline.split(',')
                # print(subline_list)
                # print(model_variable)
                for code in subline_list:
                    if code.strip() == model_variable:
                        # print('true')
                        return True
                return False
            elif 'def' in line and '(' in line and '):' in line:
                left = line.index('(')
                right = line.index(')')
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
            print('555')
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
                    print('444')
                    return 'unknown',origin_code,add_line
            elif type(call_func).__name__ == 'Name':
                print('333')
                return call_func.id,origin_code,add_line

        code_list = origin_code.split('\n')
        line_list = set()
        new_line_num = 0


        print('model_variable:', model_variable)
        print('line_num:',line_num)
        for index,line in enumerate(code_list):
            if index > line_num-1:
                break
            if check_line(model_variable,line) == True:
                line_list.add(line)
                new_line_num = index
        print(line_list)
        if len(line_list) == 0:
            print('111')
            return 'unknown',origin_code,add_line
        else:
            need_parse_line = list(line_list)[-1]
            # print('\033[0;36;40m' + need_parse_line + '\033[0m')
            # print('get_type:', need_parse_line)s
            print('222')
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
                return 'unknown'
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
                        return 'unknown'

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
                        return 'unknown'

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
                    return 'unknown'

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
def add_result(notebook_id, origin_code, walk_logs_path = "../walklogs"):
    func_def = ''
    func_def += "from utils import add_result\n"
    func_def += "def insert_result(type, content, code, model_type):\n"
    func_def += "    notebook_id = " + str(notebook_id) + '\n'
    func_def += "    add_result(notebook_id, type, content, code, model_type)\n"

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
    result = set()

    ins_num = 0
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
            need2print = "insert_result(" + str(model_id) + "," + code[head + 1:right_index + 1] + ',"' + temp_code + '", "'+ now_name +'")'

            ctxt = code[head + 1:right_index + 1]
            add_model_result = get_model_from_code(ctxt, origin_code, line, ins_num)
            result.add(add_model_result[0])
            origin_code = add_model_result[1]
            add_line = add_model_result[2]
            line += 1
            line += add_line
            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            ins_num += 1
            line += 1
            origin_code = insert_one_line_in_code(origin_code, line, 'mdtypes_' + str(ins_num)  +'  = set()')
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
            need2print = "insert_result(" + str(model_id) + "," + code[head+1:right_index + 1] +',"' + temp_code + '", "' + now_name +'")'

            ctxt = code[head+1:right_index + 1]
            add_model_result = get_model_from_code(ctxt, origin_code,line,ins_num)
            result.add(add_model_result[0])
            origin_code = add_model_result[1]
            add_line = add_model_result[2]
            # print(add_model_result[1])
            
            line += 1
            line += add_line
            origin_code = insert_one_line_in_code(origin_code, line, need2print)
            line += 1
            ins_num += 1
            origin_code = insert_one_line_in_code(origin_code, line, 'mdtypes_' + str(ins_num) + '  = set()')
            add = True
        elif m_type == 3:
            now_len = len(now_key)-1
            add = True
            index = code.find(now_key)
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
                need2print = "insert_result(" + str(model_id) + "," + code[head:right_index + 1] + ',"' + temp_code + '", "' + now_name + '")'

                ctxt = code[head:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code,line,ins_num)
                result.add(add_model_result[0])
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # print(add_model_result[1])
                
                line += 1
                line += add_line
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                line += 1
                ins_num += 1
                origin_code = insert_one_line_in_code(origin_code, line, 'mdtypes_' + str(ins_num) + '  = set()')

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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head:right_index + 1] + ',"' + temp_code + '", "' + now_name + '")'
                ctxt = code[head:right_index + 1]
                add_model_result = get_model_from_code(ctxt, origin_code,line,ins_num)
                result.add(add_model_result[0])
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # print(add_model_result[1])
                
                line += 1
                line += add_line
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                line += 1
                ins_num += 1
                origin_code = insert_one_line_in_code(origin_code, line, 'mdtypes_' + str(ins_num) + '  = set()')

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
                need2print = "insert_result(" + str(model_id) + "," + code[
                                                                      head+1:right_index + 1] + ',"' + temp_code + '", "' + now_name +'")'
                ctxt =  code[head+1:right_index + 1]
                add_model_result = get_model_from_code(ctxt,origin_code,line,ins_num)
                result.add(add_model_result[0])
                origin_code = add_model_result[1]
                add_line = add_model_result[2]
                # print(add_model_result[1])
                # get_one_result
                line += 1
                line += add_line
                origin_code = insert_one_line_in_code(origin_code, line, need2print)
                line += 1
                ins_num += 1
                origin_code = insert_one_line_in_code(origin_code, line, 'mdtypes_' + str(ins_num) + '  = set()')
        line += 1
    # print(origin_code)
    return add_model_result[1], add,result

def batch_running(ip,notebook_root='../notebook',):
    in_result = []
    cursor, db = create_connection()
    sql = 'SELECT notebook_id from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(int(row[0]))

    sql = 'SELECT id from notebook where add_run=1 and server_ip=\'' + ip + "'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    all = 0
    can_use = 0
    can_use_1 = 0
    for row in sql_res:
        notebook_id = int(row[0])
        if notebook_id not in in_result:
            continue
        try:
            origin_code = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
        except Exception as e:
            print(e)
            return "read fail"
        origin_code,add,result = add_result(notebook_id,origin_code)
        # print(type(result))
        # print(result)
        if len(result) == 0:
            can_use += 1
            update_db("notebook", "add_model", '1', 'id', "=", notebook_id)
            update_db("result", "model_type", "'unknown'", 'notebook_id', "=", notebook_id)
        if len(result) == 1:
            can_use += 1
            update_db("notebook", "add_model", '1', 'id', "=", notebook_id)
            sql = 'UPDATE result SET model_type = \'' + list(result)[0]+ "' WHERE notebook_id=" + str(notebook_id)
            cursor.execute(sql)
            # print('delete id:' + str(notebook_id))
        if len(result) > 1 :
            print(result)
            update_db("notebook", "add_model", '2', 'id', "=", notebook_id)
            sql = 'delete from result where notebook_id=' + str(notebook_id)
            cursor.execute(sql)
            db.commit()
            print('delete id:' + str(notebook_id))

            can_use_1 += 1
        all += 1
    print('1:',can_use)
    print('2:',can_use_1)
    print('all:',all)
    print('rate:', can_use/all)

def single_add_model(notebook_root, notebook_id, ip):
    origin_code = get_code_txt(notebook_root + '/' + str(notebook_id) + '.ipynb')
    origin_code, add, result = add_result(notebook_id, origin_code)
    cursor, db = create_connection()
    sql = "SELECT pair.nid, dataset.dataSourceUrl, notebook.server_ip " \
          "FROM pair, notebook, dataset " \
          "WHERE notebook.id=pair.nid " \
          "and dataset.id=pair.did " \
          "and (notebook.add_sequence=1 or notebook.add_sequence=0 and (notebook.cant_sequence=2 or notebook.cant_sequence=3)) " \
          "and dataset.isdownload=1 " \
          "and dataset.server_ip='" + ip + "' " \
          "and notebook.id="+str(notebook_id)
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        dataset_name = row[1].split('/')[-1].strip()
        break
    dataset_path_root = '../spider/unzip_dataset/' + dataset_name + '.zip'

    code_list = origin_code.split('\n')
    for index,line in enumerate(code_list):
        print(index,line)
    # can_run_code = running(notebook_id, origin_code, dataset_path_root, 0)

    # print(operator_list)


if __name__ == '__main__':
    # origin_code = add_finish_data(1413709,1)
    notebook_id = 1012468

    # walklogs = np.load('../walklogs/1413709.npy', allow_pickle=True).item()
    # print(walklogs['data_values'])
    # save_dataframe_and_update_reuslt(notebook_id)
    origin_code = get_code_txt('../spider/notebook/' + str(notebook_id) + '.ipynb')
    code_list = origin_code.split('\n')
    for index, line in enumerate(code_list):
        print(index, line)
    res = get_dataframe_and_operator_list_from_one_result(notebook_id, origin_code, 1, 1)
    print(res[1])
    print(res[2])

        # break


