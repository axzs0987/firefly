from utils import create_connection
import pymysql
import numpy as np
from utils import CONFIG

def get_one_reuslt(notebook_id):
    def deal_content(content,metric_type):
        if type(content).__name__ == 'float' or type(content).__name__ == 'int':
            if content > 0 and content < 1:
                return (content,metric_type)
            else:
                return (-1,metric_type)
        else:
            return (-1,metric_type)
    def deal_str_content(str_content,metric_type):
        if metric_type == 'cross_val_score':
            str_content = str_content[1:-1]
            score_list = str_content.split(' ')
            score = 0
            count = 0
            for i in score_list:
                try:
                    if float(i) > 0 and float(i) < 1:
                        score += float(i)
                        count += 1
                except:
                    continue
            if count != 0:
                return (score/count,metric_type)
            else:
                return (-1,metric_type)
        else:
            return (-1,metric_type)
    cursor, db = create_connection()
    # 找到所有datasetid的id和title
    sql = "SELECT distinct content,metric_type from result where notebook_id=" + str(notebook_id);
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    # score = 0
    # count = 0
    count = {}
    score = {}
    for row in sql_res:
        dc = deal_content(row[0], row[1])
        temp_score = dc[0]
        if temp_score != -1:
            if dc[1] not in score.keys():
                score[dc[1]] = 0
            if dc[1] not in count.keys():
                count[dc[1]] = 0
            score[dc[1]] += temp_score
            count[dc[1]] += 1
    sql = "SELECT distinct str_content,metric_type from result where id not in (select id from result where isnull(str_content)) and notebook_id=" + str(notebook_id);
    cursor.execute(sql)
    sql_res_1 = cursor.fetchall()

    for row in sql_res_1:
        dsc = deal_str_content(row[0],row[1])
        temp_score = dsc[0]
        if temp_score != -1:
            if dsc[1] not in score.keys():
                score[dsc[1]] = 0
            if dsc[1] not in count.keys():
                count[dsc[1]] = 0
            score[dsc[1]] += temp_score
            count[dsc[1]] += 1

    result = {}
    for i in score:
        if count[i] != 0 and score[i] != -1:
            result[i] = score[i]/count[i]
    return result

def get_all_mean_of_exist():
    cursor, db = create_connection()
    sql = "SELECT id FROM notebook WHERE id in (select distinct notebook_id from operator)"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    count = {}
    has_score = {}
    for row in sql_res:
        temp_score_result = get_one_reuslt(row[0])
        for i in temp_score_result:
            if temp_score_result[i] != -1:
                if i not in has_score:
                    has_score[i]=0
                if i not in count:
                    count[i]=0
                has_score[i] += temp_score_result[i]
                count[i] += 1

    has_result = {}
    for i in has_score:
        if count[i] != 0 and has_score[i] != -1:
            has_result[i] = has_score[i]/count[i]

    cursor, db = create_connection()
    sql = "SELECT id FROM notebook WHERE cant_sequence=2"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    nhas_score = {}
    count = {}
    for row in sql_res:
        temp_score_result = get_one_reuslt(row[0])
        for i in temp_score_result:
            if temp_score_result[i] != -1:
                if i not in nhas_score:
                    nhas_score[i] = 0
                if i not in count:
                    count[i] = 0
                nhas_score[i] += temp_score_result[i]
                count[i] += 1

    nhas_result = {}
    for i in nhas_score:
        if count[i] != 0 and nhas_score[i] != -1:
            nhas_result[i] = nhas_score[i] / count[i]
    return nhas_result,has_result

def get_operator_comp_group_by_dataset():
    dataset_temp_score = {}
    dataset_temp_score_1 = {}

    cursor, db = create_connection()
    dataset_dic = np.load('./dataset_score_dic.npy',allow_pickle=True).item()
    sql = "SELECT * from pair,operator where pair.nid in (select distinct notebook_id from operator)"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    for row in sql_res:
        notebook_id = row[0]
        dataset_id = row[1]
        result = get_one_reuslt(notebook_id)
        if dataset_id not in dataset_temp_score.keys():
            dataset_temp_score[dataset_id] = {}
        # all_score = dataset_dic[dataset_id][0]
        # all_count = dataset_dic[dataset_id][1]
        for i in result:
            if result[i] != -1:
                if i not in dataset_temp_score[dataset_id]:
                    dataset_temp_score[dataset_id][i] = (0,0)
                all_score = dataset_temp_score[dataset_id][i][0]
                all_count = dataset_temp_score[dataset_id][i][1]
                all_score += result[i]
                all_count += 1
                dataset_temp_score[dataset_id][i] = (all_score, all_count)
    for i in dataset_temp_score:
        for j in dataset_temp_score[i]:
            all_score = dataset_temp_score[i][j][0]
            all_count = dataset_temp_score[i][j][1]
            if all_count == 0:
                dataset_temp_score[i][j] = -1
            else:
                dataset_temp_score[i][j] = all_score/all_count
    cursor, db = create_connection()
    sql = "SELECT * from pair,notebook where pair.nid=notebook.id and notebook.cant_sequence=2"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    for row in sql_res:
        notebook_id = row[0]
        dataset_id = row[1]
        result = get_one_reuslt(notebook_id)
        if dataset_id not in dataset_temp_score_1.keys():
            dataset_temp_score_1[dataset_id] = {}
        # all_score = dataset_dic[dataset_id][0]
        # all_count = dataset_dic[dataset_id][1]
        for i in result:
            if result[i] != -1:
                if i not in dataset_temp_score_1[dataset_id]:
                    dataset_temp_score_1[dataset_id][i] = (0,0)
                all_score = dataset_temp_score_1[dataset_id][i][0]
                all_count = dataset_temp_score_1[dataset_id][i][1]
                all_score += result[i]
                all_count += 1
                dataset_temp_score_1[dataset_id][i] = (all_score, all_count)
    for i in dataset_temp_score_1:
        for j in dataset_temp_score_1[i]:
            all_score = dataset_temp_score_1[i][j][0]
            all_count = dataset_temp_score_1[i][j][1]
            if all_count == 0:
                dataset_temp_score_1[i][j] = -1
            else:
                dataset_temp_score_1[i][j] = all_score/all_count

    result = {}
    for i in dataset_dic:
        for j in dataset_dic:
            a = dataset_dic[i][j]
            b,c=0
            if i not in dataset_temp_score_1:
                c=-1
            elif j not in dataset_temp_score_1[i]:
                c=-1
            else:
                c=dataset_temp_score_1[i][j]

            if i not in dataset_temp_score:
                b=-1
            elif j not in dataset_temp_score[i]:
                b=-1
            else:
                b=dataset_temp_score[i][j]

            result[i] = (a,b,c)

    np.save('./exist_operator_groupby_dataset.npy',result)
    return result

def get_mean_group_by_dataset():
    cursor, db = create_connection()
    sql = "SELECT * from pair"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    dataset_dic = {}
    for row in sql_res:
        notebook_id = row[0]
        dataset_id = row[1]
        result = get_one_reuslt(notebook_id)
        if dataset_id not in dataset_dic.keys():
            dataset_dic[dataset_id] = {}
        # all_score = dataset_dic[dataset_id][0]
        # all_count = dataset_dic[dataset_id][1]
        for i in result:
            if result[i] != -1:
                if i not in dataset_dic[dataset_id]:
                    dataset_dic[dataset_id][i] = (0,0)
                all_score = dataset_dic[dataset_id][i][0]
                all_count = dataset_dic[dataset_id][i][1]
                all_score += result[i]
                all_count += 1
                dataset_dic[dataset_id][i] = (all_score, all_count)


    for i in dataset_dic:
        for j in dataset_dic[i]:
            all_score = dataset_dic[i][j][0]
            all_count = dataset_dic[i][j][1]
            if all_count == 0:
                dataset_dic[i][j] = -1
            else:
                dataset_dic[i][j] = all_score/all_count
    np.save('./dataset_score_dic.npy', dataset_dic)
    return dataset_dic

# def get_all_score():
#     cursor, db = create_connection()
#     sql = "SELECT distinct notebook_id FROM result"
#     cursor.execute(sql)
#     sql_res = cursor.fetchall()
#     notebook_score = {}
#     for row in sql_res:
#         notebook_id = row[0]
#         result = get_one_reuslt(notebook_id)
#         if notebook_id not in notebook_score.keys():
#             notebook_score[notebook_id] = {}
#         for i in result:
#             if result[i] != -1:
#                 if i not in notebook_score[notebook_id]:
#                     notebook_score[notebook_id][i] = (0, 0)
#                 all_score = notebook_score[notebook_id][i][0]
#                 all_count = notebook_score[notebook_id][i][1]
#                 all_score += result[i]
#                 all_count += 1
#                 notebook_score[notebook_id][i] = (all_score, all_count)
#
#     for i in notebook_score:
#         for j in notebook_score[i]:
#             all_score = notebook_score[i][j][0]
#             all_count = notebook_score[i][j][1]
#             if all_count == 0:
#                 notebook_score[i][j] = -1
#             else:
#                 notebook_score[i][j] = all_score/all_count
#     np.save('./notebook_score.npy', notebook_score)
#     return notebook_score

def get_operator_param_score():
    cursor, db = create_connection()
    CONFIG.read('config.ini')
    operator_dic = eval(CONFIG.get('operators', 'operations'))
    ope_dic = {}
    sql = "SELECT notebook_id,operator,parameter_1_value,parameter_2_value,parameter_3_value,parameter_4_value,parameter_5_value,parameter_6_value,parameter_7_value FROM operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    notebook_score = {}

    parameter_dic = {}
    for row in sql_res:
        if row[0] not in notebook_score:
            result = get_one_reuslt(row[0])
            notebook_score[row[0]] = result
        result = notebook_score[row[0]]
        if row[1] not in parameter_dic:
            parameter_dic[row[1]] = {}
        for num in range(2,9):
            if row[num] not in parameter_dic[row[1]]:
                parameter_dic[row[1]][row[num]] = {}
            for i in result:
                if result[i] != -1:
                    if i not in parameter_dic[row[1]][row[num]]:
                        parameter_dic[row[1]][row[num]][i] = (0, 0)
                    all_score = parameter_dic[row[1]][row[num]][i][0]
                    all_count = parameter_dic[row[1]][row[num]][i][1]
                    all_score += result[i]
                    all_count += 1
                    parameter_dic[row[1]][row[num]][i] = (all_score, all_count)

    for i in parameter_dic: # operator
        for j in parameter_dic[i]: # parameter
            for k in parameter_dic[i][j]: # score type
                all_score = parameter_dic[i][j][k][0]
                all_count = parameter_dic[i][j][k][1]
                if all_count == 0:
                    parameter_dic[i][j][k] = -1
                else:
                    parameter_dic[i][j][k] = all_score / all_count


    np.save('./param_score_dic.npy', ope_dic)

def get_physic_operator_score():
    cursor, db = create_connection()
    CONFIG.read('config.ini')
    operator_dic = eval(CONFIG.get('operators', 'operations'))
    ope_dic = {}
    for operator in operator_dic.keys():
        sql = "SELECT distinct notebook_id FROM operator where operator = '" + i + "'"
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            notebook_id = row[0]
            result = get_one_reuslt(notebook_id)
            if operator not in ope_dic.keys():
                ope_dic[operator] = {}
            for i in result:
                if result[i] != -1:
                    if i not in ope_dic[operator]:
                        ope_dic[operator][i] = (0, 0)
                    all_score = ope_dic[operator][i][0]
                    all_count = ope_dic[operator][i][1]
                    all_score += result[i]
                    all_count += 1
                    ope_dic[operator][i] = (all_score, all_count)

    for i in ope_dic:
        for j in ope_dic[i]:
            all_score = ope_dic[i][j][0]
            all_count = ope_dic[i][j][1]
            if all_count == 0:
                ope_dic[i][j] = -1
            else:
                ope_dic[i][j] = all_score/all_count
    np.save('./ope_score_dic.npy', ope_dic)

if __name__ == '__main__':
    # print(get_all_mean_of_exist())
    print('input stat type:')
    print('1: get_all_mean_of_exist')
    print('2: get_mean_group_by_dataset')
    print('3: get_operator_comp_group_by_dataset')
    print('4: get_physic_operator_score')
    print('5: get_operator_param_score')

    type = input()

    if type == '1':
        res = get_all_mean_of_exist()
    elif type == '2':
        res = get_mean_group_by_dataset()
    elif type == '3':
        res = get_operator_comp_group_by_dataset()
    elif type == '4':
        res = get_physic_operator_score()
    elif type == '5':
        res = get_operator_param_score()
    # res = get_mean_group_by_dataset()
    # for i in res:
    #     #     if res[i] != -1:
    #     #         print(str(i)+':', res[i])