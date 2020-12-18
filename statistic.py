from utils import create_connection
import pymysql
import numpy as np
from utils import CONFIG
import os
import ast
import astunparse

def get_one_model_result(notebook_id):
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
    sql = "SELECT distinct content,metric_type,model_type from result where notebook_id=" + str(notebook_id);
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    # score = 0
    # count = 0
    count = {}
    score = {}
    model_dic = {}
    for row in sql_res:
        dc = deal_content(row[0], row[1])
        temp_score = dc[0]
        if row[2] not in model_dic:
            model_dic[row[2]] = {}

        if temp_score != -1:
            if dc[1] not in model_dic[row[2]].keys():
                model_dic[row[2]][dc[1]] = [0,0]

            model_dic[row[2]][dc[1]][0] += temp_score
            model_dic[row[2]][dc[1]][1] += 1
    sql = "SELECT distinct str_content,metric_type,model_type from result where id not in (select id from result where isnull(str_content)) and notebook_id=" + str(notebook_id);
    cursor.execute(sql)
    sql_res_1 = cursor.fetchall()

    for row in sql_res_1:
        dsc = deal_str_content(row[0],row[1])
        temp_score = dsc[0]
        if row[2] not in model_dic:
            model_dic[row[2]] = {}

        if temp_score != -1:
            if dsc[1] not in score.keys():
                model_dic[row[2]][dsc[1]] = [0,0]

            model_dic[row[2]][dsc[1]][0] += temp_score
            model_dic[row[2]][dsc[1]][1] += 1

    # result = {}
    for model in model_dic:
        for score in model_dic[model]:
            if model_dic[model][score][1] != 0 and model_dic[model][score][0] != -1:
                model_dic[model][score] = model_dic[model][score][0]/model_dic[model][score][1]

    # print(model_dic)
    return model_dic



def get_one_seq_len(notebook_id):
    cursor, db = create_connection()
    # 找到所有datasetid的id和title
    sql = "SELECT count(id) from operator where notebook_id=" + str(notebook_id);
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        return row[0]

def get_one_result(notebook_id):
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

def get_exist_dic():
    """
    :return:
        (has_result:
    {
        'auc': ([0.98, 0.96, ..., 0.91], 100),
        'f1_score': ([0.95, 0.92, ..., 0.98], 12),
        ...,
    },
    nhas_result:
    {
        'auc': ([0.98, 0.96, ..., 0.91], 100),
        'f1_score': ([0.95, 0.92, ..., 0.98], 12),
        ...,
    })
    """
    in_result = []
    cursor, db = create_connection()
    sql = "select distinct notebook_id from result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    sql = "select distinct notebook_id from operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    has_count = {}
    has_score = {}
    nhas_score = {}
    nhas_count = {}
    has_score_list = []
    for row in sql_res:
        has_score_list.append(row[0])

    has_f1_score_list = []
    nhas_f1_score_list = []
    has_score_list_ = []
    nhas_score_list_ = []
    has_cross_val_score_list = []
    nhas_cross_val_score_list = []
    has_accuracy_score_list = []
    nhas_accuracy_score_list = []
    has_roc_auc_score_list = []
    nhas_roc_auc_score_list = []
    has_precision_score_list = []
    nhas_precision_score_list = []
    has_recall_score_list = []
    nhas_recall_score_list = []
    has_best_score_list = []
    nhas_best_score_list = []
    has_auc_list = []
    nhas_auc_list = []

    for notebook_id in in_result:
        if notebook_id in has_score_list:
            temp_score_result = get_one_result(notebook_id)
            for i in temp_score_result:
                if temp_score_result[i] != -1:
                    if i == 'f1_score':
                        has_f1_score_list.append(temp_score_result[i])
                    if i == 'score':
                        has_score_list_.append(temp_score_result[i])
                    if i == 'cross_val_score':
                        has_cross_val_score_list.append(temp_score_result[i])
                    if i == 'accuracy_score':
                        has_accuracy_score_list.append(temp_score_result[i])
                    if i == 'roc_auc_score':
                        has_roc_auc_score_list.append(temp_score_result[i])
                    if i == 'precision_score':
                        has_precision_score_list.append(temp_score_result[i])
                    if i == 'recall_score':
                        has_recall_score_list.append(temp_score_result[i])
                    if i == 'best_score_':
                        has_best_score_list.append(temp_score_result[i])
                    if i == 'auc':
                        has_auc_list.append(temp_score_result[i])
                    if i not in has_score:
                        has_score[i] =[]
                    if i not in has_count:
                        has_count[i] = 0
                    has_score[i].append(temp_score_result[i])
                    has_count[i] += 1

        else:
            temp_score_result = get_one_result(notebook_id)
            for i in temp_score_result:
                if temp_score_result[i] != -1:
                    if i == 'f1_score':
                        nhas_f1_score_list.append(temp_score_result[i])
                    if i == 'score':
                        nhas_score_list_.append(temp_score_result[i])
                    if i == 'cross_val_score':
                        nhas_cross_val_score_list.append(temp_score_result[i])
                    if i == 'accuracy_score':
                        nhas_accuracy_score_list.append(temp_score_result[i])
                    if i == 'roc_auc_score':
                        nhas_roc_auc_score_list.append(temp_score_result[i])
                    if i == 'precision_score':
                        nhas_precision_score_list.append(temp_score_result[i])
                    if i == 'recall_score':
                        nhas_recall_score_list.append(temp_score_result[i])
                    if i == 'best_score_':
                        nhas_best_score_list.append(temp_score_result[i])
                    if i == 'auc':
                        nhas_auc_list.append(temp_score_result[i])
                    if i not in nhas_score:
                        nhas_score[i] = []
                    if i not in nhas_count:
                        nhas_count[i] = 0
                    nhas_score[i].append(temp_score_result[i])
                    nhas_count[i] += 1
    has_result = {}
    for i in has_score:
        if has_count[i] != 0 and has_score[i] != []:
            sum = 0
            for sc in has_score[i]:
                sum += sc
            mean = sum/has_count[i]
            sq = 0
            for sc in has_score[i]:
                sq += (sc-mean)**2
            sq /= has_count[i]
            has_result[i] = (mean, sq, has_count[i])
        else:
            has_result[i] = (-1, -1, 0)
    nhas_result = {}
    for i in nhas_score:
        if nhas_count[i] != [] and nhas_score[i] != -1:
            sum = 0
            for sc in nhas_score[i]:
                sum += sc
            mean = sum / nhas_count[i]
            sq = 0
            for sc in nhas_score[i]:
                sq += (sc - mean)**2
            sq /= nhas_count[i]
            nhas_result[i] = (mean, sq, nhas_count[i])
        else:
            nhas_result[i] = (-1, -1, 0)

    np.save('has_f1_score_list',has_f1_score_list)
    np.save('nhas_f1_score_list', nhas_f1_score_list)
    np.save('has_score_list_', has_score_list_)
    np.save('nhas_score_list_', nhas_score_list_)
    np.save('has_cross_val_score_list', has_cross_val_score_list)
    np.save('nhas_cross_val_score_list', nhas_cross_val_score_list)
    np.save('has_accuracy_score_list', has_accuracy_score_list)
    np.save('nhas_accuracy_score_list', nhas_accuracy_score_list)
    np.save('has_roc_auc_score_list', has_roc_auc_score_list)
    np.save('nhas_roc_auc_score_list', nhas_roc_auc_score_list)
    np.save('has_precision_score_list', has_precision_score_list)
    np.save('nhas_precision_score_list', nhas_precision_score_list)
    np.save('has_recall_score_list', has_recall_score_list)
    np.save('nhas_recall_score_list', nhas_recall_score_list)
    np.save('has_best_score_list', has_best_score_list)
    np.save('nhas_best_score_list', nhas_best_score_list)
    np.save('has_auc_list', has_auc_list)
    np.save('nhas_auc_list', nhas_auc_list)

    np.save('./0d0mgap1_.npy',{'has_ope':has_result,'nhas_ope':nhas_result})
    return (has_result,nhas_result)


# def get_exit_dic_len():
#     in_result = []
#     cursor, db = create_connection()
#     sql = "select distinct notebook_id from result"
#     cursor.execute(sql)
#     sql_res = cursor.fetchall()
#     for row in sql_res:
#         in_result.append(int(row[0]))
#
#     sql = "select distinct notebook_id from operator"
#     cursor.execute(sql)
#     sql_res = cursor.fetchall()
#     has_count = 0
#     has_ope_notebooks = []
#     has_ope_notebooks = []
#     nhas_count = 0
#     has_score_list = []
#     for row in sql_res:
#         if int(row[0]) in in_result:
#             has_ope_notebooks.append(int(row[0]))
#
#     for notebook in has_ope_notebooks:
#         sql = "select rank from operator where notebook_id="+str(notebook)
#         cursor.execute(sql)
#         sql_res = cursor.fetchall()
#         max_rank = 0
#         for row in sql_res:
#             max_rank += 1
def get_exist_dic_len():
    """
    :return:
        (has_result:
    {
        'auc': ([0.98, 0.96, ..., 0.91], 100),
        'f1_score': ([0.95, 0.92, ..., 0.98], 12),
        ...,
    },
    nhas_result:
    {
        'auc': ([0.98, 0.96, ..., 0.91], 100),
        'f1_score': ([0.95, 0.92, ..., 0.98], 12),
        ...,
    })
    """
    in_result = []
    cursor, db = create_connection()
    sql = "select distinct notebook_id from result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    sql = "select distinct notebook_id from operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    has_count = 0
    has_score = []
    nhas_score = []
    nhas_count = 0
    has_score_list = []
    for row in sql_res:
        has_score_list.append(row[0])

    for notebook_id in in_result:
        if notebook_id in has_score_list:
            temp_score_result = get_one_seq_len(notebook_id)
            has_score.append(temp_score_result)
            has_count += 1

        else:
            temp_score_result = get_one_seq_len(notebook_id)
            nhas_score.append(temp_score_result)
            nhas_count += 1

    if has_count != 0 and has_score != []:
        sum = 0
        for sc in has_score:
            sum += sc
        mean = sum/has_count
        sq = 0
        for sc in has_score:
            sq += (sc-mean)**2
        sq /= has_count
        has_result = (mean, sq, has_count)
    else:
        has_result = (-1, -1, 0)

    if nhas_count != [] and nhas_score != -1:
        sum = 0
        for sc in nhas_score:
            sum += sc
        mean = sum / nhas_count
        sq = 0
        for sc in nhas_score:
            sq += (sc - mean)**2
        sq /= nhas_count
        nhas_result = (mean, sq, nhas_count)
    else:
        nhas_result = (-1, -1, 0)

    np.save('./0d0mgap2_.npy',{'has_operator':has_result,'nhas_operator': nhas_result})
    return (has_result,nhas_result)

def get_dataset_exist_dic():
    dataset_temp_score = {}
    dataset_temp_score_1 = {}

    cursor, db = create_connection()
    dataset_dic = np.load('./dataset_score_dic.npy',allow_pickle=True).item()

    print('get operator list and result list')
    sql = "SELECT distinct notebook_id from operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    has_operator_list = []
    count= 0

    for row in sql_res:
        count+=1
        has_operator_list.append(row[0])

    np.save('has_operator_list.npy',has_operator_list)
    sql = "SELECT distinct notebook_id from result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    has_result_list = []
    count = 0
    for row in sql_res:
        count += 1
        has_result_list.append(row[0])
    print(has_operator_list)
    print(has_result_list)
    print('get operator list and result list end')
    print('get pairs')
    sql = "SELECT pair.nid,pair.did from pair inner join notebook on pair.nid=notebook.id where notebook.add_run=1"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    print('get pairs end')
    notebook_id_list = []


    has_f1_score_list = {}
    has_score_list_ = {}
    has_cross_val_score_list = {}
    has_accuracy_score_list = {}
    has_roc_auc_score_list = {}
    has_precision_score_list = {}
    has_recall_score_list = {}
    has_best_score_list = {}
    has_auc_list = {}

    nhas_f1_score_list = {}
    nhas_score_list_ = {}
    nhas_cross_val_score_list = {}
    nhas_accuracy_score_list = {}
    nhas_roc_auc_score_list = {}
    nhas_precision_score_list = {}
    nhas_recall_score_list = {}
    nhas_best_score_list = {}
    nhas_auc_list = {}

    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id in notebook_id_list:
            # print('already in')
            continue
        notebook_id_list.append(notebook_id)
        if notebook_id not in has_result_list:
            # print('not in result')
            continue
        if notebook_id not in has_operator_list:
            print('not has operator')
            result = get_one_result(notebook_id)
            if dataset_id not in dataset_temp_score_1.keys():
                dataset_temp_score_1[dataset_id] = {}
            # all_score = dataset_dic[dataset_id][0]
            # all_count = dataset_dic[dataset_id][1]
            for i in result:
                if result[i] != -1:
                    if i not in dataset_temp_score_1[dataset_id]:
                        dataset_temp_score_1[dataset_id][i] = ([], 0)
                    if i == 'f1_score':
                        if dataset_id not in has_f1_score_list:
                            has_f1_score_list[dataset_id] = []
                        has_f1_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    if i == 'score':
                        if dataset_id not in has_score_list_:
                            has_score_list_[dataset_id] = []
                        has_score_list_[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    if i == 'cross_val_score':
                        if dataset_id not in has_cross_val_score_list:
                            has_cross_val_score_list[dataset_id] = []
                        has_cross_val_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    if i == 'accuracy_score':
                        if dataset_id not in has_accuracy_score_list:
                            has_accuracy_score_list[dataset_id] = []
                        has_accuracy_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    if i == 'roc_auc_score':
                        if dataset_id not in has_roc_auc_score_list:
                            has_roc_auc_score_list[dataset_id] = []
                        has_roc_auc_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    # if i == 'precision_score':
                    #     if dataset_id not in has_precision_score_list:
                    #         has_precision_score_list[dataset_id] = []
                    #     has_precision_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    # if i == 'recall_score':
                    #     if dataset_id not in has_recall_score_list:
                    #         has_recall_score_list[dataset_id] = []
                    #     has_recall_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    if i == 'best_score_':
                        if dataset_id not in has_best_score_list:
                            has_best_score_list[dataset_id] = []
                        has_best_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    # if i == 'auc':
                    #     if dataset_id not in has_auc_list:
                    #         has_auc_list[dataset_id] = []
                    #     has_auc_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                    all_score = dataset_temp_score_1[dataset_id][i][0]
                    all_count = dataset_temp_score_1[dataset_id][i][1]
                    all_score.append(result[i])
                    all_count += 1
                    dataset_temp_score_1[dataset_id][i] = (all_score, all_count)
                    print(dataset_id, dataset_temp_score_1[dataset_id][i])
            continue
        print('has operator')
        result = get_one_result(notebook_id)
        if dataset_id not in dataset_temp_score.keys():
            dataset_temp_score[dataset_id] = {}
        # all_score = dataset_dic[dataset_id][0]
        # all_count = dataset_dic[dataset_id][1]
        for i in result:
            if result[i] != -1:
                if i not in dataset_temp_score[dataset_id]:
                    dataset_temp_score[dataset_id][i] = ([],0)
                if i == 'f1_score':
                    if dataset_id not in nhas_f1_score_list:
                        nhas_f1_score_list[dataset_id] = []
                    nhas_f1_score_list[dataset_id].append(dataset_temp_score[dataset_id][i])
                if i == 'score':
                    if dataset_id not in nhas_score_list_:
                        nhas_score_list_[dataset_id] = []
                    nhas_score_list_[dataset_id].append(dataset_temp_score[dataset_id][i])
                if i == 'cross_val_score':
                    if dataset_id not in nhas_cross_val_score_list:
                        nhas_cross_val_score_list[dataset_id] = []
                    nhas_cross_val_score_list[dataset_id].append(dataset_temp_score[dataset_id][i])
                if i == 'accuracy_score':
                    if dataset_id not in nhas_accuracy_score_list:
                        nhas_accuracy_score_list[dataset_id] = []
                    nhas_accuracy_score_list[dataset_id].append(dataset_temp_score[dataset_id][i])
                if i == 'roc_auc_score':
                    if dataset_id not in nhas_roc_auc_score_list:
                        nhas_roc_auc_score_list[dataset_id] = []
                    nhas_roc_auc_score_list[dataset_id].append(dataset_temp_score[dataset_id][i])
                # if i == 'precision_score':
                #     if dataset_id not in has_precision_score_list:
                #         has_precision_score_list[dataset_id] = []
                #     has_precision_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                # if i == 'recall_score':
                #     if dataset_id not in has_recall_score_list:
                #         has_recall_score_list[dataset_id] = []
                #     has_recall_score_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                if i == 'best_score_':
                    if dataset_id not in nhas_best_score_list:
                        nhas_best_score_list[dataset_id] = []
                    nhas_best_score_list[dataset_id].append(dataset_temp_score[dataset_id][i])
                # if i == 'auc':
                #     if dataset_id not in has_auc_list:
                #         has_auc_list[dataset_id] = []
                #     has_auc_list[dataset_id].append(dataset_temp_score_1[dataset_id][i])
                all_score = dataset_temp_score[dataset_id][i][0]
                all_count = dataset_temp_score[dataset_id][i][1]
                all_score.append(result[i])
                all_count += 1
                dataset_temp_score[dataset_id][i] = (all_score, all_count)
                print(dataset_id, dataset_temp_score[dataset_id][i])

    # for i in dataset_temp_score:
    #     for j in dataset_temp_score[i]:
    #         all_score = dataset_temp_score[i][j][0]
    #         all_count = dataset_temp_score[i][j][1]
    #         if all_count == 0:
    #             dataset_temp_score[i][j] = (-1,0)
    #         else:
    #             dataset_temp_score[i][j] = (all_score/all_count,all_count)
    # for i in dataset_temp_score_1:
    #     for j in dataset_temp_score_1[i]:
    #         all_score = dataset_temp_score_1[i][j][0]
    #         all_count = dataset_temp_score_1[i][j][1]
    #         if all_count == 0:
    #             dataset_temp_score_1[i][j] = (-1,0)
    #         else:
    #             dataset_temp_score_1[i][j] = (all_score/all_count,all_count)

    result = {}
    for i in dataset_dic:
        result[i] = {}
        for j in dataset_dic[i]:
            # try:
            #     a = dataset_dic[i][j][0]
            # except:
            #     continue
            if i not in dataset_temp_score_1:
                c=(-1,-1,0)
            elif j not in dataset_temp_score_1[i]:
                c=(-1,-1,0)
            else:
                if dataset_temp_score_1[i][j][1] == 0:
                    c = (-1, -1, 0)
                else:
                    score_list = dataset_temp_score_1[i][j][0]
                    sum = 0
                    for sc in score_list:
                        sum += sc
                    mean = sum/dataset_temp_score_1[i][j][1]
                    sq = 0
                    for sc in score_list:
                        sq += (sc-mean)**2
                    sq = sq / dataset_temp_score_1[i][j][1]
                    c = (mean, sq, dataset_temp_score_1[i][j][1])

            if i not in dataset_temp_score:
                b=(-1,-1,0)
            elif j not in dataset_temp_score[i]:
                b=(-1,-1,0)
            else:
                if dataset_temp_score[i][j][1] == 0:
                    b = (-1, -1, 0)
                else:
                    score_list = dataset_temp_score[i][j][0]
                    sum = 0
                    for sc in score_list:
                        sum += sc
                    mean = sum / dataset_temp_score[i][j][1]
                    sq = 0
                    for sc in score_list:
                        sq += (sc - mean)**2
                    sq = sq / dataset_temp_score[i][j][1]
                    b = (mean, sq, dataset_temp_score[i][j][1])

            result[i][j] = (b,c)

    np.save('1d0mgap1data/dataset_has_f1_score_list', has_f1_score_list)
    np.save('1d0mgap1data/dataset_nhas_f1_score_list', nhas_f1_score_list)
    np.save('1d0mgap1data/dataset_has_score_list_', has_score_list_)
    np.save('1d0mgap1data/dataset_nhas_score_list_', nhas_score_list_)
    np.save('1d0mgap1data/dataset_has_cross_val_score_list', has_cross_val_score_list)
    np.save('1d0mgap1data/dataset_nhas_cross_val_score_list', nhas_cross_val_score_list)
    np.save('1d0mgap1data/dataset_has_accuracy_score_list', has_accuracy_score_list)
    np.save('1d0mgap1data/dataset_nhas_accuracy_score_list', nhas_accuracy_score_list)
    np.save('1d0mgap1data/dataset_has_roc_auc_score_list', has_roc_auc_score_list)
    np.save('1d0mgap1data/dataset_nhas_roc_auc_score_list', nhas_roc_auc_score_list)
    np.save('1d0mgap1data/dataset_has_best_score_list', has_best_score_list)
    np.save('1d0mgap1data/dataset_nhas_best_score_list', nhas_best_score_list)

    np.save('./exist_operator_groupby_dataset.npy',result)
    return result
def get_dataset_exist_dic_len():
    dataset_temp_score = {}
    dataset_temp_score_1 = {}

    cursor, db = create_connection()
    dataset_dic = np.load('./dataset_score_dic.npy',allow_pickle=True).item()

    print('get operator list and result list')
    sql = "SELECT distinct notebook_id from operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    has_operator_list = []
    count= 0

    for row in sql_res:
        count+=1
        has_operator_list.append(row[0])

    np.save('has_operator_list.npy',has_operator_list)
    sql = "SELECT distinct notebook_id from result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    has_result_list = []
    count = 0
    for row in sql_res:
        count += 1
        has_result_list.append(row[0])
    print(has_operator_list)
    print(has_result_list)
    print('get operator list and result list end')
    print('get pairs')
    sql = "SELECT pair.nid,pair.did from pair inner join notebook on pair.nid=notebook.id where notebook.add_run=1"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    print('get pairs end')
    notebook_id_list = []
    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id in notebook_id_list:
            # print('already in')
            continue
        notebook_id_list.append(notebook_id)
        if notebook_id not in has_result_list:
            # print('not in result')
            continue
        if notebook_id not in has_operator_list:
            print('not has operator')
            result = get_one_seq_len(notebook_id)
            if dataset_id not in dataset_temp_score_1.keys():
                dataset_temp_score_1[dataset_id] =([], 0)
            # all_score = dataset_dic[dataset_id][0]
            # all_count = dataset_dic[dataset_id][1]
            all_score = dataset_temp_score_1[dataset_id][0]
            all_count = dataset_temp_score_1[dataset_id][1]
            all_score.append(result)
            all_count += 1
            dataset_temp_score_1[dataset_id] = (all_score, all_count)
            print(dataset_id, dataset_temp_score_1[dataset_id])
            continue

        print('has operator')
        result = get_one_seq_len(notebook_id)
        if dataset_id not in dataset_temp_score.keys():
            dataset_temp_score[dataset_id] = ([], 0)

        all_score = dataset_temp_score[dataset_id][0]
        all_count = dataset_temp_score[dataset_id][1]
        all_score.append(result)
        all_count += 1
        dataset_temp_score[dataset_id] = (all_score, all_count)
        print(dataset_id, dataset_temp_score[dataset_id])

    # for i in dataset_temp_score:
    #     for j in dataset_temp_score[i]:
    #         all_score = dataset_temp_score[i][j][0]
    #         all_count = dataset_temp_score[i][j][1]
    #         if all_count == 0:
    #             dataset_temp_score[i][j] = (-1,0)
    #         else:
    #             dataset_temp_score[i][j] = (all_score/all_count,all_count)
    # for i in dataset_temp_score_1:
    #     for j in dataset_temp_score_1[i]:
    #         all_score = dataset_temp_score_1[i][j][0]
    #         all_count = dataset_temp_score_1[i][j][1]
    #         if all_count == 0:
    #             dataset_temp_score_1[i][j] = (-1,0)
    #         else:
    #             dataset_temp_score_1[i][j] = (all_score/all_count,all_count)

    result = {}
    for i in dataset_dic:
        result[i] = {}

        if i not in dataset_temp_score_1:
            c=(-1,-1,0)
        else:
            if dataset_temp_score_1[i][1] == 0:
                c = (-1, -1, 0)
            else:
                score_list = dataset_temp_score_1[i][0]
                sum = 0
                for sc in score_list:
                    sum += sc
                mean = sum/dataset_temp_score_1[i][1]
                sq = 0
                for sc in score_list:
                    sq += (sc-mean)**2
                sq = sq / dataset_temp_score_1[i][1]
                c = (mean, sq, dataset_temp_score_1[i][1])

        if i not in dataset_temp_score:
            b=(-1,-1,0)
        else:
            if dataset_temp_score[i][1] == 0:
                c = (-1, -1, 0)
            else:
                score_list = dataset_temp_score[i][0]
                sum = 0
                for sc in score_list:
                    sum += sc
                mean = sum / dataset_temp_score[i][1]
                sq = 0
                for sc in score_list:
                    sq += (sc - mean)**2
                sq = sq / dataset_temp_score[i][1]
                c = (mean, sq, dataset_temp_score[i][1])

        result[i] = (b,c)

    np.save('./exist_operator_groupby_dataset_len.npy',result)
    return result
def get_mean_group_by_dataset():
    """
    :return:
    {
        12:{
            'auc': ([0.97,0.22,...], 21),
            ...
        }
    }
    """
    in_result = []
    cursor, db = create_connection()
    sql = "select distinct notebook_id from result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])
        sql = "SELECT pair.nid,pair.did from pair inner join notebook on pair.nid=notebook.id where notebook.add_run=1"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    dataset_dic = {}
    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id not in in_result:
            continue
        result = get_one_result(notebook_id)
        print("notebookid:" + str(notebook_id)+ ',result:',result)
        if dataset_id not in dataset_dic.keys():
            dataset_dic[dataset_id] = {}
        # all_score = dataset_dic[dataset_id][0]
        # all_count = dataset_dic[dataset_id][1]
        for i in result:
            if result[i] != -1:
                if i not in dataset_dic[dataset_id]:
                    dataset_dic[dataset_id][i] = ([],0)
                all_score = dataset_dic[dataset_id][i][0]
                all_count = dataset_dic[dataset_id][i][1]
                all_score.append(result[i])
                all_count += 1
                dataset_dic[dataset_id][i] = (all_score, all_count)


    # for i in dataset_dic:
    #     for j in dataset_dic[i]:
    #         all_score = dataset_dic[i][j][0]
    #         all_count = dataset_dic[i][j][1]
    #         if all_count == 0:
    #             dataset_dic[i][j] = -1
    #         else:
    #             dataset_dic[i][j] = all_score/all_count
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
#         result = get_one_result(notebook_id)
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
def param_walking(node):
    if type(node).__name__ == 'Str':
        return ('str',node.s)
    elif type(node).__name__ == 'Module':
        if len(node.body) != 0:
            return param_walking(node.body[0])
        else:
            return ('str','')

    elif type(node).__name__ == 'Num':
        return ('Num',node.n)
    elif type(node).__name__ == 'Name':
        return ('variable',node.id)
    elif type(node).__name__ == 'Call':
        if (type(node.func).__name__ == 'Name'):
            return ('func',node.func.id)
        elif (type(node.func).__name__ == 'Attribute'):
            return ('func',node.func.attr)
    elif type(node).__name__ == 'Attribute':
        return ('attr', astunparse.unparse(node))
    elif type(node).__name__ == 'Assign':
        return param_walking(node.value)
    elif type(node).__name__ == 'BinOp':
        return ('binop', astunparse.unparse(node))
    elif type(node).__name__ == 'BoolOp':
        return ('boolcomp', astunparse.unparse(node))
    elif type(node).__name__ == 'List':
        return ('list', astunparse.unparse(node))
    elif type(node).__name__ == 'NameConstant':
        return ('nameconst', node.value)
    elif type(node).__name__ == 'Subscript':
        return ('subdata', astunparse.unparse(node))
    elif type(node).__name__ == 'Dict':
        return ('dict', astunparse.unparse(node))
    elif type(node).__name__ == 'Tuple':
        return ('tuple', astunparse.unparse(node))
    elif type(node).__name__ == 'Set':
        return ('set', astunparse.unparse(node))
    elif type(node).__name__ == 'ListComp':
        return ('listcomp', astunparse.unparse(node))
    elif type(node).__name__ == 'Expr':
        return param_walking(node.value)
    elif type(node).__name__ == 'UnaryOp':
        return ('UnaryOp', astunparse.unparse(node))
    elif type(node).__name__ == 'AnnAssign':
        return ('annassign', astunparse.unparse(node))
    else:
        return (str(type(node).__name__).lower(), astunparse.unparse(node))

def parse_param(param_code):
    # print(param_code)
    try:
        r_node = ast.parse(param_code)
    except:
        return ('compile fail', param_code)
    # print(ast.dump(r_node))
    res = param_walking(r_node)
    return res

def get_operator_param_rate(regenerate=True):
    param_code_dic = {}
    param_type_rate_dic = {}
    param_code_rate_dic = {}
    if regenerate == True:
        # CONFIG.read('config.ini')
        operator_dic = eval(CONFIG.get('operators', 'operations'))
        print('select all operator')
        cursor, db = create_connection()
        sql = "SELECT operator,parameter_1_code,parameter_2_code,parameter_3_code,parameter_4_code,parameter_5_code,parameter_6_code,parameter_7_code FROM operator"
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        print('select all operator end')
        print('generate param_code_dic')
        for row in sql_res:
            operator = row[0]
            param_list = [row[1],row[2],row[3],row[4],row[5],row[6],row[7]]
            rm_1 = False
            if operator not in param_code_dic:
                param_code_dic[operator] = {}
            if operator_dic[operator]['call_type']== 2 or operator_dic[operator]['call_type']== 4 :
                rm_1 = True
            for index,param_code in enumerate(param_list):
                if rm_1 == True and index == 0:
                    continue
                if param_code == None:
                    param_type, param_content = ('default','Null')
                else:
                    param_type,param_content = parse_param(param_code)
                    # print(param_type,param_content)
                if index not in param_code_dic[operator]:
                    param_code_dic[operator][index] = []
                param_code_dic[operator][index].append((param_type, param_content))

        np.save('./param_code_dic.npy', param_code_dic)
        print('save param_code_dic end')
    else:
        param_code_dic = np.load('./param_code_dic.npy', allow_pickle=True).item()
    print('count param_code')
    for operator in param_code_dic:
        if operator not in param_code_rate_dic:
            param_code_rate_dic[operator] = {}
        if operator not in param_type_rate_dic:
            param_type_rate_dic[operator] = {}
        for index in param_code_dic[operator]:
            if index not in param_code_rate_dic[operator]:
                param_code_rate_dic[operator][index] = {}
            if index not in param_type_rate_dic[operator]:
                param_type_rate_dic[operator][index] = {}
            all = 0
            for tup in param_code_dic[operator][index]:
                # print(tup)
                if tup[0] not in param_type_rate_dic[operator][index]:
                    param_type_rate_dic[operator][index][tup[0]] = 0
                param_type_rate_dic[operator][index][tup[0]] += 1
                if tup[1] not in param_code_rate_dic[operator][index]:
                    param_code_rate_dic[operator][index][tup[1]] = 0
                param_code_rate_dic[operator][index][tup[1]] += 1
                all += 1
            for ptype in param_type_rate_dic[operator][index]:
                if all != 0:
                    param_type_rate_dic[operator][index][ptype] = (param_type_rate_dic[operator][index][ptype]/all, param_type_rate_dic[operator][index][ptype])
                else:
                    param_type_rate_dic[operator][index][ptype] =(0,0)
            for pcode in param_code_rate_dic[operator][index]:
                if all != 0:
                    param_code_rate_dic[operator][index][pcode] = (param_code_rate_dic[operator][index][pcode] / all, param_code_rate_dic[operator][index][pcode])
                else:
                    param_code_rate_dic[operator][index][pcode] = (0, 0)


    np.save('./param_type_rate_dic.npy', param_type_rate_dic)
    np.save('./param_code_rate_dic.npy', param_code_rate_dic)
    print('save param_code end')


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
            result = get_one_result(row[0])
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
                        parameter_dic[row[1]][row[num]][i] = ([], 0)
                    all_score = parameter_dic[row[1]][row[num]][i][0]
                    all_count = parameter_dic[row[1]][row[num]][i][1]
                    all_score.append(result[i])
                    all_count += 1
                    parameter_dic[row[1]][row[num]][i] = (all_score, all_count)
    # np.save('./param_score_dic.npy', ope_dic)
    for i in parameter_dic: # operator
        for j in parameter_dic[i]: # parameter
            for k in parameter_dic[i][j]: # score type
                all_score = parameter_dic[i][j][k][0]
                all_count = parameter_dic[i][j][k][1]
                if all_count == 0:
                    parameter_dic[i][j][k] = (-1,-1,0)
                else:
                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum/all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc-mean)**2
                    sq /= all_count
                    parameter_dic[i][j][k] = (mean, sq, all_count)


    np.save('./param_score_dic.npy', ope_dic)

def get_operator_exist_dic():
    cursor, db = create_connection()
    CONFIG.read('config.ini')
    operator_dic = eval(CONFIG.get('operators', 'operations'))

    in_result = []
    sql = "SELECT distinct notebook_id FROM result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(int(row[0]))

    operator_notebook_dic = {}
    for operator in operator_dic.keys():
        operator_notebook_dic[operator] = []
        sql = "SELECT distinct notebook_id FROM operator where operator = '" + operator + "'"
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            operator_notebook_dic[operator].append(int(row[0]))

    ope_dic = {}
    nope_dic = {}
    for notebook_id in in_result:
        print('notebook_idL',notebook_id)
        for operator in operator_dic.keys():
            if notebook_id in operator_notebook_dic[operator]:
                result = get_one_result(notebook_id)
                if operator not in ope_dic.keys():
                    ope_dic[operator] = {}
                for i in result:
                    if result[i] != -1:
                        if i not in ope_dic[operator]:
                            ope_dic[operator][i] = ([], 0)
                        all_score = ope_dic[operator][i][0]
                        all_count = ope_dic[operator][i][1]
                        all_score.append(result[i])
                        all_count += 1
                        ope_dic[operator][i] = (all_score, all_count)
            else:
                result = get_one_result(notebook_id)
                if operator not in nope_dic.keys():
                    nope_dic[operator] = {}
                for i in result:
                    if result[i] != -1:
                        if i not in nope_dic[operator]:
                            nope_dic[operator][i] = ([], 0)
                        all_score = nope_dic[operator][i][0]
                        all_count = nope_dic[operator][i][1]
                        all_score.append(result[i])
                        all_count += 1
                        nope_dic[operator][i] = (all_score, all_count)

    for i in nope_dic:
        for j in nope_dic[i]:
            all_score = 0
            all_count = 0
            if i in ope_dic:
                if j in ope_dic[i]:
                    all_score = ope_dic[i][j][0]
                    all_count = ope_dic[i][j][1]
            n_all_score = nope_dic[i][j][0]
            n_all_count = nope_dic[i][j][1]

            if all_count == 0:
                ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in all_score:
                    sum += sc
                mean = sum / all_count
                sq = 0
                for sc in all_score:
                    sq += (sc - mean) ** 2
                sq /= all_count
                ope_temp = (mean, sq, all_count)

            if n_all_count == 0:
                n_ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in n_all_score:
                    sum += sc
                mean = sum / n_all_count
                sq = 0
                for sc in n_all_score:
                    sq += (sc - mean) ** 2
                sq /= n_all_count
                n_ope_temp = (mean, sq, n_all_count)

            if i not in ope_dic:
                ope_dic[i] = {}
            ope_dic[i][j] =(ope_temp,n_ope_temp)
    np.save('./ope_score_dic.npy', ope_dic)


# def get_operator_exist_dic_by_one_dataset(dataset_id, notebook_list):
#     cursor, db = create_connection()
#     CONFIG.read('config.ini')
#     operator_dic = eval(CONFIG.get('operators', 'operations'))
#     ope_dic = {}
#     notebook_list_of_dataset = []
#     operator_notebook_dic = {}
#     noperator_notebook_dic = {}
#
#     if notebook_list !=  []:
#         notebook_list_of_dataset = notebook_list
#     else:
#         sql = "SELECT pair.nid FROM pair where pair.did=" + str(dataset_id)
#         cursor.execute(sql)
#         sql_res = cursor.fetchall()
#         for row in sql_res:
#             notebook_list_of_dataset.append(int(row[0]))
#
#     for operator in operator_dic.keys():
#         sql = "SELECT distinct notebook_id FROM operator where operator = '" + operator + "'"
#         cursor.execute(sql)
#         sql_res = cursor.fetchall()
#         operator_notebook_dic[operator] = []
#         if operator not in ope_dic.keys():
#             ope_dic[operator] = {}
#         for row in sql_res:
#             notebook_id = int(row[0])
#             if notebook_id not in notebook_list_of_dataset:
#                 continue
#             operator_notebook_dic[operator].append(notebook_id)
#             result = get_one_result(notebook_id)
#             for i in result:
#                 if result[i] != -1:
#                     if i not in ope_dic[operator]:
#                         ope_dic[operator][i] = (0, 0)
#                     all_score = ope_dic[operator][i][0]
#                     all_count = ope_dic[operator][i][1]
#                     all_score += result[i]
#                     all_count += 1
#                     ope_dic[operator][i] = (all_score, all_count)
#                     print('add_one_result:',ope_dic[operator][i])
#
#     for operator in operator_notebook_dic:
#         noperator_notebook_dic[operator] = []
#         for notebook in notebook_list_of_dataset:
#             if notebook not in operator_notebook_dic[operator]:
#                 noperator_notebook_dic[operator].append(notebook)
#
#     nope_dic = {}
#     for operator in operator_dic.keys():
#         for notebook_id in noperator_notebook_dic[operator]:
#             result = get_one_result(notebook_id)
#             if operator not in nope_dic.keys():
#                 nope_dic[operator] = {}
#             for i in result:
#                 if result[i] != -1:
#                     if i not in nope_dic[operator]:
#                         nope_dic[operator][i] = (0, 0)
#                     all_score = nope_dic[operator][i][0]
#                     all_count = nope_dic[operator][i][1]
#                     all_score += result[i]
#                     all_count += 1
#                     nope_dic[operator][i] = (all_score, all_count)
#
#     for i in nope_dic:
#         for j in nope_dic[i]:
#             all_score = 0
#             all_count = 0
#             if i in ope_dic:
#                 if j in ope_dic[i]:
#                     all_score = ope_dic[i][j][0]
#                     all_count = ope_dic[i][j][1]
#             n_all_score = nope_dic[i][j][0]
#             n_all_count = nope_dic[i][j][1]
#
#             if all_count == 0:
#                 ope_temp = (-1, 0)
#             else:
#                 ope_temp = (all_score / all_count, all_count)
#             if n_all_count == 0:
#                 n_ope_temp = (-1, 0)
#             else:
#                 n_ope_temp = (n_all_score/n_all_count, n_all_count)
#             if i not in ope_dic:
#                 ope_dic[i] = {}
#             ope_dic[i][j] =(ope_temp,n_ope_temp)
#
#     return ope_dic

def get_dataset_operator_exist_dic():
    print("get_pair_dic")
    cursor, db = create_connection()
    in_result = []
    sql = 'select distinct(notebook_id) from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    pair_dic = {}
    sql = 'select * from pair'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id not in in_result:
            continue
        if dataset_id not in pair_dic.keys():
            pair_dic[dataset_id] = []
        pair_dic[int(dataset_id)].append(int(notebook_id))

    np.save('./pair_dic.npy',pair_dic)
    print("get_pair_dic end")
    # if number != '-1':
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id) limit " + str(number)
    # else:
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id)"
    # cursor.execute(sql)
    # sql_res = cursor.fetchall()
    result = {}
    count = 0
    for dataset_id in pair_dic:
        print(count)
        count += 1
        result[dataset_id] = get_operator_exist_dic_by_one_dataset(dataset_id, pair_dic[dataset_id])
    np.save('./1d0mgap3_.npy',result)
    return result

def get_notebook_operator_dic():
    all_seq_notebook_list = []
    cursor, db = create_connection()
    sql = "select distinct notebook_id from operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    notebook_operator_dic = {}
    for row in sql_res:
        all_seq_notebook_list.append(row[0])
        notebook_operator_dic[row[0]] = []
        sql = "select rank,logic_operation from operator where notebook_id="+str(row[0])
        cursor.execute(sql)
        sql_res1 = cursor.fetchall()
        for row1 in sql_res1:
            notebook_operator_dic[row[0]].append(row1[0])
    np.save('./notebook_operator_dic.npy',notebook_operator_dic)
    return notebook_operator_dic

def get_sequence2id_dic(run_award = False):
    if run_award == True or not os.path.exists('./notebook_operator_dic.npy'):
        notebook_operator_dic = get_notebook_operator_dic()
    else:
        notebook_operator_dic = np.load('./notebook_operator_dic.npy', allow_pickle=True).item()
    sequence2id = {}
    sequence2notebook = {}
    seq_id = 1
    for notebook in notebook_operator_dic:
        sequence = notebook_operator_dic[notebook]
        in_dic = False
        for j in sequence2id:
            if sequence2id[j] == sequence:
                in_dic = True
                if j not in sequence2notebook:
                    sequence2notebook[j] = []
                sequence2notebook[j].append(notebook)
        if in_dic == False:
            sequence2id[seq_id] = sequence
            seq_id += 1
            if seq_id not in sequence2notebook:
                sequence2notebook[seq_id] = []
            sequence2notebook[seq_id].append(notebook)
    np.save('./sequence2id.npy', sequence2id)
    np.save('./sequence2notebook.npy', sequence2notebook)
    return sequence2notebook

def get_result_of_seq(run_award = False):
    if run_award == True or not os.path.exists('./sequence2notebook.npy'):
        sequence2notebook = get_sequence2id_dic()
    else:
        sequence2notebook = np.load('./sequence2notebook.npy', allow_pickle=True).item()

    result = {}
    for seq_id in sequence2notebook:
        notebook_list = sequence2notebook[seq_id]
        result[seq_id] = {}
        for notebook in notebook_list:
            score = get_one_result(notebook)
            for i in score:
                if score[i] != -1:
                    if i not in result[seq_id]:
                        result[seq_id][i] = (0,0)
                    temp1 = result[seq_id][i][0] + score[i]
                    temp2 = result[seq_id][i][1] + 1
                    result[seq_id][i] = (temp1,temp2)

    for seq_id in result:
        for i in result[seq_id]:
            if result[seq_id][i][1] != 0:
                temp1 = result[seq_id][i][0]/result[seq_id][i][1]
                result[seq_id][i] = (temp1, result[seq_id][i][1])

    np.save('./seq_score_dic.npy',result)
def show_dic(showtype):
    showtype = int(showtype)
    if showtype==4:
        ope_score_dic = np.load('./ope_score_dic.npy',allow_pickle=True).item()
        new_dic = {}
        has_add=set()
        for i in ope_score_dic:
            if i in has_add:
                continue
            has_add.add(i)
            if (ope_score_dic[i]['accuracy_score'][0][1] + ope_score_dic[i]['auc'][0][1]) == 0:
                temp1 = (-1,0)
            else:
                temp1 = ((ope_score_dic[i]['accuracy_score'][0][0]*ope_score_dic[i]['accuracy_score'][0][1] + ope_score_dic[i]['auc'][0][0]*ope_score_dic[i]['auc'][0][1])/(ope_score_dic[i]['accuracy_score'][0][1] + ope_score_dic[i]['auc'][0][1]),ope_score_dic[i]['accuracy_score'][0][1] + ope_score_dic[i]['auc'][0][1])
            if (ope_score_dic[i]['accuracy_score'][1][1] + ope_score_dic[i]['auc'][1][1]) == 0:
                temp2 = (-1, 0)
            else:
                temp2 = ((ope_score_dic[i]['accuracy_score'][1][0]*ope_score_dic[i]['accuracy_score'][1][1] + ope_score_dic[i]['auc'][1][0]*ope_score_dic[i]['auc'][1][1])/(ope_score_dic[i]['accuracy_score'][1][1] + ope_score_dic[i]['auc'][1][1]),ope_score_dic[i]['accuracy_score'][1][1] + ope_score_dic[i]['auc'][1][1])
            new_dic[i] = {}
            new_dic[i]['accuracy_score'] = (temp1,temp2)
            new_dic[i]['f1_score'] = ope_score_dic[i]['f1_score']
            for ope in ope_score_dic[i]:
                try:
                    if ope != 'accuracy_score' and ope != 'auc' and ope_score_dic[i][ope][0][1] + ope_score_dic[i][ope][1][1] > 10 and ope_score_dic[i][ope][0][0] != -1:
                        new_dic[i][ope] = ope_score_dic[i][ope]
                except:
                    continue
            print(i)
            for score in new_dic[i]:
                if new_dic[i][score][0][0] == -1 or new_dic[i][score][1][0] == -1:
                    continue
                if new_dic[i][score][0][1] < 5 :
                    continue
                print('\t' + score)
                if new_dic[i][score][0][0] > new_dic[i][score][1][0]:
                    if score == 'mean_squared_error' or score == 'mean_absolute_error':
                        print('\t' + "\033[0;31;40m" + str(new_dic[i][score]) + "\033[0m")
                    else:
                        print('\t' + "\033[0;32;40m" + str(new_dic[i][score]) + "\033[0m")
                else:
                    if score == 'mean_squared_error' or score == 'mean_absolute_error':
                        print('\t' + "\033[0;32;40m" + str(new_dic[i][score]) + "\033[0m")
                    else:
                        print('\t' + "\033[0;31;40m" + str(new_dic[i][score]) + "\033[0m")
            print('********************')
    elif showtype==2:
        dic = np.load('./dataset_score_dic.npy',allow_pickle=True).item()
        for i in dic:
            if len(dic[i]) == 0:
                continue
            print(i)
            print(dic[i])
            print('********************')
    elif showtype == 1:
        tup = np.load('./all_exit_tuple.npy',allow_pickle=True)
        for score in tup[0].keys():
            if score in tup[1].keys():
                print(score)
                if tup[0][score][0] > tup[1][score][0]:
                    if score == 'mean_squared_error' or score == 'mean_absolute_error':
                        print('\t' + "\033[0;31;40m" + str((tup[0][score],tup[1][score])) + "\033[0m")
                    else:
                        print('\t' + "\033[0;32;40m" + str((tup[0][score],tup[1][score])) + "\033[0m")
                else:
                    if score == 'mean_squared_error' or score == 'mean_absolute_error':
                        print('\t' + "\033[0;32;40m" + str((tup[0][score],tup[1][score]))+ "\033[0m")
                    else:
                        print('\t' + "\033[0;31;40m" + str((tup[0][score],tup[1][score])) + "\033[0m")

        print(tup)
    elif showtype==3:
        dic = np.load('./exist_operator_groupby_dataset.npy',allow_pickle=True).item()
        need_print =[]
        false_count = 0
        true_count = 0
        false_count_1 = 0
        true_count_1 = 0

        dataset_num = {}
        print('dataset_num:', len(dic.keys()))
        for i in dic:
            dataset_num[i] = [0, 0]
            for score in dic[i]:
                dataset_num[i][0] += dic[i][score][0][1]
                dataset_num[i][1] += dic[i][score][1][1]
                if dic[i][score][0][1] == 0 or dic[i][score][1][1] == 0:
                    continue
                need_print.append(i)

        printed = []
        for i in dic:
            if i not in need_print:
                continue
            if i in printed:
                continue
            printed.append(i)
            print(i)
            for score in dic[i]:
                if dic[i][score][0][1] == 0 or dic[i][score][1][1] == 0:
                    continue
                print('\t' + score)
                if dic[i][score][0][0] > dic[i][score][1][0]:
                    if score == 'mean_squared_error' or score == 'mean_absolute_error':
                        print('\t' + "\033[0;31;40m" + str(dic[i][score]) + "\033[0m")
                        false_count += 1
                        false_count_1 += dic[i][score][0][1]
                        false_count_1 += dic[i][score][1][1]
                    else:
                        print('\t' + "\033[0;32;40m" + str(dic[i][score]) + "\033[0m")
                        true_count += 1
                        true_count_1 += dic[i][score][0][1]
                        true_count_1 += dic[i][score][1][1]
                else:
                    if score == 'mean_squared_error' or score == 'mean_absolute_error':
                        print('\t' + "\033[0;32;40m" + str(dic[i][score]) + "\033[0m")
                        true_count += 1
                        true_count_1 += dic[i][score][0][1]
                        true_count_1 += dic[i][score][1][1]
                    else:
                        print('\t' + "\033[0;31;40m" + str(dic[i][score]) + "\033[0m")
                        false_count += 1
                        false_count_1 += dic[i][score][0][1]
                        false_count_1 += dic[i][score][1][1]
            print('********************')
            # if dic[i][1] == -1:
            #     continue
        print("false_count:",false_count)
        print("true_count:", true_count)
        print("false_count_1:", false_count_1)
        print("true_count_1:", true_count_1)

        false_num = 0
        true_num = 0
        for i in dataset_num:
            if dataset_num[i][0] > dataset_num[i][1]:
                false_num += 1
            else:
                true_num  += 1

        print('')
        print("false_num:",false_num)
        print("true_num:", true_num)
    elif showtype==6:
        dic = np.load('./dataset_operation_dic.npy', allow_pickle=True).item()
        # print(dic)
        need_print_dataset = {}
        false_count = 0
        false_count_1 = 0
        true_count = 0
        true_count_1 = 0

        false_num = 0
        true_num = 0

        true_all = 0
        false_all = 0
        print(dic)
        # for i in dic:
        #     if 'accuracy_score' in dic[i] and 'auc' in dic[i]:
        #         if (dic[i]['accuracy_score'][0][1] + dic[i]['auc'][0][1]) == 0:
        #             temp1 = (-1,0)
        #         else:
        #             temp1 = ((dic[i]['accuracy_score'][0][0]*dic[i]['accuracy_score'][0][1] + dic[i]['auc'][0][0]*dic[i]['auc'][0][1])/(dic[i]['accuracy_score'][0][1] + dic[i]['auc'][0][1]),dic[i]['accuracy_score'][0][1] + dic[i]['auc'][0][1])
        #         if (dic[i]['accuracy_score'][1][1] + dic[i]['auc'][1][1]) == 0:
        #             temp2 = (-1, 0)
        #         else:
        #             temp2 = ((dic[i]['accuracy_score'][1][0]*dic[i]['accuracy_score'][1][1] + dic[i]['auc'][1][0]*dic[i]['auc'][1][1])/(dic[i]['accuracy_score'][1][1] + dic[i]['auc'][1][1]),dic[i]['accuracy_score'][1][1] + dic[i]['auc'][1][1])
        #         dic[i]['accuracy_score'] = (temp1,temp2)
        #     elif 'auc' in dic[i]:
        #         dic[i]['accuracy_score'] = dic[i]['auc']
        #
        #     # print("\033[0;34;40m" + str(i) + "\033[0m")
        #
        #     for operator in dic[i]:
        #         # print("\033[0;35;40m" + operator + "\033[0m")
        #         for score in dic[i][operator]:
        #             # print("\033[0;36;40m" + score + "\033[0m")
        #             try:
        #                 if dic[i][operator][score][0][1] < dic[i][operator][score][1][1]:
        #                     true_num += 1
        #                 else:
        #                     false_num += 1
        #
        #                 if dic[i][operator][score][0][0] == -1:
        #                     continue
        #                 if dic[i][operator][score][0][1] + dic[i][operator][score][1][1] < 10:
        #                     continue
        #                 if i not in need_print_dataset:
        #                     need_print_dataset[i] = {}
        #                 if operator not in need_print_dataset[i]:
        #                     need_print_dataset[i][operator] = []
        #                 need_print_dataset[i][operator].append(score)
        #             except:
        #                 continue
        #
        # for i in dic:
        #     if i not in need_print_dataset:
        #         continue
        #     print(str(i))
        #     for operator in dic[i]:
        #         if operator not in need_print_dataset[i]:
        #             continue
        #         print('\t'+operator)
        #         for score in dic[i][operator]:
        #             if score not in need_print_dataset[i][operator]:
        #                 continue
        #             print("\t\t" + score)
        #             try:
        #                 if dic[i][operator][score][0][0] == -1:
        #                     continue
        #                 if dic[i][operator][score][0][1] + dic[i][operator][score][1][1] < 10:
        #                     continue
        #                 elif dic[i][operator][score][0][0] > dic[i][operator][score][1][0]:
        #                     print("\033[0;32;40m\t\t" + str(dic[i][operator][score]) + "\033[0m")
        #                     true_count += 1
        #                     true_count_1 += dic[i][operator][score][0][1]
        #                     true_count_1 += dic[i][operator][score][1][1]
        #                     if dic[i][operator][score][0][1] < dic[i][operator][score][1][1]:
        #                         true_all += 1
        #                     else:
        #                         false_all += 1
        #                 elif dic[i][operator][score][0][0] < dic[i][operator][score][1][0]:
        #                     print("\033[0;31;40m\t\t" + str(dic[i][operator][score]) + "\033[0m")
        #                     false_count += 1
        #                     false_count_1 += dic[i][operator][score][0][1]
        #                     false_count_1 += dic[i][operator][score][1][1]
        #                     false_all += 1
        #             except:
        #                 continue
        #     print('********************')
        #     print('true_count:', true_count)
        #     print('true_count_1:', true_count_1)
        #     print('true_num:', true_num)
        #     print('true_all:', true_all)
        #     print('false_count:',false_count)
        #     print('false_count_1:', false_count_1)
        #     print('false_num:', false_num)
        #     print('false_all:', false_all)
    elif showtype == 8:
        print('input page num')
        page_num = int(input())
        print('input page size')
        page_size = int(input())
        show_notebook_seq = np.load('./show_notebook_seq.npy', allow_pickle=True).item()

        start = (page_num-1)*page_size
        for index,i in enumerate(show_notebook_seq):
            if index < start:
                continue
            if index > start + page_size:
                break
            print(str(i))
            for item in show_notebook_seq[i]:
                # print(item)
                if item[2] == 'Show' and item[0] != 'show':
                    if item[3] == None:
                        item[3] = ''
                    print('\t\033[0;35;40m' + str(item[4]) + ':' + item[0] + '  ' + item[1] + '  '+ item[3] + "\033[0m")
                elif item[2] == 'Show_config'or item[0] == 'show':
                    if item[3] == None:
                        item[3] = ''
                    print('\t\033[0;37;40m'  + str(item[4]) + ':'+ item[0] + '  ' + item[1] + '  '+ item[3] + ' '"\033[0m")
                else:
                    if item[3] == None:
                        item[3] = ''
                    print('\t\033[0;36;40m'  + str(item[4]) + ':'+ item[0] + '  ' + item[1] + '  '+ item[3] + ' '"\033[0m")
            # break
    elif showtype == 9:
        print('code or type or all')
        ct = input()
        if ct == 'code':
            rate_dic = np.load('./param_code_rate_dic.npy',allow_pickle=True).item()
        elif ct == 'type':
            rate_dic = np.load('./param_type_rate_dic.npy', allow_pickle=True).item()
        elif ct == 'all':
            rate_dic = np.load('./param_code_dic.npy', allow_pickle=True).item()
        else:
            rate_dic = {}
            return
        operator_dic = eval(CONFIG.get('operators', 'operations'))
        if ct == 'all':
            for operator in rate_dic:
                print(operator)
                for index in rate_dic[operator]:
                    if index >= len(operator_dic[operator]['params']):
                        continue
                    print('\t'+ str(operator_dic[operator]['params'][index]))
                    for content in rate_dic[operator][index]:
                        print('\t\t' + str(content))
        else:
            need_pass_print = {}
            for operator in rate_dic:
                for index in rate_dic[operator]:
                    if index >= len(operator_dic[operator]['params']):
                        continue
                    for content in rate_dic[operator][index]:
                        if content == 'default' and rate_dic[operator][index][content][0] > 0.95:
                            if operator not in need_pass_print:
                                need_pass_print[operator] = []
                            need_pass_print[operator].append(index)


            for operator in rate_dic:
                print(operator)
                for index in rate_dic[operator]:
                    if index >= len(operator_dic[operator]['params']):
                        continue
                    if operator in need_pass_print:
                        if index in need_pass_print[operator]:
                            continue
                    print('\t'+ str(operator_dic[operator]['params'][index]))
                    for content in rate_dic[operator][index]:
                        print('\t\t' + str(content) + ':' + str(rate_dic[operator][index][content]))




def get_show_sequence():
    notebook_list = []
    cursor, db = create_connection()
    sql = "select distinct notebook_id from show_operator where logic_operation='Show'"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        notebook_list.append(int(row[0]))

    sql = "select notebook_id, operator,data_object_value,logic_operation,parameter_1_code,rank from show_operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    show_notebook_seq = {}
    for row in sql_res:
        notebook_id = int(row[0])
        if notebook_id not in notebook_list:
            continue
        operator = row[1]
        data_object_value = row[2]
        logic_operation = row[3]
        parameter_1_code = row[4]
        rank = row[5]
        if notebook_id not in show_notebook_seq:
            show_notebook_seq[notebook_id] = []
        show_notebook_seq[notebook_id].append([operator,data_object_value,logic_operation,parameter_1_code,rank])

    np.save('./show_notebook_seq.npy',show_notebook_seq)

    # for i in show_notebook_seq:
    #     print("\033[0;32;40m" + str(i) + "\033[0m")
    #     for item in show_notebook_seq[i]:
    #         print('\t' + str(item))
    #

def get_model_exist_dic():
    cursor, db = create_connection()

    in_result = []
    sql = "SELECT distinct notebook_id FROM result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    in_ope = []
    sql = "SELECT distinct notebook_id FROM operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_ope.append(row[0])

    model_dic = {}
    nmodel_dic = {}

    has_f1_score_list = {}
    has_score_list_ = {}
    has_cross_val_score_list = {}
    has_accuracy_score_list = {}
    has_roc_auc_score_list = {}
    has_precision_score_list = {}
    has_recall_score_list = {}
    has_best_score_list = {}
    has_auc_list = {}

    nhas_f1_score_list = {}
    nhas_score_list_ = {}
    nhas_cross_val_score_list = {}
    nhas_accuracy_score_list = {}
    nhas_roc_auc_score_list = {}
    nhas_precision_score_list = {}
    nhas_recall_score_list = {}
    nhas_best_score_list = {}
    nhas_auc_list = {}

    model_key_dic = eval(CONFIG.get('models', 'model_dic'))
    for notebook_id in in_result:
        if notebook_id in in_ope:
            one_res = get_one_model_result(notebook_id)
            # print('one_res',one_res)
            for model in one_res:
                # print('model:',model)
                if model not in model_key_dic:
                    continue
                if model not in model_dic:
                    model_dic[model] = {}
                    for score in one_res[model]:
                        if score == 'f1_score':
                            if model not in has_f1_score_list:
                                has_f1_score_list[model] = []
                            has_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in has_score_list_:
                                has_score_list_[model] = []
                            has_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in has_cross_val_score_list:
                                has_cross_val_score_list[model] = []
                            has_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in has_accuracy_score_list:
                                has_accuracy_score_list[model] = []
                            has_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in has_roc_auc_score_list:
                                has_roc_auc_score_list[model] = []
                            has_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'precision_score':
                            if model not in has_precision_score_list:
                                has_precision_score_list[model] = []
                            has_precision_score_list[model].append(one_res[model][score])
                        if score == 'recall_score':
                            if model not in has_recall_score_list:
                                has_recall_score_list[model] = []
                            has_recall_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in has_best_score_list:
                                has_best_score_list[model] = []
                            has_best_score_list[model].append(one_res[model][score])
                        if score == 'auc':
                            if model not in has_auc_list:
                                has_auc_list[model] = []
                            has_auc_list[model].append(one_res[model][score])

                        model_dic[model][score] = [[one_res[model][score]],1]
                        # print('model_dic item:',model_dic[model][score])
                        print('model,score:', (model, score))
                        print(model_dic[model][score])
                else:
                    for score in one_res[model]:
                        if score == 'f1_score':
                            if model not in has_f1_score_list:
                                has_f1_score_list[model] = []
                            has_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in has_score_list_:
                                has_score_list_[model] = []
                            has_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in has_cross_val_score_list:
                                has_cross_val_score_list[model] = []
                            has_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in has_accuracy_score_list:
                                has_accuracy_score_list[model] = []
                            has_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in has_roc_auc_score_list:
                                has_roc_auc_score_list[model] = []
                            has_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'precision_score':
                            if model not in has_precision_score_list:
                                has_precision_score_list[model] = []
                            has_precision_score_list[model].append(one_res[model][score])
                        if score == 'recall_score':
                            if model not in has_recall_score_list:
                                has_recall_score_list[model] = []
                            has_recall_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in has_best_score_list:
                                has_best_score_list[model] = []
                            has_best_score_list[model].append(one_res[model][score])
                        if score == 'auc':
                            if model not in has_auc_list:
                                has_auc_list[model] = []
                            has_auc_list[model].append(one_res[model][score])

                        if score not in model_dic[model]:
                            model_dic[model][score] = [[one_res[model][score]], 1]
                        else:
                            temp_li = model_dic[model][score][0]
                            temp_li.append(one_res[model][score])
                            model_dic[model][score] = [ temp_li , model_dic[model][score][1]+1 ]
                        print('model,score:', (model,score))
                        print(model_dic[model][score])
        else:
            one_res = get_one_model_result(notebook_id)
            for model in one_res:
                if model not in nmodel_dic:
                    if model not in model_key_dic:
                        continue
                    nmodel_dic[model] = {}
                    for score in one_res[model]:
                        if score == 'f1_score':
                            if model not in nhas_f1_score_list:
                                nhas_f1_score_list[model] = []
                            nhas_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in nhas_score_list_:
                                nhas_score_list_[model] = []
                            nhas_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in nhas_cross_val_score_list:
                                nhas_cross_val_score_list[model] = []
                            nhas_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in nhas_accuracy_score_list:
                                nhas_accuracy_score_list[model] = []
                            nhas_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in nhas_roc_auc_score_list:
                                nhas_roc_auc_score_list[model] = []
                            nhas_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'precision_score':
                            if model not in nhas_precision_score_list:
                                nhas_precision_score_list[model] = []
                            nhas_precision_score_list[model].append(one_res[model][score])
                        if score == 'recall_score':
                            if model not in nhas_recall_score_list:
                                nhas_recall_score_list[model] = []
                            nhas_recall_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in nhas_best_score_list:
                                nhas_best_score_list[model] = []
                            nhas_best_score_list[model].append(one_res[model][score])
                        if score == 'auc':
                            if model not in nhas_auc_list:
                                nhas_auc_list[model] = []
                            nhas_auc_list[model].append(one_res[model][score])

                        nmodel_dic[model][score] = [[one_res[model][score]],1]
                        print('model,score:', (model, score))
                        print(nmodel_dic[model][score])
                else:
                    for score in one_res[model]:
                        if score == 'f1_score':
                            if model not in nhas_f1_score_list:
                                nhas_f1_score_list[model] = []
                            nhas_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in nhas_score_list_:
                                nhas_score_list_[model] = []
                            nhas_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in nhas_cross_val_score_list:
                                nhas_cross_val_score_list[model] = []
                            nhas_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in nhas_accuracy_score_list:
                                nhas_accuracy_score_list[model] = []
                            nhas_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in nhas_roc_auc_score_list:
                                nhas_roc_auc_score_list[model] = []
                            nhas_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'precision_score':
                            if model not in nhas_precision_score_list:
                                nhas_precision_score_list[model] = []
                            nhas_precision_score_list[model].append(one_res[model][score])
                        if score == 'recall_score':
                            if model not in nhas_recall_score_list:
                                nhas_recall_score_list[model] = []
                            nhas_recall_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in nhas_best_score_list:
                                nhas_best_score_list[model] = []
                            nhas_best_score_list[model].append(one_res[model][score])
                        if score == 'auc':
                            if model not in nhas_auc_list:
                                nhas_auc_list[model] = []
                            nhas_auc_list[model].append(one_res[model][score])

                        if score not in nmodel_dic[model]:
                            nmodel_dic[model][score] = [[one_res[model][score]], 1]
                        else:
                            temp_li = nmodel_dic[model][score][0]
                            temp_li.append(one_res[model][score])
                            nmodel_dic[model][score] = [ temp_li , nmodel_dic[model][score][1]+1 ]
                        print('model,score:', (model, score))
                        print(nmodel_dic[model][score])

    for model in nmodel_dic:
        for score in nmodel_dic[model]:
            all_score = 0
            all_count = 0
            if model in model_dic:
                if score in model_dic[model]:
                    all_score = model_dic[model][score][0]
                    all_count = model_dic[model][score][1]
            n_all_score = nmodel_dic[model][score][0]
            n_all_count = nmodel_dic[model][score][1]

            if all_count == 0:
                ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in all_score:
                    sum += sc
                mean = sum/all_count
                sq = 0
                for sc in all_score:
                    sq += (sc-mean)**2
                sq /= all_count
                ope_temp = (mean, sq, all_count)
            if n_all_count == 0:
                n_ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in n_all_score:
                    sum += sc
                mean = sum / n_all_count
                sq = 0
                for sc in n_all_score:
                    sq += (sc - mean)**2
                sq /= n_all_count
                n_ope_temp = (mean, sq, n_all_count)

            if model not in model_dic:
                model_dic[model] = {}
            model_dic[model][score] = (ope_temp, n_ope_temp)
    for model in model_dic:
        for score in model_dic[model]:
            if type(model_dic[model][score]).__name__ != 'tuple':
                all_score = model_dic[model][score][0]
                all_count = model_dic[model][score][1]

                sum = 0
                for sc in all_score:
                    sum += sc
                mean = sum/all_count
                sq = 0
                for sc in all_score:
                    sq += (sc-mean)**2
                sq /= all_count
                ope_temp = (mean, sq, all_count)

                n_ope_temp = (-1, -1, 0)
                model_dic[model][score] = (ope_temp, n_ope_temp)

    np.save('model_has_f1_score_list', has_f1_score_list)
    np.save('model_nhas_f1_score_list', nhas_f1_score_list)
    np.save('model_has_score_list_', has_score_list_)
    np.save('model_nhas_score_list_', nhas_score_list_)
    np.save('model_has_cross_val_score_list', has_cross_val_score_list)
    np.save('model_nhas_cross_val_score_list', nhas_cross_val_score_list)
    np.save('model_has_accuracy_score_list', has_accuracy_score_list)
    np.save('model_nhas_accuracy_score_list', nhas_accuracy_score_list)
    np.save('model_has_roc_auc_score_list', has_roc_auc_score_list)
    np.save('model_nhas_roc_auc_score_list', nhas_roc_auc_score_list)
    np.save('model_has_precision_score_list', has_precision_score_list)
    np.save('model_nhas_precision_score_list', nhas_precision_score_list)
    np.save('model_has_recall_score_list', has_recall_score_list)
    np.save('model_nhas_recall_score_list', nhas_recall_score_list)
    np.save('model_has_best_score_list', has_best_score_list)
    np.save('model_nhas_best_score_list', nhas_best_score_list)
    np.save('model_has_auc_list', has_auc_list)
    np.save('model_nhas_auc_list', nhas_auc_list)

    np.save('./model_score_dic.npy', model_dic)
def get_model_exist_dic_len():
    cursor, db = create_connection()

    in_result = {}
    sql = "SELECT distinct notebook_id FROM result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()

    for row in sql_res:
        in_result[row[0]]=[]
        sql = "SELECT distinct model_type from result where notebook_id = " + str(row[0])
        cursor.execute(sql)
        sql_res1 = cursor.fetchall()
        for row1 in sql_res1:
            in_result[row[0]].append(row1[0])


    in_ope = []
    sql = "SELECT distinct notebook_id FROM operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_ope.append(row[0])

    model_dic = {}
    nmodel_dic = {}
    for notebook_id in in_result:
        if notebook_id in in_ope:
            one_res = get_one_seq_len(notebook_id)
            for model in in_result[notebook_id]:
                if model not in model_dic:
                    model_dic[model] = [[one_res],1]
                else:
                    temp_li = model_dic[model][0]
                    temp_li.append(one_res)
                    model_dic[model] = [ temp_li , model_dic[model][1]+1 ]
        else:
            one_res = get_one_seq_len(notebook_id)
            for model in in_result[notebook_id]:
                if model not in nmodel_dic:
                    model_dic[model] = [[one_res],1]
                else:
                    temp_li = nmodel_dic[model][0]
                    temp_li.append(one_res)
                    nmodel_dic[model] = [ temp_li , nmodel_dic[model][1]+1 ]

    for model in nmodel_dic:
        for score in nmodel_dic[model]:
            all_score = 0
            all_count = 0
            if model in model_dic:
                if score in model_dic[score]:
                    all_score = model_dic[model][score][0]
                    all_count = model_dic[model][score][1]
            n_all_score = nmodel_dic[model][score][0]
            n_all_count = nmodel_dic[model][score][1]

            if all_count == 0:
                ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in all_score:
                    sum += sc
                mean = sum/all_count
                sq = 0
                for sc in all_score:
                    sq += (sc-mean)**2
                sq /= all_count
                ope_temp = (mean, sq, all_count)
            if n_all_count == 0:
                n_ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in n_all_score:
                    sum += sc
                mean = sum / n_all_count
                sq = 0
                for sc in n_all_score:
                    sq += (sc - mean)**2
                sq /= n_all_count
                n_ope_temp = (mean, sq, n_all_count)

            if model not in model_dic:
                model_dic[model] = {}
            model_dic[model][score] = (ope_temp, n_ope_temp)
    np.save('./model_score_dic_len.npy', model_dic)

def get_model_exist_dic_by_one_dataset(dataset_id, notebook_list):
    cursor, db = create_connection()

    in_result = []
    sql = "SELECT distinct notebook_id FROM result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    in_ope = []
    sql = "SELECT distinct notebook_id FROM operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_ope.append(row[0])

    notebook_list_of_dataset = []
    model_dic = {}
    nmodel_dic = {}

    has_f1_score_list = {}
    nhas_f1_score_list = {}
    has_score_list_ = {}
    nhas_score_list_ = {}
    has_cross_val_score_list = {}
    nhas_cross_val_score_list = {}
    has_accuracy_score_list = {}
    nhas_accuracy_score_list = {}
    has_roc_auc_score_list = {}
    nhas_roc_auc_score_list = {}
    has_best_score_list = {}
    nhas_best_score_list = {}

    if notebook_list !=  []:
        notebook_list_of_dataset = notebook_list
    else:
        sql = "SELECT pair.nid FROM pair where pair.did=" + str(dataset_id)
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            notebook_list_of_dataset.append(int(row[0]))

    for notebook_id in in_result:
        if notebook_id not in notebook_list_of_dataset:
            continue
        if notebook_id in in_ope:
            one_res = get_one_model_result(notebook_id)
            for model in one_res:
                if model not in model_dic:
                    model_dic[model] = {}
                    for score in one_res[model]:
                        if score == 'f1_score':
                            if model not in has_f1_score_list:
                                has_f1_score_list[model] = []
                            has_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in has_score_list_:
                                has_score_list_[model] = []
                            has_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in has_cross_val_score_list:
                                has_cross_val_score_list[model] = []
                            has_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in has_accuracy_score_list:
                                has_accuracy_score_list[model] = []
                            has_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in has_roc_auc_score_list:
                                has_roc_auc_score_list[model] = []
                            has_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in has_best_score_list:
                                has_best_score_list[model] = []
                            has_best_score_list[model].append(one_res[model][score])


                        model_dic[model][score] = [[one_res[model][score]],1]
                else:
                    for score in one_res[model]:
                        if score == 'f1_score':
                            if model not in has_f1_score_list:
                                has_f1_score_list[model] = []
                            has_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in has_score_list_:
                                has_score_list_[model] = []
                            has_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in has_cross_val_score_list:
                                has_cross_val_score_list[model] = []
                            has_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in has_accuracy_score_list:
                                has_accuracy_score_list[model] = []
                            has_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in has_roc_auc_score_list:
                                has_roc_auc_score_list[model] = []
                            has_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in has_best_score_list:
                                has_best_score_list[model] = []
                            has_best_score_list[model].append(one_res[model][score])

                        if score not in model_dic[model]:
                            model_dic[model][score] = [[one_res[model][score]], 1]
                        else:
                            temp_li = model_dic[model][score][0]
                            temp_li.append(one_res[model][score])
                            model_dic[model][score] = [ temp_li , model_dic[model][score][1]+1 ]
        else:
            one_res = get_one_model_result(notebook_id)
            for model in one_res:
                if model not in nmodel_dic:
                    nmodel_dic[model] = {}
                    for score in one_res[model]:
                        print('model', model)
                        if score == 'f1_score':
                            if model not in nhas_f1_score_list:
                                nhas_f1_score_list[model] = []
                            nhas_f1_score_list[model].append(one_res[model][score])
                        if score == 'score':
                            if model not in nhas_score_list_:
                                nhas_score_list_[model] = []
                            nhas_score_list_[model].append(one_res[model][score])
                        if score == 'cross_val_score':
                            if model not in nhas_cross_val_score_list:
                                nhas_cross_val_score_list[model] = []
                            nhas_cross_val_score_list[model].append(one_res[model][score])
                        if score == 'accuracy_score':
                            if model not in nhas_accuracy_score_list:
                                nhas_accuracy_score_list[model] = []
                            nhas_accuracy_score_list[model].append(one_res[model][score])
                        if score == 'roc_auc_score':
                            if model not in nhas_roc_auc_score_list:
                                nhas_roc_auc_score_list[model] = []
                            nhas_roc_auc_score_list[model].append(one_res[model][score])
                        if score == 'best_score_':
                            if model not in nhas_best_score_list:
                                nhas_best_score_list[model] = []
                            nhas_best_score_list[model].append(one_res[model][score])

                        nmodel_dic[model][score] = [[one_res[model][score]],1]
                else:
                    for score in one_res[model]:
                        if score not in nmodel_dic[model]:
                            nmodel_dic[model][score] = [[one_res[model][score]], 1]
                        else:
                            temp_li = nmodel_dic[model][score][0]
                            temp_li.append(one_res[model][score])
                            nmodel_dic[model][score] = [ temp_li , nmodel_dic[model][score][1]+1 ]

    for model in nmodel_dic:
        for score in nmodel_dic[model]:
            all_score = 0
            all_count = 0
            if model in model_dic:
                if score in model_dic[model]:
                    all_score = model_dic[model][score][0]
                    all_count = model_dic[model][score][1]
            n_all_score = nmodel_dic[model][score][0]
            n_all_count = nmodel_dic[model][score][1]

            if all_count == 0:
                ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in all_score:
                    sum += sc
                mean = sum/all_count
                sq = 0
                for sc in all_score:
                    sq += (sc-mean)**2
                sq /= all_count
                ope_temp = (mean, sq, all_count)
            if n_all_count == 0:
                n_ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in n_all_score:
                    sum += sc
                mean = sum / n_all_count
                sq = 0
                for sc in n_all_score:
                    sq += (sc - mean)**2
                sq /= n_all_count
                n_ope_temp = (mean, sq, n_all_count)

            if model not in model_dic:
                model_dic[model] = {}
            model_dic[model][score] = (ope_temp, n_ope_temp)
    for model in model_dic:
        for score in model_dic[model]:
            if type(model_dic[model][score]).__name__ != 'tuple':
                all_score = model_dic[model][score][0]
                all_count = model_dic[model][score][1]
                n_ope_temp = (-1, -1, 0)
                if all_count == 0:
                    ope_temp = (-1, -1, 0)
                else:
                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum / all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc - mean) ** 2
                    sq /= all_count
                    ope_temp = (mean, sq, all_count)
                model_dic[model][score] = (ope_temp, n_ope_temp)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_has_f1_score_list', has_f1_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_nhas_f1_score_list', nhas_f1_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_has_score_list_', has_score_list_)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_nhas_score_list_', nhas_score_list_)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_has_cross_val_score_list', has_cross_val_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_nhas_cross_val_score_list', nhas_cross_val_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_has_accuracy_score_list', has_accuracy_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_nhas_accuracy_score_list', nhas_accuracy_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_has_roc_auc_score_list', has_roc_auc_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_nhas_roc_auc_score_list', nhas_roc_auc_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_has_best_score_list', has_best_score_list)
    np.save('1d1mgap1data/'+str(dataset_id)+'_model_nhas_best_score_list', nhas_best_score_list)
    return model_dic
def get_model_exist_dic_by_one_dataset_len(dataset_id, notebook_list):
    cursor, db = create_connection()

    in_result = {}
    sql = "SELECT distinct notebook_id FROM result"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result[row[0]]=[]
        sql = "SELECT distinct model_type from result where notebook_id = " + str(row[0])
        cursor.execute(sql)
        sql_res1 = cursor.fetchall()
        for row1 in sql_res1:
            in_result[row[0]].append(row1[0])

    in_ope = []
    sql = "SELECT distinct notebook_id FROM operator"
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_ope.append(row[0])

    notebook_list_of_dataset = []
    model_dic = {}
    nmodel_dic = {}

    if notebook_list !=  []:
        notebook_list_of_dataset = notebook_list
    else:
        sql = "SELECT pair.nid FROM pair where pair.did=" + str(dataset_id)
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            notebook_list_of_dataset.append(int(row[0]))

    for notebook_id in in_result:
        if notebook_id not in notebook_list_of_dataset:
            continue
        if notebook_id in in_ope:
            one_res = get_one_seq_len(notebook_id)
            for model in in_result[notebook_id]:
                if model not in model_dic:
                    model_dic[model] = [[one_res],1]
                else:
                    temp_li = model_dic[model][0]
                    temp_li.append(one_res)
                    model_dic[model] = [ temp_li , model_dic[model][1]+1 ]
        else:
            one_res = get_one_seq_len(notebook_id)
            for model in in_result[notebook_id]:
                if model not in nmodel_dic:
                    model_dic[model] = [[one_res],1]
                else:
                    temp_li = nmodel_dic[model][0]
                    temp_li.append(one_res)
                    nmodel_dic[model] = [ temp_li , nmodel_dic[model][1]+1 ]

    for model in nmodel_dic:
        for score in nmodel_dic[model]:
            all_score = 0
            all_count = 0
            if model in model_dic:
                if score in model_dic[score]:
                    all_score = model_dic[model][score][0]
                    all_count = model_dic[model][score][1]
            n_all_score = nmodel_dic[model][score][0]
            n_all_count = nmodel_dic[model][score][1]

            if all_count == 0:
                ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in all_score:
                    sum += sc
                mean = sum/all_count
                sq = 0
                for sc in all_score:
                    sq += (sc-mean)**2
                sq /= all_count
                ope_temp = (mean, sq, all_count)
            if n_all_count == 0:
                n_ope_temp = (-1, -1, 0)
            else:
                sum = 0
                for sc in n_all_score:
                    sum += sc
                mean = sum / n_all_count
                sq = 0
                for sc in n_all_score:
                    sq += (sc - mean)**2
                sq /= n_all_count
                n_ope_temp = (mean, sq, n_all_count)

            if model not in model_dic:
                model_dic[model] = {}
            model_dic[model][score] = (ope_temp, n_ope_temp)
    return model_dic

def get_operator_exist_dic_by_one_dataset(dataset_id, notebook_list):
    cursor, db = create_connection()
    CONFIG.read('config.ini')
    operator_dic = eval(CONFIG.get('operators', 'operations'))
    ope_dic = {}
    notebook_list_of_dataset = []
    operator_notebook_dic = {}
    noperator_notebook_dic = {}

    if notebook_list !=  []:
        notebook_list_of_dataset = notebook_list
    else:
        sql = "SELECT pair.nid FROM pair where pair.did=" + str(dataset_id)
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            notebook_list_of_dataset.append(int(row[0]))

    for operator in operator_dic.keys():
        sql = "SELECT distinct notebook_id FROM operator where operator = '" + operator + "'"
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        operator_notebook_dic[operator] = []
        if operator not in ope_dic.keys():
            ope_dic[operator] = {}
        for row in sql_res:
            notebook_id = int(row[0])
            if notebook_id not in notebook_list_of_dataset:
                continue
            operator_notebook_dic[operator].append(notebook_id)
            result = get_one_result(notebook_id)
            for i in result:
                if result[i] != -1:
                    if i not in ope_dic[operator]:
                        ope_dic[operator][i] = ([], 0)
                    all_score = ope_dic[operator][i][0]
                    all_count = ope_dic[operator][i][1]
                    all_score.append(result[i])
                    all_count += 1
                    ope_dic[operator][i] = (all_score, all_count)
                    print('add_one_result:',ope_dic[operator][i])

    for operator in operator_notebook_dic:
        noperator_notebook_dic[operator] = []
        for notebook in notebook_list_of_dataset:
            if notebook not in operator_notebook_dic[operator]:
                noperator_notebook_dic[operator].append(notebook)

    nope_dic = {}
    for operator in operator_dic.keys():
        for notebook_id in noperator_notebook_dic[operator]:
            result = get_one_result(notebook_id)
            if operator not in nope_dic.keys():
                nope_dic[operator] = {}
            for i in result:
                if result[i] != -1:
                    if i not in nope_dic[operator]:
                        nope_dic[operator][i] = ([], 0)
                    all_score = nope_dic[operator][i][0]
                    all_count = nope_dic[operator][i][1]
                    all_score.append(result[i])
                    all_count += 1
                    nope_dic[operator][i] = (all_score, all_count)

    for operator in ope_dic.keys():
        for score in ope_dic[operator].keys():
            score_list = ope_dic[operator][score][0]
            all_count = ope_dic[operator][score][1]
            if all_count != 0:
                sum = 0
                for sc in score_list:
                    sum += sc
                mean = sum/all_count
                sq=0
                for sc in score_list:
                    sq += (sc-mean)**2
                sq /= all_count
                ope_dic[operator][score]=(mean, sq, all_count)
            else:
                ope_dic[operator][score] = (-1, -1, all_count)

    for operator in nope_dic.keys():
        for score in nope_dic[operator].keys():
            score_list = nope_dic[operator][score][0]
            all_count = nope_dic[operator][score][1]
            if all_count != 0:
                sum = 0
                for sc in score_list:
                    sum += sc
                mean = sum/all_count
                sq=0
                for sc in score_list:
                    sq += (sc-mean)**2
                sq /= all_count
                nope_dic[operator][score]=(mean, sq, all_count)
            else:
                nope_dic[operator][score] = (-1, -1, all_count)

    # result = (ope_dic, nope_dic)
    for i in nope_dic:
        for j in nope_dic[i]:
            all_score = 0
            all_count = 0
            if i in ope_dic:
                if j in ope_dic[i]:
                    all_count = ope_dic[i][j][2]
            n_all_count = nope_dic[i][j][2]

            if all_count == 0:
                ope_temp = (-1, -1, 0)
            else:
                ope_temp = ope_dic[i][j]
            if n_all_count == 0:
                n_ope_temp = (-1,-1, 0)
            else:
                n_ope_temp = nope_dic[i][j]
            if i not in ope_dic:
                ope_dic[i] = {}
            ope_dic[i][j] =(ope_temp,n_ope_temp)
    for i in ope_dic:
        for j in ope_dic[i]:
            if type(ope_dic[i][j]).__name__ != 'tuple':
                all_score = ope_dic[i][j][0]
                all_count = ope_dic[i][j][1]
                n_ope_temp = (-1, -1, 0)
                if all_count == 0:
                    ope_temp = (-1, -1, 0)
                else:
                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum / all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc - mean) ** 2
                    sq /= all_count
                    ope_temp = (mean, sq, all_count)
                ope_dic[i][j] = (ope_temp, n_ope_temp)
    return ope_dic


def get_dataset_model_exist_dic():
    print("get_pair_dic")
    cursor, db = create_connection()
    in_result = []
    sql = 'select distinct(notebook_id) from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    pair_dic = {}
    sql = 'select * from pair'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id not in in_result:
            continue
        if dataset_id not in pair_dic.keys():
            pair_dic[dataset_id] = []
        pair_dic[int(dataset_id)].append(int(notebook_id))

    np.save('./pair_dic.npy',pair_dic)
    print("get_pair_dic end")
    # if number != '-1':
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id) limit " + str(number)
    # else:
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id)"
    # cursor.execute(sql)
    # sql_res = cursor.fetchall()
    result = {}
    count = 0
    for dataset_id in pair_dic:
        print(count)
        count += 1
        result[dataset_id] = get_model_exist_dic_by_one_dataset(dataset_id, pair_dic[dataset_id])
    np.save('./dataset_model_dic.npy',result)
    return result

def get_dataset_model_exist_dic_len():
    print("get_pair_dic")
    cursor, db = create_connection()
    in_result = []
    sql = 'select distinct(notebook_id) from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    pair_dic = {}
    sql = 'select * from pair'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id not in in_result:
            continue
        if dataset_id not in pair_dic.keys():
            pair_dic[dataset_id] = []
        pair_dic[int(dataset_id)].append(int(notebook_id))

    # if number != '-1':
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id) limit " + str(number)
    # else:
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id)"
    # cursor.execute(sql)
    # sql_res = cursor.fetchall()
    result = {}
    count = 0
    for dataset_id in pair_dic:
        print(count)
        count += 1
        result[dataset_id] = get_model_exist_dic_by_one_dataset_len(dataset_id, pair_dic[dataset_id])
    np.save('./0d1mgap1list.npy',result)
    return result


def get_model_operator_exist_dic():
    CONFIG.read('config.ini')
    operator_dic = eval(CONFIG.get('operators', 'operations'))

    model_operator_exist_dic = {}
    nmodel_operator_exist_dic = {}
    operator_notebook_dic = {}
    noperator_notebook_dic = {}

    cursor, db = create_connection()
    in_result = []
    sql = 'select distinct(notebook_id) from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    for operator in operator_dic.keys():
        sql = "SELECT distinct notebook_id FROM operator where operator = '" + operator + "'"
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        operator_notebook_dic[operator] = []
        for row in sql_res:
            notebook_id = int(row[0])
            if notebook_id not in in_result:
                continue

            operator_notebook_dic[operator].append(notebook_id)
            one_model_result = get_one_model_result(notebook_id)
            for model in one_model_result:
                if model not in model_operator_exist_dic:
                    model_operator_exist_dic[model] = {}
                if operator not in model_operator_exist_dic[model]:
                    model_operator_exist_dic[model][operator] = {}

                for i in one_model_result[model]:
                    if one_model_result[model][i] != -1:
                        if i not in model_operator_exist_dic[model][operator]:
                            model_operator_exist_dic[model][operator][i] = ([], 0)
                        all_score = model_operator_exist_dic[model][operator][i][0]
                        all_count = model_operator_exist_dic[model][operator][i][1]
                        all_score.append(one_model_result[model][i])
                        all_count += 1
                        model_operator_exist_dic[model][operator][i] = (all_score, all_count)
                        print('add_one_result:',model_operator_exist_dic[model][operator][i])

    for operator in operator_notebook_dic:
        noperator_notebook_dic[operator] = []
        for notebook in in_result:
            if notebook not in operator_notebook_dic[operator]:
                noperator_notebook_dic[operator].append(notebook)

    for operator in operator_dic.keys():
        for notebook_id in noperator_notebook_dic[operator]:
            if notebook_id not in in_result:
                continue

            one_model_result = get_one_model_result(notebook_id)
            for model in one_model_result:
                if model not in nmodel_operator_exist_dic:
                    nmodel_operator_exist_dic[model] = {}
                if operator not in nmodel_operator_exist_dic[model]:
                    nmodel_operator_exist_dic[model][operator] = {}

                for i in one_model_result[model]:
                    if one_model_result[model][i] != -1:
                        if i not in nmodel_operator_exist_dic[model][operator]:
                            nmodel_operator_exist_dic[model][operator][i] = ([], 0)
                        all_score = nmodel_operator_exist_dic[model][operator][i][0]
                        all_count = nmodel_operator_exist_dic[model][operator][i][1]
                        all_score.append(one_model_result[model][i])
                        all_count += 1
                        nmodel_operator_exist_dic[model][operator][i] = (all_score, all_count)
                        print('add_one_result:', nmodel_operator_exist_dic[model][operator][i])

    print('end_add_one_result')
    print(nmodel_operator_exist_dic)
    for model in nmodel_operator_exist_dic:
        print('model:',model)
        for operator in nmodel_operator_exist_dic[model]:
            print('operator:', operator)
            for score in nmodel_operator_exist_dic[model][operator]:
                print('score:', score)
                all_score = []
                all_count = 0
                if model in model_operator_exist_dic:
                    if operator in model_operator_exist_dic[model]:
                        if score in model_operator_exist_dic[model][operator]:
                            all_score = model_operator_exist_dic[model][operator][score][0]
                            all_count = model_operator_exist_dic[model][operator][score][1]
                n_all_score = nmodel_operator_exist_dic[model][operator][score][0]
                n_all_count = nmodel_operator_exist_dic[model][operator][score][1]

                if all_count == 0:
                    ope_temp = (-1, -1, 0)
                else:
                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum/all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc-mean)**2
                    sq /= all_count
                    ope_temp = (mean, sq, all_count)
                if n_all_count == 0:
                    n_ope_temp = (-1, -1, 0)
                else:
                    sum = 0
                    for sc in n_all_score:
                        sum += sc
                    mean = sum / n_all_count
                    sq = 0
                    for sc in n_all_score:
                        sq += (sc - mean)**2
                    sq /= n_all_count
                    n_ope_temp = (mean, sq, n_all_count)
                print('ope_temp', ope_temp)
                print('n_ope_temp', n_ope_temp)
                if model not in model_operator_exist_dic:
                    model_operator_exist_dic[model] = {}
                if operator not in model_operator_exist_dic[model]:
                    model_operator_exist_dic[model][operator] = {}
                model_operator_exist_dic[model][operator][score] =(ope_temp,n_ope_temp)
                print('model_operator_exist_dic',model_operator_exist_dic[model][operator][score])
    for model in model_operator_exist_dic:
        for operator in model_operator_exist_dic[model]:
            for score in model_operator_exist_dic[model][operator]:
                if type(model_operator_exist_dic[model][operator][score]).__name__ != 'tuple':
                    all_score = model_operator_exist_dic[model][operator][score][0]
                    all_count = model_operator_exist_dic[model][operator][score][1]
                    n_ope_temp = (-1, -1, 0)

                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum/all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc-mean)**2
                    sq /= all_count
                    ope_temp = (mean, sq, all_count)

                    model_operator_exist_dic[model][operator][score] =(ope_temp,n_ope_temp)
                    print('model_operator_exist_dic', model_operator_exist_dic[model][operator][score])
    np.save('./model_operator_exist_dic.npy', model_operator_exist_dic)

def get_model_operator_by_one_dataset(dataset_id,notebook_list):
    cursor, db = create_connection()
    CONFIG.read('config.ini')
    operator_dic = eval(CONFIG.get('operators', 'operations'))
    notebook_list_of_dataset = []
    operator_notebook_dic = {}
    noperator_notebook_dic = {}

    nmodel_operator_exist_dic = {}
    model_operator_exist_dic = {}
    if notebook_list != []:
        notebook_list_of_dataset = notebook_list
    else:
        sql = "SELECT pair.nid FROM pair where pair.did=" + str(dataset_id)
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            notebook_list_of_dataset.append(int(row[0]))

    for operator in operator_dic.keys():
        sql = "SELECT distinct notebook_id FROM operator where operator = '" + operator + "'"
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        operator_notebook_dic[operator] = []
        for row in sql_res:
            notebook_id = int(row[0])
            if notebook_id not in notebook_list_of_dataset:
                continue

            operator_notebook_dic[operator].append(notebook_id)
            one_model_result = get_one_model_result(notebook_id)
            for model in one_model_result:
                if model not in model_operator_exist_dic:
                    model_operator_exist_dic[model] = {}
                if operator not in model_operator_exist_dic[model]:
                    model_operator_exist_dic[model][operator] = {}

                for i in one_model_result[model]:
                    if one_model_result[model][i] != -1:
                        if i not in model_operator_exist_dic[model][operator]:
                            model_operator_exist_dic[model][operator][i] = ([], 0)
                        all_score = model_operator_exist_dic[model][operator][i][0]
                        all_count = model_operator_exist_dic[model][operator][i][1]
                        all_score.append(one_model_result[model][i])
                        all_count += 1
                        model_operator_exist_dic[model][operator][i] = (all_score, all_count)
                        print('add_one_result:',model_operator_exist_dic[model][operator][i])

    for operator in operator_notebook_dic:
        noperator_notebook_dic[operator] = []
        for notebook in notebook_list_of_dataset:
            if notebook not in operator_notebook_dic[operator]:
                noperator_notebook_dic[operator].append(notebook)

    for operator in operator_dic.keys():
        for notebook_id in noperator_notebook_dic[operator]:
            one_model_result = get_one_model_result(notebook_id)
            for model in one_model_result:
                if model not in nmodel_operator_exist_dic:
                    nmodel_operator_exist_dic[model] = {}
                if operator not in nmodel_operator_exist_dic[model]:
                    nmodel_operator_exist_dic[model][operator] = {}

                for i in one_model_result[model]:
                    if one_model_result[model][i] != -1:
                        if i not in nmodel_operator_exist_dic[model][operator]:
                            nmodel_operator_exist_dic[model][operator][i] = ([], 0)
                        all_score = nmodel_operator_exist_dic[model][operator][i][0]
                        all_count = nmodel_operator_exist_dic[model][operator][i][1]
                        all_score.append(one_model_result[model][i])
                        all_count += 1
                        nmodel_operator_exist_dic[model][operator][i] = (all_score, all_count)
                        print('add_one_result:', nmodel_operator_exist_dic[model][operator][i])

    for model in nmodel_operator_exist_dic:
        for operator in nmodel_operator_exist_dic[model]:
            for score in nmodel_operator_exist_dic[model][operator]:
                all_score = 0
                all_count = 0
                if model in model_operator_exist_dic:
                    if operator in model_operator_exist_dic[model]:
                        if score in model_operator_exist_dic[model][operator]:
                            all_score = model_operator_exist_dic[model][operator][score][0]
                            all_count = model_operator_exist_dic[model][operator][score][1]
                n_all_score = nmodel_operator_exist_dic[model][operator][score][0]
                n_all_count = nmodel_operator_exist_dic[model][operator][score][1]

                if all_count == 0:
                    ope_temp = (-1, -1, 0)
                else:
                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum/all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc-mean)**2
                    sq /= all_count
                    ope_temp = (mean, sq, all_count)
                if n_all_count == 0:
                    n_ope_temp = (-1, -1, 0)
                else:
                    sum = 0
                    for sc in n_all_score:
                        sum += sc
                    mean = sum / n_all_count
                    sq = 0
                    for sc in n_all_score:
                        sq += (sc - mean)**2
                    sq /= n_all_count
                    n_ope_temp = (mean, sq, n_all_count)
                print('ope_temp', ope_temp)
                print('n_ope_temp', n_ope_temp)
                if model not in model_operator_exist_dic:
                    model_operator_exist_dic[model] = {}
                if operator not in model_operator_exist_dic[model]:
                    model_operator_exist_dic[model][operator] = {}
                model_operator_exist_dic[model][operator][score] =(ope_temp,n_ope_temp)
                print('model_operator_exist_dic',model_operator_exist_dic[model][operator][score])
                # if all_count == 0:
                #     ope_temp = (-1, 0)
                # else:
                #     ope_temp = (all_score / all_count, all_count)
                # if n_all_count == 0:
                #     n_ope_temp = (-1, 0)
                # else:
                #     n_ope_temp = (n_all_score/n_all_count, n_all_count)
                #
                # if model not in model_operator_exist_dic:
                #     model_operator_exist_dic[model] = {}
                # if operator not in model_operator_exist_dic[model]:
                #     model_operator_exist_dic[model][operator] = {}
                # model_operator_exist_dic[model][operator][score] =(ope_temp,n_ope_temp)
    for model in model_operator_exist_dic:
        for operator in model_operator_exist_dic[model]:
            for score in model_operator_exist_dic[model][operator]:
                if type(model_operator_exist_dic[model][operator][score]).__name__ != 'tuple':
                    all_score = model_operator_exist_dic[model][operator][score][0]
                    all_count = model_operator_exist_dic[model][operator][score][1]
                    n_ope_temp = (-1, -1, 0)

                    sum = 0
                    for sc in all_score:
                        sum += sc
                    mean = sum/all_count
                    sq = 0
                    for sc in all_score:
                        sq += (sc-mean)**2
                    sq /= all_count
                    ope_temp = (mean, sq, all_count)

                    model_operator_exist_dic[model][operator][score] =(ope_temp,n_ope_temp)
                    # print('model_operator_exist_dic', model_operator_exist_dic[model][operator][score])
    return model_operator_exist_dic

def get_dataset_model_operator_exist_dic():
    print("get_pair_dic")
    cursor, db = create_connection()
    in_result = []
    sql = 'select distinct(notebook_id) from result'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        in_result.append(row[0])

    pair_dic = {}
    sql = 'select * from pair'
    cursor.execute(sql)
    sql_res = cursor.fetchall()
    for row in sql_res:
        notebook_id = int(row[0])
        dataset_id = int(row[1])
        if notebook_id not in in_result:
            continue
        if dataset_id not in pair_dic.keys():
            pair_dic[dataset_id] = []
        pair_dic[int(dataset_id)].append(int(notebook_id))

    np.save('./pair_dic.npy',pair_dic)
    print("get_pair_dic end")
    # if number != '-1':
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id) limit " + str(number)
    # else:
    #     sql = "SELECT count(distinct result.notebook_id),pair.did FROM result inner join pair on result.notebook_id = pair.nid group by pair.did order by count(distinct result.notebook_id)"
    # cursor.execute(sql)
    # sql_res = cursor.fetchall()
    result = {}
    count = 0
    for dataset_id in pair_dic:
        print(count)
        count += 1
        result[dataset_id] = get_model_operator_by_one_dataset(dataset_id, pair_dic[dataset_id])
    np.save('./dataset_model_operation_dic.npy',result)

if __name__ == '__main__':
    # print(get_exist_dic())
    print('input stat type:')
    print('1: get_exist_dic') # 0d0m gap1 y
    print('1.1: get_exist_dic_len') # 0d0m gap2 y
    print('2: get_mean_group_by_dataset') # no use
    print('3: get_dataset_exist_dic') # 1d0m gap1 y
    print('3.1: get_dataset_exist_dic') # 1d0m gap2 y
    print('4: get_operator_exist_dic') # 0d0m gap3 y
    # print('5: get_operator_param_score')
    print('6: get_dataset_operator_exist_dic') # 1d0m gap3 y
    print('7: get_model_exist_dic') # 0d1m gap1 n
    print('7.1: get_model_exist_dic') # 0d1m gap2 n
    print('8: get_dataset_model_exist_dic') # 1d1m gap1 y
    print('8.1: get_dataset_model_exist_dic_len') # 1d1m gap2 y
    print('9: get_model_operator_exist_dic') # 0d1m gap3 y
    print('10: get_dataset_model_operator_exist_dic') # 1d1m gap3 y
    # print('7: get_result_of_seq')
    # print('8: get_show_sequence')
    # print('9: get_operator_param_rate')
    print('-1: show_dic')
    rtype = input()

    if rtype == '1':
        res = get_exist_dic()
    elif rtype == '1.1':
        res = get_exist_dic_len()
    elif rtype == '2':
        res = get_mean_group_by_dataset()
    elif rtype == '3':
        res = get_dataset_exist_dic()
    elif rtype == '3.1':
        res = get_dataset_exist_dic_len()
    elif rtype == '4':
        res = get_operator_exist_dic()
    elif rtype == '5':
        res = get_operator_param_score()
    elif rtype == '6':
        # print('input dataset number:')
        # dataset_number = input()
        res = get_dataset_operator_exist_dic()
    elif rtype == '7':
        res = get_model_exist_dic()
    elif rtype == '7.1':
        res = get_model_exist_dic_len()
    elif rtype == '8':
        res = get_dataset_model_exist_dic()
    elif rtype == '8.1':
        res = get_dataset_model_exist_dic_len()
    elif rtype == '9':
        res = get_model_operator_exist_dic()
    elif rtype == '10':
        res = get_dataset_model_operator_exist_dic()

    elif rtype == '-1':
        # dataset,operator,model,exist -> result
        print('input show type:')
        print('1: get_exist_dic') # 0d0mgap1
        # print('2: get_mean_group_by_dataset')
        print('3: get_dataset_exist_dic') # 1d0mgap1
        print('4: get_operator_exist_dic') # 0d0mgap3
        # print('5: get_operator_param_score')
        print('6: get_dataset_operator_exist_dic')
        print('7: get_model_exist_dic')
        print('8: get_dataset_model_exist_dic')
        print('9: get_model_operator_exist_dic')
        print('10: get_dataset_model_operator_exist_dic')
        # print('8: get_show_sequence')
        dic_path = input()
        res = show_dic(dic_path)
    # res = get_mean_group_by_dataset()
    # for i in res:
    #     #     if res[i] != -1:
    #     #         print(str(i)+':', res[i])