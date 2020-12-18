from utils import create_connection
from utils import CONFIG
from utils import get_host_ip

import random
from action import get_code_txt
from action import get_origin_state
import numpy as np


def addSamples(notebook_root,dataset_root,ip):
    def create_notebook_pool():
        notebook_pool = []
        in_result = []
        cursor, db = create_connection()
        sql = 'select distinct notebook_id from result'
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            in_result.append(int(row[0]))
        sql = 'select pair.nid from pair,dataset where pair.did=dataset.id and dataset.server_ip = \'' + ip + '\''
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            if int(row[0]) not in in_result:
                continue
            if int(row[0]) not in notebook_pool:
                notebook_pool.append(int(row[0]))
        return notebook_pool

    notebook_pool = create_notebook_pool()
    train_config = eval(CONFIG.get('train', 'train'))
    nepisode = train_config['nepisode']
    obs_dim = train_config['obs_dim']
    ope_dic = eval(CONFIG.get('operators', 'operations'))
    learning_rate = train_config['learning_rate']
    gamma = train_config['gamma']
    dense_dim = train_config['dense_dim']
    column_num = train_config['column_num']
    act_1_dim = 0
    for item in ope_dic:
        if ope_dic[item]['index'] > act_1_dim:
            act_1_dim = ope_dic[item]['index'] # 27

    # agent = PolicyGradient(act_1_dim=act_1_dim, act_2_dim=column_num, obs_dim=obs_dim, dense_dim=dense_dim, lr=learning_rate, gamma=gamma)
    for i_episode in range(nepisode):
        ep_rwd = 0
        notebook_id = random.choice(notebook_pool)
        print("\033[0;35;40m"+"notebook_id:"+str(notebook_id)+"\033[0m")
        notebook_path = notebook_root + str(notebook_id) + '.ipynb'
        notebook_code = get_code_txt(notebook_path)
        res_line_number = -1
        s_t = get_origin_state(notebook_id,notebook_code,column_num, dataset_root=dataset_root)
        print(s_t)
        while s_t == 'run failed':
            notebook_pool.remove(notebook_id)
            notebook_id = random.choice(notebook_pool)
            print("\033[0;34;40m" +"notebook_id:"+ str(notebook_id) + "\033[0m")
            notebook_path = notebook_root + str(notebook_id) + '.ipynb'
            notebook_code = get_code_txt(notebook_path)
            s_t = get_origin_state(notebook_id, notebook_code,column_num, dataset_root=dataset_root)
        print(s_t)
        while True:
            s_t = np.ravel(s_t)
            action1, action2 = agent.step(s_t) # 已知当前状态，通过网络预测预测下一步的动作(这里要改)
            target_content = {
                'operation': action1,
                'data_object': action2,
            }
            act = (action1, action2)
            s_t, action, reward, terminal, s_t_plus_1, notebook_code, res_line_number = do_an_action(notebook_id,notebook_code,target_content,column_num,res_line_number) # 执行动作，得到新状态，立即回报，是否终止
            print("\033[0;36;40m" + "reward:" + str(reward) + "\033[0m")
            print("\033[0;36;40m" + "terminal:" + str(terminal) + "\033[0m")
            s_t = np.ravel(s_t)
            s_t_plus_1 = np.ravel(s_t_plus_1)
            agent.memory.store_transition(s_t, act, reward) # 放入采样池(这里要改)
            s_t = s_t_plus_1
            ep_rwd += reward

            if terminal:
                agent.learn() # 一个完整过程终止，开始优化网络
                print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
                break

if __name__ == '__main__':
    ip = get_host_ip()
    server_dic = eval(CONFIG.get('server', 'server'))
    notebook_root = server_dic[ip]['npath']
    dataset_root = server_dic[ip]['dpath']

    addSamples(notebook_root, dataset_root, ip)