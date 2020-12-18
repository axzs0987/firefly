
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
# from action import do_an_action
from action import get_code_txt
# from action import get_origin_state
from action import create_connection
from action import CONFIG

from utils import get_host_ip

import grpc
import os
import message_pb2
import message_pb2_grpc
import eventlet
import pprint

class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs0, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)

    def covert_to_array(self):
        array_obs = np.vstack(self.ep_obs)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []


class ActorNetwork(object):
    def __init__(self, act_1_dim, act_2_dim, dense_dim, name):
        self.act_1_dim = act_1_dim
        self.act_2_dim = act_2_dim
        self.dense_dim = dense_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, self.dense_dim, # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh, # 激活函数
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            act_1_prob = tf.layers.dense(h1, self.act_1_dim, None,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            temp_layer = tf.concat([h1,act_1_prob], axis=1)
            act_2_prob = tf.layers.dense(temp_layer, self.act_2_dim, None,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return act_1_prob,act_2_prob

    def choose_action(self, obs, reuse=False):
        act_1_prob, act_2_prob = self.step(obs, reuse)
        all_act_1_prob = tf.nn.softmax(act_1_prob, name='act_prob')  # use softmax to convert to probability
        all_act_2_prob = tf.nn.softmax(act_2_prob, name='act_prob')  # use softmax to convert to probability
        return (all_act_1_prob,all_act_2_prob)

    def get_neglogp(self, obs, act, reuse=True):
        act_1_prob, act_2_prob = self.step(obs, reuse)
        neglogp_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_1_prob, labels=act[:,0]) # 计算act_prob和act的稀疏softmax 交叉熵
        neglogp_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_2_prob,labels=act[:,1])  # 计算act_prob和act的稀疏softmax 交叉熵
        return act_1_prob,act_2_prob,neglogp_1,neglogp_2


class PolicyGradient(object):
    def __init__(self, act_1_dim, act_2_dim, obs_dim, dense_dim, lr, gamma):
        self.act_1_dim = act_1_dim #有多少个operator
        self.act_2_dim = act_2_dim  # 有多少个column
        self.obs_dim = obs_dim #有多少个状态？

        self.lr = lr #学习率
        self.gamma = gamma #衰减率
        self.dense_dim = dense_dim

        self.OBS = tf.placeholder(tf.float32, [None, self.obs_dim], name="observation")
        self.ACT = tf.placeholder(tf.int32, [None, 2], name="action")
        self.RWD = tf.placeholder(tf.float32, [None, ], name="discounted_reward")

        actor = ActorNetwork(self.act_1_dim, self.act_2_dim , self.dense_dim, 'actor')
        self.memory = Memory()

        self.action = actor.choose_action(self.OBS)
        self.temp_a1_a2 = actor.step(self.OBS, True)
        self.temp_n1n2 = actor.get_neglogp(self.OBS, self.ACT)

        act_prob1,act_prob2,neglogp1, neglogp2 = self.temp_n1n2
        self.loss = tf.reduce_mean(neglogp1 * self.RWD) + tf.reduce_mean(neglogp2 * self.RWD)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        ckpt_state = tf.train.get_checkpoint_state('./models/')
        self.sess = tf.Session()
        if ckpt_state:
            print('restore')
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_state.model_checkpoint_path)
        else:
            print('create new')
            self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        self.is_choiced = []
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        self.prob_weights_1,self.prob_weights_2 = self.sess.run(self.action, feed_dict={self.OBS: obs}) # action,得到每一个action的概率
        print_item = self.prob_weights_1.ravel()
        for index, item in enumerate(print_item):
            print(index, str(item))
        action1 = np.random.choice(range(self.prob_weights_1.shape[1]), p=self.prob_weights_1.ravel()) # 按照给出的概率随机选一个action
        action2 = np.random.choice(range(self.prob_weights_2.shape[1]), p=self.prob_weights_2.ravel())  # 按照给出的概率随机选一个action
        self.is_choiced.append(str(action1)+'+' + str(action2))
        return action1,action2 # 输入obs，按照策略pai给出相应的执行动作
    def step_1(self):
        action1 = np.random.choice(range(self.prob_weights_1.shape[1]),
                                   p=self.prob_weights_1.ravel())  # 按照给出的概率随机选一个action
        action2 = np.random.choice(range(self.prob_weights_2.shape[1]),
                                   p=self.prob_weights_2.ravel())  # 按照给出的概率随机选一个action
        count = 0
        while str(action1)+'+' + str(action2) in self.is_choiced and count < 20:
            count += 1
            if count < 10:
                # print(str(action1)+'+' + str(action2))
                action1 = np.random.choice(range(self.prob_weights_1.shape[1]),
                                           p=self.prob_weights_1.ravel())  # 按照给出的概率随机选一个action
                action2 = np.random.choice(range(self.prob_weights_2.shape[1]),
                                           p=self.prob_weights_2.ravel())  # 按照给出的概率随机选一个action
            else:
                # print(str(action1) + '+' + str(action2))
                action1 = np.random.choice(range(self.prob_weights_1.shape[1]))  # 按照给出的概率随机选一个action
                action2 = np.random.choice(range(self.prob_weights_2.shape[1]))  # 按照给出的概率随机选一个action
        self.is_choiced.append(str(action1) + '+' + str(action2))
        return action1, action2  # 输入obs，按照策略pai给出相应的执行动作
        # return action2  # 输入obs，按照策略pai给出相应的执行动作
    def learn(self):
        #现在样本池有了一个完整序列样本

        obs, act, rwd = self.memory.covert_to_array() # 取出所有样本（完整序列）
        # print('reward:',rwd)
        discounted_rwd = self.discount_and_norm_rewards(rwd) #通过已知每一步的r，计算累计奖励R（利用衰减gamma），并且归一化
        # print('discount:',discounted_rwd)
        saver = tf.train.Saver()
        res = self.sess.run(self.temp_a1_a2, feed_dict={self.OBS: obs})
        # print(discounted_rwd)
        self.sess.run(self.optimizer, feed_dict={self.OBS: obs, self.ACT: act, self.RWD: discounted_rwd})
        loss = self.sess.run(self.loss, feed_dict={self.OBS: obs, self.ACT: act, self.RWD: discounted_rwd})
        res = self.sess.run(self.temp_n1n2, feed_dict={self.OBS: obs, self.ACT: act, self.RWD: discounted_rwd})
        print('act1_prob', res[0])
        print('act2_prob', res[1])
        print('neg1', res[2])
        print('neg2', res[3])
        print('act1',act[:,0])
        print('act2', act[:, 1])
        print(loss)
        self.prob_weights_1, self.prob_weights_2 = self.sess.run(self.action,
                                                                 feed_dict={self.OBS: obs})  # action,得到每一个action的概率
        # print('act_1_weight_after:', self.prob_weights_1.ravel())
        save_path = saver.save(self.sess, "models/1125.ckpt")
        self.memory.reset()
        return loss

    def discount_and_norm_rewards(self, rwd):
        discounted_rwd = np.zeros_like(rwd)
        running_add = 0
        for t in reversed(range(0, len(rwd))):
            running_add = running_add * self.gamma + rwd[t]
            discounted_rwd[t] = running_add
        # discounted_rwd -= np.mean(discounted_rwd)
        # if np.std(discounted_rwd) != 0:
        #     discounted_rwd /= np.std(discounted_rwd)
        return discounted_rwd

def rpc_client_do_an_action(notebook_id,notebook_code,target_content,column_num,res_line_number,ip):
    def unparse_matrix(response_values):
        result = []
        for one in response_values:
            b = []
            for i in one.row:
                b.append(i)
            result.append(b)
        return result
    request = message_pb2.do_an_action_param(notebook_id=notebook_id,notebook_code=notebook_code,column_num=column_num-1,res_line_number=res_line_number)
    request.target_content_operation = target_content['operation']
    request.target_content_data_object = target_content['data_object']

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = message_pb2_grpc.MsgServiceStub(channel)
        # print('?>>???')
        response = stub.rpc_do_an_action(request)
        # print('?>>???')
        s_t = unparse_matrix(response.s_t)
        s_t_plus_1 = unparse_matrix(response.s_t_plus_1)
        action_1 = response.action_1
        action_2 = response.action_2
        action = (action_1,action_2)
        reward = response.reward
        terminal = response.terminal
        notebook_code = response.new_code
        res_line_number = response.res_line_number
        len_data_plus_1 = response.len_data_plus_1
        return s_t, action, reward, terminal, s_t_plus_1, notebook_code, res_line_number, len_data_plus_1

def rpc_client_get_origin_state(notebook_id, notebook_code,column_num,ip):
    def unparse_matrix(response_values):
        result = []
        for one in response_values:
            b = []
            for i in one.row:
                b.append(i)
            result.append(b)
        if result == []:
            result = 'run failed'
        return result
    request = message_pb2.get_origin_state_param(notebook_id=notebook_id,notebook_code=notebook_code,column_num=column_num-1)
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = message_pb2_grpc.MsgServiceStub(channel)
        response = stub.rpc_get_origin_state(request)
        s_t = unparse_matrix(response.s_t)
        len_data = response.len_data
        return s_t,len_data

def check_action_by_rule(action1, action2, s_t, len_data,column_num):
    if action2 != column_num and action2 >= len_data:
        return False
    is_column = True
    if action2 == column_num:
        is_column = False
    if is_column == False:
        if action1 in [1,2,3,4,5,6,8,9,12,13,14,15,16,17,18,19,22,23,24]:
            return True
        else:
            return False
    else:
        # print('s_t[action2]',s_t[action2])
        column_type = s_t[action2][3]
        if column_type == 1: # numeric
            if action1 in [1,5,6,7,9,10,11,15,18,19,20,21,22,23,24,25]:
                return True
            else:
                return False
        if column_type == 2: # numeric
            if action1 in [2,6,9,10,11,15,18,19]:
                return True
            else:
                return False

def train(notebook_root,dataset_root,ip):
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

    def create_notebook_pool():
        notebook_pool = []
        in_result = []
        cursor, db = create_connection()
        sql = 'select distinct notebook_id from result'
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            in_result.append(int(row[0]))


        in_notebook = []
        sql = 'select distinct id from notebook where isRandom=1'
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            in_notebook.append(int(row[0]))

        sql = 'select pair.nid from pair,dataset where pair.did=dataset.id and dataset.server_ip = \'' + ip + '\''
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            if int(row[0]) not in in_result:
                continue
            if int(row[0]) not in in_notebook:
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

    agent = PolicyGradient(act_1_dim=act_1_dim+1, act_2_dim=column_num, obs_dim=obs_dim, dense_dim=dense_dim, lr=learning_rate, gamma=gamma)

    if os.path.exists('reward_list.npy'):
        print('exists')
        reward_list = list(np.load('./reward_list.npy',allow_pickle=True))
    else:
        reward_list = []

    if os.path.exists('loss_pg.npy'):
        print('loss exists')
        loss_list = list(np.load('./loss_pg.npy',allow_pickle=True))
    else:
        loss_list = []
    iteration = 0
    if os.path.exists('act_reward_pg.npy'):
        print('exists')
        act_reward = np.load('./act_reward_pg.npy',allow_pickle=True).item()
    else:
        act_reward = {}
    for i_episode in range(nepisode):
        ep_rwd = 0
        notebook_id = random.choice(notebook_pool)
        print("\033[0;35;40m"+"notebook_id:"+str(notebook_id)+"\033[0m")
        notebook_path = notebook_root + str(notebook_id) + '.ipynb'
        notebook_code = get_code_txt(notebook_path)
        res_line_number = -1
        s_t,len_data = rpc_client_get_origin_state(notebook_id,notebook_code,column_num,ip)
        # print(s_t)
        check_result,model_list = check_model(notebook_id)
        while s_t == 'run failed' or check_result == False:
            notebook_pool.remove(notebook_id)
            notebook_id = random.choice(notebook_pool)
            print("\033[0;34;40m" +"notebook_id:"+ str(notebook_id) + "\033[0m")
            notebook_path = notebook_root + str(notebook_id) + '.ipynb'
            notebook_code = get_code_txt(notebook_path)
            s_t,len_data = rpc_client_get_origin_state(notebook_id, notebook_code,column_num,ip)
            check_result,model_list = check_model(notebook_id)

        s_t_p = s_t
        s_t = np.ravel(s_t)
        type_ = np.array([int(np.load('type.npy', allow_pickle=True))])
        if len(s_t) == 1900:
            s_t = np.concatenate((type_, s_t), axis=0)
        if len(s_t) == 1901:
            s_t = np.concatenate((s_t, model_list), axis=0)
        # pprint.pprint(s_t)
        if len(s_t) == 0 :
            continue
        while True:
            terminal1 = False
            if int(np.load('type.npy', allow_pickle=True)) != 1:
                terminal1 = True
            action1, action2 = agent.step(s_t) # 已知当前状态，通过网络预测预测下一步的动作(这里要改)
            # print(len_data)
            # print(len(s_t_p))
            check_res = check_action_by_rule(action1 + 1, action2, s_t_p, len_data,column_num=column_num)
            count = 0
            s_t_plus_1 = np.zeros([1942])
            if check_res == False:
                reward = -1.0
                terminal = True
            # while cehck_res == False and count < 10:
            #     print('changed_act1:',action1)
            #     count += 1
            #     action1,action2 = agent.step_1()  # 已知当前状态，通过网络预测预测下一步的动作(这里要改)
            #     cehck_res = check_action_by_rule(action1 + 1, action2 + 1, s_t_p,len_data,column_num=column_num)
            else:
                if action2 == column_num-1:
                    target_content = {
                        'operation': action1 + 1,
                        'data_object': -1,
                    }
                else:
                    target_content = {
                        'operation': action1+1,
                        'data_object': action2,
                    }
                # print('?>>?')
                # print('act:',act)
                # 执行动作，得到新状态，立即回报，是否终止
                # s_t = []
                eventlet.monkey_patch()
                try:
                    with eventlet.Timeout(60, False):  # 设置超时时间为2秒
                        s_t, action, reward, terminal, s_t_plus_1, notebook_code, res_line_number, len_data_plus_1 = rpc_client_do_an_action(notebook_id,notebook_code,target_content,column_num,res_line_number,ip)
                except:
                    break
                if s_t == []:
                    break
                # print('?>>?')
                if reward == -2:
                    continue
                # print("\033[0;36;40m" + "s_t:" + str(s_t) + "\033[0m")

                s_t = np.ravel(s_t)
                type_ = np.array([int(np.load('type.npy', allow_pickle=True))])
                if int(np.load('type.npy', allow_pickle=True)) != 1:
                    terminal = True
                if len(s_t) == 1900:
                    s_t = np.concatenate((type_, s_t), axis=0)
                if len(s_t) == 1901:
                    s_t = np.concatenate((s_t, model_list), axis=0)

                s_t_p = s_t_plus_1
                s_t_plus_1 = np.ravel(s_t_plus_1)
                if len(s_t_plus_1) == 1900:
                    s_t_plus_1 = np.concatenate(([0], s_t_plus_1), axis=0)
                if len(s_t_plus_1) == 1901:
                    s_t_plus_1 = np.concatenate((s_t_plus_1, model_list), axis=0)
                s_t = s_t_plus_1
                len_data = len_data_plus_1
            act = (action1, action2)
            agent.memory.store_transition(s_t, act, reward)  # 放入采样池(这里要改)
            ep_rwd += reward
            print("\033[0;36;40m" + "reward:" + str(reward) + "\033[0m")
            print("\033[0;36;40m" + "terminal:" + str(terminal) + "\033[0m")
            print("\033[0;36;40m" + "act:" + str(act) + "\033[0m")
            if terminal or terminal1:
                loss = agent.learn() # 一个完整过程终止，开始优化网络
                loss_list.append(loss)

                print('Ep: %i' % i_episode, "|Ep_r: %f" % ep_rwd)
                sql = 'update notebook set trained_time = trained_time + 1 where id=' + str(notebook_id)
                cursor, db = create_connection()
                cursor.execute(sql)
                db.commit()
                if act[0] not in act_reward:
                    act_reward[act[0]] = []
                act_reward[act[0]].append(ep_rwd)
                reward_list.append(ep_rwd)
                # if i_episode % 50 == 0:
                np.save('./reward_list.npy',reward_list)
                np.save('loss_pg', loss_list)
                np.save('act_reward_pg', act_reward)
                break


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    ip = get_host_ip()
    server_dic = eval(CONFIG.get('server', 'server'))
    notebook_root = server_dic[ip]['npath']
    dataset_root = server_dic[ip]['dpath']

    train(notebook_root, dataset_root, ip)