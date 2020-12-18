import numpy as np
# import tensorflow as tf
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from action import create_connection
from action import CONFIG

from action import get_code_txt
from utils import get_host_ip
import message_pb2
import message_pb2_grpc
import grpc
import os
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        if os.path.exists('samplepool.npy'):
            self.buffer = list(np.load('samplepool.npy',allow_pickle=True))
        else:
            self.buffer = []
        self.capacity = capacity
        if os.path.exists('index.npy'):
            self.index = int(np.load('index.npy',allow_pickle=True))
        else:
            self.index = 0

    def store_transition(self, obs0, act_1, act_2, rwd, obs1, done):
        data = (obs0, act_1, act_2, rwd, obs1, done)
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs0, act_1, act_2, rwd, obs1, done = map(np.stack, zip(*batch))
        return obs0, act_1, act_2, rwd, obs1, done
    def save_buffer(self):
        np.save('samplepool.npy', self.buffer)
        np.save('index.npy', self.index)

class QValueNetwork(object):
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
            value_1 = tf.layers.dense(h1, self.act_1_dim, None,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            temp_layer = tf.concat([h1,value_1], axis=1)
            value_2 = tf.layers.dense(temp_layer, self.act_2_dim, None,
                                      kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return value_1,value_2

    def get_q_value(self, obs, reuse=False):
        q_1_value, q_2_value = self.step(obs, reuse)
        return q_1_value, q_2_value


class DQN(object):
    def __init__(self, act_dim_1, act_dim_2, dense_dim, obs_dim, lr_q_value, gamma, epsilon, batch_size):
        self.act_dim_1 = act_dim_1
        self.act_dim_2 = act_dim_2
        self.dense_dim = dense_dim
        self.obs_dim = obs_dim
        self.lr_q_value = lr_q_value
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.OBS0 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations0")
        self.OBS1 = tf.placeholder(tf.float32, [None, self.obs_dim], name="observations1")
        self.ACT_1 = tf.placeholder(tf.int32, [None], name="action1")
        self.ACT_2 = tf.placeholder(tf.int32, [None], name="action2")
        self.RWD = tf.placeholder(tf.float32, [None], name="reward")
        self.TARGET_Q_1 = tf.placeholder(tf.float32, [None], name="target_q_value")
        self.TARGET_Q_2 = tf.placeholder(tf.float32, [None], name="target_q_value")
        self.DONE = tf.placeholder(tf.float32, [None], name="done")

        q_value_network = QValueNetwork(self.act_dim_1, self.act_dim_2, self.dense_dim, 'q_value')
        target_q_value_network = QValueNetwork(self.act_dim_1, self.act_dim_2, self.dense_dim, 'target_q_value')
        self.memory = ReplayBuffer(capacity=int(1e6))

        self.q_value = q_value_network.get_q_value(self.OBS0)
        q_value_1, q_value_2 = self.q_value
        self.action_1_onehot = tf.one_hot(self.ACT_1, self.act_dim_1, dtype=tf.float32)
        self.q_value_1_onehot = tf.reduce_sum(tf.multiply(q_value_1, self.action_1_onehot), axis=1)

        self.action_2_onehot = tf.one_hot(self.ACT_2, self.act_dim_2, dtype=tf.float32)
        self.q_value_2_onehot = tf.reduce_sum(tf.multiply(q_value_2, self.action_2_onehot), axis=1)

        target_q_value = target_q_value_network.get_q_value(self.OBS1)
        target_q_value_1, target_q_value_2 = target_q_value

        self.target_q_value1_1 = self.RWD + (1. - self.DONE) * self.gamma \
                               * tf.reduce_max(target_q_value_1, axis=1)
        self.target_q_value1_2 = self.RWD + (1. - self.DONE) * self.gamma \
                               * tf.reduce_max(target_q_value_2, axis=1)

        self.q_value_loss = tf.reduce_mean(tf.square(self.q_value_1_onehot - self.TARGET_Q_1) + tf.square(self.q_value_2_onehot - self.TARGET_Q_2))
        self.q_value_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_q_value).minimize(self.q_value_loss)

        self.q_value_params = tf.global_variables('q_value')
        self.target_q_value_params = tf.global_variables('target_q_value')
        self.target_updates = [tf.assign(tq, q) for tq, q in zip(self.target_q_value_params, self.q_value_params)]

        self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

        ckpt_state = tf.train.get_checkpoint_state('./models_dqn_new_reward/')
        self.sess = tf.Session()
        if ckpt_state:
            print('restore')
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_state.model_checkpoint_path)
            self.sess.run(self.target_updates)
        else:
            print('create new')
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_updates)

    def choose_action_2(self, act_2_prob, obs, len_data, action1):
        def up_to_zero(act_2_prob):
            # print('before:', act_2_prob)
            min_value = np.min(act_2_prob)
            # print('min_value:', min_value)
            if min_value >= 0:
                return act_2_prob
            else:
                act_2_prob = act_2_prob + abs(min_value)
                # print('after:', act_2_prob)
                return act_2_prob

        def change_zero(action2):
            act_2_prob[action2] = 0

        act_2_prob =up_to_zero(act_2_prob)
        action1 = action1 + 1
        for action2 in range(0, self.act_dim_2):
            if action2 != self.act_dim_2-1 and action2 >= len_data:
                # print('error', action2)
                # change_zero(action2)
                change_zero(action2)
            is_column = True
            if action2 == self.act_dim_2-1:
                is_column = False
            if is_column == False:
                if action1 in [1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24]:
                    continue
                else:
                    # print('error', action2)
                    change_zero(action2)
            else:
                column_type = obs[action2][3]
                if column_type == 1:  # numeric
                    if action1 in [1, 5, 6, 7, 9, 10, 11, 15, 18, 19, 20, 21, 22, 23, 24, 25]:
                        continue
                    else:
                        # print('error', action2)
                        change_zero(action2)
                if column_type == 2:  # numeric
                    if action1 in [2, 6, 9, 10, 11, 15, 18, 19]:
                        continue
                    else:
                        # print('error',action2)
                        change_zero(action2)

        # print('self.act_dim_2-1',self.act_dim_2-1)
        # print('len_data', len_data)
        # print('act1:',action1)
        # for index, item in enumerate(obs):
        #     print(index,item)
        # print('act_2_prob:', act_2_prob)
        # print('obs:', np.array(obs).shape)
        return act_2_prob
    def step(self, obs, len_data):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        action1,action2 = self.sess.run(self.q_value, feed_dict={self.OBS0: obs})
        if np.random.rand(1) < self.epsilon:
            action1 = np.random.randint(0, self.act_dim_1)
        else:
            print('action1_value:',action1)
            action1 = np.argmax(action1, axis=1)[0]

        if np.random.rand(1) < self.epsilon:
            print('\033[0;35;40mchose action: random\033[0m')
            action2 = np.random.randint(0, self.act_dim_2)
        else:
            print('\033[0;35;40mchose action: max\033[0m')
            action2 = self.choose_action_2(action2, obs, len_data, action1)
            action2 = np.argmax(action2, axis=1)[0]

        return action1, action2

    def learn(self):
        obs0, act_1, act_2, rwd, obs1, done = self.memory.sample(batch_size=128)

        target_q_value1_1 = self.sess.run(self.target_q_value1_1,
                                        feed_dict={self.OBS1: obs1, self.RWD: rwd, self.DONE: np.float32(done)})
        target_q_value1_2 = self.sess.run(self.target_q_value1_2,
                                        feed_dict={self.OBS1: obs1, self.RWD: rwd, self.DONE: np.float32(done)})

        # print('tqv2:', target_q_value1_2)
        # print('tqv1:', target_q_value1_1)
        self.sess.run(self.q_value_train_op,feed_dict={self.OBS0: obs0, self.ACT_1: act_1,
                                                       self.ACT_2: act_2,
                                                       self.TARGET_Q_1: target_q_value1_1,
                                                       self.TARGET_Q_2: target_q_value1_2,})
        loss = self.sess.run(self.q_value_loss,feed_dict={self.OBS0: obs0, self.ACT_1: act_1,
                                                       self.ACT_2: act_2,
                                                       self.TARGET_Q_1: target_q_value1_1,
                                                       self.TARGET_Q_2: target_q_value1_2,})
        q_value_1_onehot = self.sess.run(self.q_value_1_onehot, feed_dict={self.OBS0: obs0, self.ACT_1: act_1,
                                                           self.ACT_2: act_2,
                                                           self.TARGET_Q_1: target_q_value1_1,
                                                           self.TARGET_Q_2: target_q_value1_2, })
        TARGET_Q_1 = self.sess.run(self.TARGET_Q_1, feed_dict={self.OBS0: obs0, self.ACT_1: act_1,
                                                           self.ACT_2: act_2,
                                                           self.TARGET_Q_1: target_q_value1_1,
                                                           self.TARGET_Q_2: target_q_value1_2, })
        q_value_2_onehot = self.sess.run(self.q_value_2_onehot, feed_dict={self.OBS0: obs0, self.ACT_1: act_1,
                                                           self.ACT_2: act_2,
                                                           self.TARGET_Q_1: target_q_value1_1,
                                                           self.TARGET_Q_2: target_q_value1_2, })
        TARGET_Q_2 = self.sess.run(self.TARGET_Q_2, feed_dict={self.OBS0: obs0, self.ACT_1: act_1,
                                                           self.ACT_2: act_2,
                                                           self.TARGET_Q_1: target_q_value1_1,
                                                           self.TARGET_Q_2: target_q_value1_2, })

        print('q_value_1_onehot',q_value_1_onehot)
        print('TARGET_Q_1', TARGET_Q_1)
        print('q_value_2_onehot', q_value_2_onehot)
        print('TARGET_Q_2', TARGET_Q_2)
        print('loss:',loss)
        self.sess.run(self.target_updates)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "models_dqn_new_reward/dqn.ckpt")
        return loss

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
    epsilon = train_config['epsilon']
    epsilon_step = train_config['epsilon_step']
    act_1_dim = 0
    batch_size = train_config['batch_size']
    for item in ope_dic:
        if ope_dic[item]['index'] > act_1_dim:
            act_1_dim = ope_dic[item]['index']  # 27

    agent = DQN(act_dim_1=act_1_dim + 1, act_dim_2=column_num, obs_dim=obs_dim, dense_dim=dense_dim,
        lr_q_value=learning_rate, gamma=gamma, epsilon=epsilon, batch_size=batch_size)

    if os.path.exists('reward_list_dqn_1.npy'):
        print('reward_list_dqn_1 exists')
        reward_list = list(np.load('./reward_list_dqn_1.npy',allow_pickle=True))
    else:
        reward_list = []

    if os.path.exists('loss_list_dqn.npy'):
        print('loss exists')
        loss_list = list(np.load('./loss_list_dqn.npy',allow_pickle=True))
    else:
        loss_list = []
    iteration = 0
    if os.path.exists('act_reward_dqn.npy'):
        print('exists')
        act_reward = np.load('./act_reward.npy',allow_pickle=True).item()
    else:
        act_reward = {}
    for i_episode in range(nepisode):
        ep_rwd = 0
        notebook_id = random.choice(notebook_pool)
        print("\033[0;35;40m" + "notebook_id:" + str(notebook_id) + "\033[0m")
        notebook_path = notebook_root + str(notebook_id) + '.ipynb'
        notebook_code = get_code_txt(notebook_path)
        res_line_number = -1
        s_t, len_data = rpc_client_get_origin_state(notebook_id, notebook_code, column_num, ip)
        check_result, model_list = check_model(notebook_id)
        while s_t == 'run failed' or check_result == False:
            notebook_pool.remove(notebook_id)
            notebook_id = random.choice(notebook_pool)
            print("\033[0;34;40m" + "notebook_id:" + str(notebook_id) + "\033[0m")
            notebook_path = notebook_root + str(notebook_id) + '.ipynb'
            notebook_code = get_code_txt(notebook_path)
            s_t, len_data = rpc_client_get_origin_state(notebook_id, notebook_code, column_num, ip)
            check_result, model_list = check_model(notebook_id)

        s_t_p = s_t
        s_t = np.ravel(s_t)
        type_ = np.array([int(np.load('type.npy', allow_pickle=True))])
        if len(s_t) == 1900:
            s_t = np.concatenate((type_, s_t), axis=0)
        if len(s_t) == 1901:
            s_t = np.concatenate((s_t, model_list), axis=0)
        if len(s_t) == 0:
            continue
        while True:
            terminal1 = False
            if int(np.load('type.npy', allow_pickle=True)) != 1:
                terminal1 = True

            action1, action2 = agent.step(s_t, len_data)
            check_res = check_action_by_rule(action1 + 1, action2 + 1, s_t_p, len_data, column_num=column_num)
            s_t_plus_1 = np.zeros([1942])
            if check_res == False:
                reward = -1.0
                terminal = True

            else:
                if action2 == column_num - 1:
                    target_content = {
                        'operation': action1 + 1,
                        'data_object': -1,
                    }
                else:
                    target_content = {
                        'operation': action1 + 1,
                        'data_object': action2,
                    }

                try:
                    s_t, action, reward, terminal, s_t_plus_1, notebook_code, res_line_number, len_data_plus_1 = rpc_client_do_an_action(
                        notebook_id, notebook_code, target_content, column_num, res_line_number, ip)
                except:
                    break
                if s_t == []:
                    break

                if reward == -2:
                    continue

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
            if reward > 0:
                reward *= 1000

            if reward == 0:
                reward = 0.5

            agent.memory.store_transition(s_t, act[0], act[1], reward, s_t_plus_1, terminal)
            agent.memory.save_buffer()
            ep_rwd += reward
            print('iteration',iteration)
            print("\033[0;36;40m" + "reward:" + str(reward) + "\033[0m")
            print("\033[0;36;40m" + "terminal:" + str(terminal) + "\033[0m")
            print("\033[0;36;40m" + "act:" + str(act) + "\033[0m")

            iteration += 1
            if iteration >= 20:
                loss = agent.learn()
                loss_list.append(loss)
                if iteration % epsilon_step == 0:
                    agent.epsilon = max([agent.epsilon * 0.99, 0.001])
                if act[0] not in act_reward:
                    act_reward[act[0]] = []
                act_reward[act[0]].append(ep_rwd)
            if terminal or terminal1:
                print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
                reward_list.append(ep_rwd)
                np.save('reward_list_dqn_1', reward_list)
                np.save('loss_list_dqn', loss_list)
                np.save('act_reward_dqn', act_reward)
                np.save('max_reward_test_3', max_reward)
                np.save('suceed_action_test_3', suceed_action)
                np.save('fail_action_test_3', fail_action)
                np.save('value_loss_test_3', value_loss_list)
                np.save('actor_loss_test_3', actor_loss_list)
                break


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    ip = get_host_ip()
    server_dic = eval(CONFIG.get('server', 'server'))
    notebook_root = server_dic[ip]['npath']
    dataset_root = server_dic[ip]['dpath']

    train(notebook_root, dataset_root, ip)