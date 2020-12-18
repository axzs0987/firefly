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
import pprint
class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act_1, self.ep_act_2, self.ep_rwd = [], [], [], []

    def store_transition(self, obs0, act1, act2, rwd):
        self.ep_obs.append(obs0)
        self.ep_act_1.append(act1)
        self.ep_act_2.append(act2)
        self.ep_rwd.append(rwd)

    def covert_to_array(self):
        array_obs = np.vstack(self.ep_obs)
        array_act_1 = np.array(self.ep_act_1)
        array_act_2 = np.array(self.ep_act_2)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act_1, array_act_2, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act_1, self.ep_act_2, self.ep_rwd = [], [], [], []
    #
    # def save_buffer(self):
    #     np.save('ep_obs_A2C.npy', self.ep_obs)
    #     np.save('ep_act_1_A2C.npy', self.ep_act_1)
    #     np.save('ep_rwd_A2C.npy', self.ep_rwd)
    #     np.save('ep_act_2_A2C.npy', self.ep_act_2)

class ActorNetwork(object):
    def __init__(self, act_1_dim, act_2_dim, dense_dim, name):
        self.act_1_dim = act_1_dim
        self.act_2_dim = act_2_dim
        self.dense_dim = dense_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 1942,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h2 = tf.layers.dense(h1, 1942,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h3 = tf.layers.dense(h2, 971,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h4 = tf.layers.dense(h3, 971,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h5 = tf.layers.dense(h4, 485,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h6 = tf.layers.dense(h5, 485,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h7 = tf.layers.dense(h6, 242,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h8 = tf.layers.dense(h7, 121,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器

            act_1_prob = tf.layers.dense(h8, self.act_1_dim, None,
                                         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            act_2_prob = tf.layers.dense(h8, self.act_2_dim, None,
                                         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return act_1_prob,act_2_prob

    def choose_action_1(self, obs, reuse=False):
        act_1_prob, act_2_prob = self.step(obs, reuse)
        all_act_1_prob = tf.nn.softmax(act_1_prob, name='act_prob')  # use softmax to convert to probability
        return all_act_1_prob, act_2_prob

    def get_cross_entropy(self, obs, act_1, act_2, reuse=True):
        act_1_prob, act_2_prob = self.step(obs, reuse)
        neglogp_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_1_prob, labels=act_1) # 计算act_prob和act的稀疏softmax 交叉熵
        neglogp_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_2_prob,labels=act_2)  # 计算act_prob和act的稀疏softmax 交叉熵
        return neglogp_1,neglogp_2

class ValueNetwork(object):
    def __init__(self, act_1_dim, act_2_dim, dense_dim, name):
        self.act_1_dim = act_1_dim
        self.act_2_dim = act_2_dim
        self.dense_dim = dense_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 1942,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h2 = tf.layers.dense(h1, 1942,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h3 = tf.layers.dense(h2, 971,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h4 = tf.layers.dense(h3, 971,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h5 = tf.layers.dense(h4, 485,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h6 = tf.layers.dense(h5, 485,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h7 = tf.layers.dense(h6, 242,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            h8 = tf.layers.dense(h7, 121,  # obs:1*4, units:10 输出的维度大小，改变inputs的最后一维。dense就是全连层
                                 tf.nn.tanh,  # 激活函数
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))  # 卷积核的初始化器
            value = tf.layers.dense(h8, 1, None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return value

    def get_value(self, obs, reuse=False):
        q_1_value = self.step(obs, reuse)
        return q_1_value

class ActorCritic:
    def __init__(self, act_dim_1, act_dim_2, obs_dim, dense_dim, lr_actor, lr_value, gamma):
        self.act_dim_1 = act_dim_1
        self.act_dim_2 = act_dim_2
        self.dense_dim = dense_dim
        self.obs_dim = obs_dim
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma

        self.OBS = tf.placeholder(tf.float32, [None, self.obs_dim], name="observation")
        self.ACT_1 = tf.placeholder(tf.int32, [None], name="action")
        self.ACT_2 = tf.placeholder(tf.int32, [None], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")
        self.act_prob_place_2 = tf.placeholder(tf.float32, [None, act_dim_2], name="act_place_2")
        self.act_prob_place_1 = tf.placeholder(tf.float32, [None, act_dim_1], name="act_place_1")

        actor = ActorNetwork(self.act_dim_1, self.act_dim_2, self.dense_dim,'actor')
        critic = ValueNetwork(self.act_dim_1, self.act_dim_2, self.dense_dim,'critic1')
        critic1 = ValueNetwork(self.act_dim_1, self.act_dim_2, self.dense_dim, 'critic2')
        self.memory = Memory()

        self.act_prob = actor.choose_action_1(self.OBS)
        self.act_prob_2 = tf.nn.softmax(self.act_prob_place_2, name='act_prob_2')
        self.act_prob_1 = tf.nn.softmax(self.act_prob_place_1, name='act_prob_1')

        self.cross_entropy = actor.get_cross_entropy(self.OBS, self.ACT_1, self.ACT_2)
        cross_entropy1, cross_entropy2 = self.cross_entropy
        # print(cross_entropy1.shape)
        self.value = critic.get_value(self.OBS)
        self.advantage = self.Q_VAL - self.value # 激励值，累计奖励算出来的价值-估计价值
        # print(self.advantage_1.shape)
        self.advantage1 = self.Q_VAL - critic1.get_value(self.OBS)
        self.actor_loss = tf.reduce_mean(self.cross_entropy[0] * self.advantage1) + tf.reduce_mean(self.cross_entropy[1] * self.advantage1) # 用激励值代替累计奖励，更新策略函数
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss)

        self.value_loss = tf.reduce_mean(tf.square(self.advantage)) # 激励值当作价值函数的误差
        self.value_train_op = tf.train.AdamOptimizer(self.lr_value).minimize(self.value_loss)

        self.critic_value_params = tf.global_variables('critic1')
        self.critic1_value_params = tf.global_variables('critic2')
        # print('critic1:', self.critic_value_params)
        # print('critic2:', self.critic1_value_params)
        self.target_updates = [tf.assign(tq, q) for tq, q in zip(self.critic1_value_params, self.critic_value_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        ckpt_state = tf.train.get_checkpoint_state('./d3m4/')
        self.sess = tf.Session()
        if ckpt_state:
            print('restore')
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_state.model_checkpoint_path)
        else:
            print('create new')
            self.sess.run(tf.global_variables_initializer())
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
        return [act_2_prob]
    def step(self,obs, obs_matrix, lendata):
        self.is_choiced = []
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        # print('obs:', obs)

        self.prob_weights_1, self.prob_weights_2 = self.sess.run(self.act_prob, feed_dict={self.OBS: obs}) # action,得到每一个action的概率
        print('prob weights 1', self.prob_weights_1)
        random_chose = False
        if os.path.exists('max_action_d3m4.npy'):
            max_action_d3m4 = list(np.load('max_action_d3m4.npy',allow_pickle=True))
        else:
            max_action_d3m4 = []

        max_action_d3m4.append(np.argmax(self.prob_weights_1.ravel()))
        np.save('max_action_d3m4.npy',max_action_d3m4)
        if np.random.rand(1) < 0.5:
            print('\033[0;35;40mchose action1: random\033[0m')
            action1 = np.random.randint(0, self.act_dim_1)
            check_res = check_action_by_rule_1(action1 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
            count = 0
            while check_res == False and count < 27:
                self.prob_weights_1[0][action1] = 0
                self.prob_weights_1 = self.sess.run(self.act_prob_1,
                                                    feed_dict={self.act_prob_place_1: self.prob_weights_1})
                action1 = np.random.randint(0, self.act_dim_1)
                if self.prob_weights_1[0][action1] == 0:
                    continue
                check_res = check_action_by_rule_1(action1 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
                count += 1
            random_chose = True
        else:
            print('\033[0;35;40mchose action1: max\033[0m')
            action1 = np.argmax(self.prob_weights_1) # 按照给出的概率随机选一个action
            check_res = check_action_by_rule_1(action1 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
            count = 0
            while check_res == False and count < 27:
                self.prob_weights_1[0][action1] = 0
                self.prob_weights_1 = self.sess.run(self.act_prob_1,
                                                    feed_dict={self.act_prob_place_1: self.prob_weights_1})
                action1 = np.argmax(self.prob_weights_1)
                if self.prob_weights_1[0][action1] == 0:
                    continue
                check_res = check_action_by_rule_1(action1 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
                count += 1

        # print('self.p1:', self.prob_weights_1)
        # print('self.p2:', self.prob_weights_2)
        self.prob_weights_2 = self.choose_action_2(self.prob_weights_2[0], obs_matrix, lendata, action1)  # action,得到每一个action的概率
        # print('self.p2:', self.prob_weights_2)

        self.prob_weights_2 = self.sess.run(self.act_prob_2, feed_dict={self.act_prob_place_2: self.prob_weights_2}) # action,得到每一个action的概率
        print('prob weights 2', self.prob_weights_2)
        if np.random.rand(1) < 0.5:
            print('\033[0;35;40mchose action2: random\033[0m')
            action2 = np.random.randint(0, self.act_dim_2)
            random_chose = True
            check_res = check_action_by_rule_2(action1 + 1, action2 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
            count = 0
            while check_res == False and count < 100:
                self.prob_weights_2[0][action2] = 0
                self.prob_weights_2 = self.sess.run(self.act_prob_2,
                                                    feed_dict={self.act_prob_place_2: self.prob_weights_2})
                action2 = np.random.randint(0, self.act_dim_2)
                if self.prob_weights_2[0][action2] == 0:
                    continue
                check_res = check_action_by_rule_2(action1 + 1, action2 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
                count += 1
                # print('reruning', action2)
        else:
            print('\033[0;35;40mchose action2: max\033[0m')
            action2 = np.random.choice(range(self.prob_weights_2.shape[1]), p=self.prob_weights_2.ravel()) # 按照给出的概率随机选一个action
            check_res = check_action_by_rule_2(action1 + 1, action2 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
            count = 0
            while check_res == False and count < 100:
                count += 1
                self.prob_weights_2[0][action2] = 0
                self.prob_weights_2 = self.sess.run(self.act_prob_2,
                                                    feed_dict={self.act_prob_place_2: self.prob_weights_2})
                action2 = np.argmax(self.prob_weights_2.shape[1])  # 按照给出的概率随机选一个action
                check_res = check_action_by_rule_2(action1 + 1, action2 + 1, obs_matrix, lendata, column_num=self.act_dim_2)
                # print('reruning', action2)
        # print_item = self.prob_weights_1.ravel()
        # for index, item in enumerate(print_item):
        #     print(index, str(item))
        # if np.random.rand(1) < 0.5:
        #     print('\033[0;35;40mchose action: random\033[0m')
        #     action1 = np.random.randint(0, self.act_dim_1)
        #     action2 = np.random.randint(0, self.act_dim_2)
        # else:
        #     print('\033[0;35;40mchose action: max\033[0m')
        #     action1 = np.random.choice(range(self.prob_weights_1.shape[1]), p=self.prob_weights_1.ravel()) # 按照给出的概率随机选一个action
        #     action2 = np.random.choice(range(self.prob_weights_2.shape[1]), p=self.prob_weights_2.ravel())  # 按照给出的概率随机选一个action
        # # self.is_choiced.append(str(action1)+'+' + str(action2))
        return action1,self.prob_weights_1[0][action1],action2,self.prob_weights_2[0][action2],random_chose # 输入obs，按照策略pai给出相应的执行动作

    def get_value(self,obs):
        value = self.sess.run(self.value, feed_dict={self.OBS: obs})
        return value

    def learn(self, last_value_1, done):
        # print('tf.trainable_variables():',tf.trainable_variables())
        obs, act_1, act_2, rwd = self.memory.covert_to_array()

        q_value = self.compute_q_value(last_value_1, done, rwd) # 累积奖励

        c1,c2 = self.sess.run(self.cross_entropy, {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        print('cross_entropy1:',c1)
        print('cross_entropy2:', c2)
        advantage_1 = self.sess.run(self.advantage,{self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        print('advantage_1',advantage_1)
        print('Q_VAL',q_value)
        value = self.sess.run(self.value,
                                    {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        print('value', value)
        value_loss = self.sess.run(self.value_loss, {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        actor_loss = self.sess.run(self.actor_loss, {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        print('value_loss', value_loss)
        print('actor_loss', actor_loss)

        # if os.path.exists('act_advantage_d3m4.npy'):
        #     act_advantage_dic = np.load('act_advantage_d3m4.npy', allow_pickle=True).item()
        # else:
        act_advantage_dic = []
        for index,item in enumerate(act_1):
            act_advantage_dic.append((item,act_2[index],advantage_1[index], actor_loss))
        # np.save('act_advantage_d3m4', act_advantage_dic)

        self.sess.run(self.actor_train_op, {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        self.sess.run(self.value_train_op, {self.OBS: obs, self.Q_VAL: q_value})
        # critic_value_params = self.sess.run(self.critic_value_params, {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        # critic1_value_params = self.sess.run(self.critic1_value_params,
        #                                     {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        # for tq, q in zip(critic1_value_params, critic_value_params):
        #     print('#######')
        #     print('tq',tq)
        #     print('q', q)
        self.sess.run(self.target_updates, {self.OBS: obs, self.ACT_1: act_1, self.ACT_2: act_2, self.Q_VAL: q_value})
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "d3m4/A2C.ckpt")
        self.memory.reset()
        return value_loss,actor_loss,act_advantage_dic

    def compute_q_value(self, last_value_1, done, rwd):
        q_value_1 = np.zeros_like(rwd)
        value_1 = 0 if done else last_value_1
        for t in reversed(range(0, len(rwd))):
            value_1 = value_1 * self.gamma + rwd[t] # Qst = gamma*Qst+1 + r
            q_value_1[t] = value_1
        return q_value_1[:, np.newaxis] # q_value 是用即时奖励和累计函数算出来的（这个可以理解为累积奖励）



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

def check_action_by_rule_1(action1, s_t, len_data, column_num):
    def get_table_info():
        is_all_num = True
        nhas_nan = True
        is_all_cat = True
        for action2 in range(0, len_data):
            column_type = s_t[action2 - 1][3]
            num_ratio = s_t[action2 - 1][2]
            if column_type == 2:
                is_all_num=False
            else:
                is_all_cat = False
            if num_ratio != 0:
                nhas_nan = False
        return nhas_nan, is_all_num, is_all_cat

    global_nhas_nan, global_is_all_num,global_is_all_cat = get_table_info()
    if action1 in [11,12,13,14,15,16]:
        if global_is_all_num == False:
            return False
    if action1 in [8, 16]:
        if global_nhas_nan == False:
            return False
    if action1 not in [1,5,6,7,9,10,14,17,18,19,20,21]: #这些操作对num列会报错
        if global_is_all_num == True:
            return False
    if action1 not in [2,6,9,10,14,17,18]:
        if global_is_all_cat == True:
            return False
    return True

def check_action_by_rule_2(action1, action2, s_t, len_data,column_num):
    def get_table_info():
        is_all_num = True
        has_nan = True
        for action2 in range(0, len_data):
            column_type = s_t[action2 - 1][3]
            num_ratio = s_t[action2 - 1][2]
            if column_type == 2:
                is_all_num=False
            if num_ratio != 0:
                has_nan = False
        return has_nan, is_all_num
    global_nhas_nan, global_is_all_num = get_table_info()
    if action2 != column_num and action2 >= len_data:
        return False
    is_column = True
    if action2 == column_num:
        is_column = False
    if is_column == False:
        if action1 in [1,2,3,4,5,6,8,9,11,12,13,14,15,16,18,20]:
            if global_nhas_nan == False :
                if action1 in [5, 8, 9, 16, 19, 20, 21]:
                    return False
            if global_is_all_num == False:
                if action1 in [11,12,13,14,15,16]:
                    return False
            return True
        else:
            return False
    else:
        # print('s_t.shape', np.array(s_t).shape)
        # print(action2)
        # print('s_t',s_t)
        column_type = s_t[action2-1][3]
        num_ratio = s_t[action2-1][2]
        if column_type == 1: # numeric
            if action1 in [1,5,6,7,9,10,14,17,18,19,20,21]:
                if num_ratio != 0:
                    if action1 in [5, 8, 9, 16, 19, 20, 21]:
                        return False
                return True
            else:
                return False
        if column_type == 2: # category
            if action1 in [2,6,9,10,14,17,18]:
                if num_ratio != 0:
                    if action1 in [5, 8, 9, 16, 19, 20, 21]:
                        return False
                return True
            else:
                return False


def compare_state(s_t, s_t_plus_1):
    if len(s_t) != len(s_t_plus_1):
        return False
    return (s_t == s_t_plus_1).all()

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

        in_ope = []
        sql = 'select distinct notebook_id from operator'
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            if int(row[0]) not in in_ope:
                in_ope.append(int(row[0]))

        sql = 'select pair.nid from pair,dataset where pair.did=dataset.id and dataset.id != 3130 and dataset.server_ip = \'' + ip + '\''
        cursor.execute(sql)
        sql_res = cursor.fetchall()
        for row in sql_res:
            if int(row[0]) not in in_result:
                continue
            if int(row[0]) not in in_notebook:
                continue
            if int(row[0]) in in_ope:
                continue
            if int(row[0]) not in notebook_pool:
                notebook_pool.append(int(row[0]))
        print('ntoebook_pool:',len(notebook_pool))
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
            act_1_dim = ope_dic[item]['index']  # 27

    agent = ActorCritic(act_dim_1=act_1_dim + 1, act_dim_2=column_num, obs_dim=obs_dim,dense_dim=dense_dim,
                        lr_actor=learning_rate, lr_value=learning_rate, gamma=gamma)


    if os.path.exists('reward_list_d3m4.npy'):
        print('reward_list_A2C_1 exists')
        reward_list = list(np.load('./reward_list_d3m4.npy',allow_pickle=True))
    else:
        reward_list = []

    if os.path.exists('reward_list_r_d3m4.npy'):
        print('reward_list_A2C_r_1 exists')
        reward_list_r = list(np.load('./reward_list_r_d3m4.npy',allow_pickle=True))
    else:
        reward_list_r = []

    if os.path.exists('./act_reward_d3m4.npy'):
        print('exists')
        act_reward = np.load('./act_reward_d3m4.npy',allow_pickle=True).item()
    else:
        act_reward = {}

    if os.path.exists('max_reward_d3m4.npy'):
        print('max_reward_A2C_1 exists')
        max_reward = list(np.load('./max_reward_d3m4.npy',allow_pickle=True))
    else:
        max_reward = []
    if os.path.exists('value_loss_d3m4.npy'):
        print('value_loss_d3m4 exists')
        value_loss_list = list(np.load('./value_loss_d3m4.npy',allow_pickle=True))
    else:
        value_loss_list = []
    if os.path.exists('actor_loss_d3m4.npy'):
        print('actor_loss_d3m4 exists')
        actor_loss_list = list(np.load('./actor_loss_d3m4.npy',allow_pickle=True))
    else:
        actor_loss_list = []

    if os.path.exists('suceed_action_d3m4.npy'):
        print('suceed_action exists')
        suceed_action = list(np.load('./suceed_action_d3m4.npy',allow_pickle=True))
    else:
        suceed_action = []
    if os.path.exists('fail_action_d3m4.npy'):
        print('fail_action exists')
        fail_action = list(np.load('./fail_action_d3m4.npy',allow_pickle=True))
    else:
        fail_action = []


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

        temp_act_reward_dic = []
        chosed_list = []
        while True:
            # act, _ = agent.step(obs0)  # act是通过概率选择的一个动作，_是所有动作的Q值
            terminal1 = False
            if int(np.load('type.npy', allow_pickle=True)) != 1:
                terminal1 = True

            action1, act_prob1,action2, act_prob2,random_chose = agent.step(s_t, s_t_p, len_data)
            check_res_1 = check_action_by_rule_1(action1 + 1, s_t_p, len_data, column_num=column_num)
            check_res_2 = check_action_by_rule_2(action1 + 1, action2 + 1, s_t_p, len_data, column_num=column_num)
            s_t_plus_1 = np.zeros([1942]) # failed s_t_plus_1
            try_time = 0
            while (str(action1)+'+'+str(action2) in chosed_list or str(action1)+'+'+str(100) in chosed_list) or (check_res_1 and check_res_2) == False:
                action1, act_prob1, action2, act_prob2, random_chose = agent.step(s_t, s_t_p, len_data)
                check_res_1 = check_action_by_rule_1(action1 + 1, s_t_p, len_data, column_num=column_num)
                check_res_2 = check_action_by_rule_2(action1 + 1, action2 + 1, s_t_p, len_data, column_num=column_num)
                try_time += 1
            chosed_list.append(str(action1) +'+'+ str(action2))
            print('check_res', (check_res_1 and check_res_2))
            if check_res_1 and check_res_2  == False:
                reward = -3.0
                terminal = True
                compare = False
            else:
                if action2 == column_num - 1:
                    target_content = {
                        'operation': action1 + 1,
                        'data_object': -1,
                    }
                    compare = compare_state(s_t, s_t_plus_1)
                    if compare == True:
                        print('action1:', action1)
                else:
                    target_content = {
                        'operation': action1 + 1,
                        'data_object': action2,
                    }

                # try:
                s_t, action, reward, terminal, s_t_plus_1, notebook_code, res_line_number, len_data_plus_1 = rpc_client_do_an_action(
                        notebook_id, notebook_code, target_content, column_num, res_line_number, ip)
                # except Exception as e:
                #     print(e)
                #     break
                if s_t == []:
                    print('st is nulls')
                    break

                if reward == -2:
                    print('??')
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
                type_1 = np.array([int(np.load('type_1.npy', allow_pickle=True))])
                if len(s_t_plus_1) == 1900:
                    s_t_plus_1 = np.concatenate((type_1, s_t_plus_1), axis=0)
                if len(s_t_plus_1) == 1901:
                    s_t_plus_1 = np.concatenate((s_t_plus_1, model_list), axis=0)

                compare = compare_state(s_t, s_t_plus_1)
                if compare == True:
                    print('action1:', action1)
                    reward = -2

                # s_t = s_t_plus_1
                len_data = len_data_plus_1
                if int(np.load('type_1.npy', allow_pickle=True)) != 1:
                    terminal1 = True
            act = (action1, action2)
            # if str(action1) not in act_reward.keys():
            #     act_reward[str(action1)] = []
            # if str(action1) not in temp_act_reward_dic.keys():
            #     temp_act_reward_dic[str(action1)] = []

            # act_reward[str(action1)].append((act_prob1, action2, act_prob2, reward, 'not changed:'+str(compare), 'terminal:'+str(terminal), 'is random chosed:'+str(random_chose)))
            temp_act_reward_dic.append((action1, action2, act_prob1, act_prob2, 'reward:'+str(reward), 'not changed:' + str(compare),
                                             'terminal:' + str(terminal), 'is random chosed:' + str(random_chose)))
                # print('s_t:', s_t)
                # print('s_t_plus_1:',s_t_plus_1)

            if reward > 0:
                suceed_action.append((notebook_id,act))
                reward *= 1000
            if reward == -1:
                reward=-3
            if reward < 0 and reward != -3 and reward != -2:
                fail_action.append((notebook_id,act))
            if reward == 0:
                reward = 0.5
            agent.memory.store_transition(s_t, act[0], act[1], reward)
            ep_rwd += reward
            reward_list_r.append(reward)
            if reward==-2:
                s_t_plus_1 = np.zeros([1942])
                terminal = True
            if len(s_t_plus_1) == 0:
                reward = -2
                terminal = True
                s_t_plus_1 = np.zeros([1942])
            # print('s_t_plus_1:',s_t_plus_1)
            s_t = s_t_plus_1
            print("\033[0;36;40m" + "reward:" + str(reward) + "\033[0m")
            print("\033[0;36;40m" + "terminal:" + str(terminal) + "\033[0m")
            print("\033[0;36;40m" + "act:" + str(act) + "\033[0m")
            if random_chose == False:
                max_reward.append(reward)
            if terminal or terminal1:
                last_value = agent.get_value([s_t])  # last_value是执行完一轮，最后一个操作的所有动作的Q值
                value_loss, actor_loss, act_advantage_dic = agent.learn(last_value, terminal)
                for index,item in enumerate(temp_act_reward_dic):
                    action1 = item[0]
                    if str(action1) not in act_reward.keys():
                        act_reward[str(action1)] = []
                    act_reward[str(action1)].append((item[2],item[1],item[3],item[4],'advantage:'+str(act_advantage_dic[index][2]),'actor_loss:'+str(act_advantage_dic[index][3]),item[5],item[6],item[7]))
                value_loss_list.append(value_loss)
                actor_loss_list.append(actor_loss)
                print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
                reward_list.append(ep_rwd)
                np.save('reward_list_d3m4', reward_list)
                np.save('reward_list_r_d3m4', reward_list_r)
                np.save('act_reward_d3m4', act_reward)
                np.save('max_reward_d3m4', max_reward)
                np.save('suceed_action_d3m4', suceed_action)
                np.save('fail_action_d3m4', fail_action)
                np.save('value_loss_d3m4', value_loss_list)
                np.save('actor_loss_d3m4', actor_loss_list)
                break

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    ip = get_host_ip()
    server_dic = eval(CONFIG.get('server', 'server'))
    notebook_root = server_dic[ip]['npath']
    dataset_root = server_dic[ip]['dpath']

    train(notebook_root, dataset_root, ip)