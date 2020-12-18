import numpy as np
import configparser
import matplotlib.pyplot as plt

CONFIG = configparser.ConfigParser()
CONFIG.read('config.ini')
act_reward = np.load('/Users/baiping/fsdownload/act_reward_d3m4.npy',allow_pickle=True).item()
suceed_action = np.load('/Users/baiping/fsdownload/suceed_action_d3m4.npy',allow_pickle=True)
fail_action = np.load('/Users/baiping/fsdownload/fail_action_d3m4.npy',allow_pickle=True)
value_loss = np.load('/Users/baiping/fsdownload/value_loss_d3m4.npy',allow_pickle=True)
actor_loss = np.load('/Users/baiping/fsdownload/actor_loss_d3m4.npy',allow_pickle=True)
reward_list_r = np.load('/Users/baiping/fsdownload/reward_list_r_d3m4.npy',allow_pickle=True)
reward_list = np.load('/Users/baiping/fsdownload/reward_list_d3m4.npy',allow_pickle=True)
max_reward_list = np.load('/Users/baiping/fsdownload/max_reward_d3m4.npy',allow_pickle=True)
max_action_list = np.load('/Users/baiping/fsdownload/max_action_d3m4.npy',allow_pickle=True)
def show_act_reward():
    operator_dic = eval(CONFIG.get('operators1', 'operations1'))
    for i in act_reward:
        for item in operator_dic:
            if int(i)+1 == int(operator_dic[item]['index']):
                print(i,item)
        for j in act_reward[i]:
            print('    '+str(j))

def show_suceed():
    print(suceed_action)

def show_fail():
    print(fail_action)

def show_value_loss():
    plt.ylim(-10,10000)
    plt.plot(value_loss)
    plt.plot(reward_list)
    plt.show()

def show_actor_loss():
    plt.plot(actor_loss)
    plt.plot(reward_list)
    # plt.plot(value_loss)
    plt.show()

def show_max_action():
    print(max_action_list)

def show_reward_r():
    # plt.ylim(-2, 5)
    plt.plot(reward_list_r)
    plt.show()

def show_max_reward_list():
    plt.plot(max_reward_list)
    plt.show()

def show_reward():
    # plt.ylim(-2, 5)
    plt.plot(reward_list)
    plt.show()
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    show_act_reward()