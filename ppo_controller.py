import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from controller import Controller
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=0)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPOController(Controller):

    def __init__(self, hidden_dim, actor_lr, critic_lr, gamma, lmbda, eps, epochs):
        # self.device = torch.device("gpu")
        self.device = torch.device("cuda")
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.epochs = epochs  # 一条trajectory用来更新多少轮

    def init(self, bs):
        self.bs = bs
        # 状态是边缘服务器信息+任务队列信息+队头任务信息
        state_dim = len(self.bs.edge_servers) * 3 + self.bs.task_queue_maxlength * 3 + 3
        # 动作是卸载到基站/边缘服务器
        action_dim = len(self.bs.edge_servers) + len(self.bs.bss)
        self.actor = PolicyNet(state_dim, self.hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, self.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr)
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    def schedule(self, task, bss, edge_servers):
        edge_servers_information = []
        tasks_information = []
        for edge_server in self.bs.edge_servers:
            edge_server_info = []
            edge_server_info.append(edge_server.avg_f())
            edge_server_info.append(edge_server.avg_u())
            edge_server_info.append(self.bs.edge_server_data_rates[self.bs.edge_servers.index(edge_server)])
            edge_servers_information.append(edge_server_info)
        for task in self.bs.task_queue:
            task_info = []
            task_info.append(task.cpu_length)
            task_info.append(task.gpu_length)
            task_info.append(task.data_size)
            tasks_information.append(task_info)
        for i in range(self.bs.task_queue_maxlength - len(self.bs.task_queue)):
            task_info = []
            task_info.append(0)
            task_info.append(0)
            task_info.append(0)
            tasks_information.append(task_info)
        task_info = []
        task_info.append(task.cpu_length)
        task_info.append(task.gpu_length)
        task_info.append(task.data_size)
        tasks_information.append(task_info)

        standard_scaler = StandardScaler()  # 按维度标准化
        edge_servers_information = standard_scaler.fit_transform(edge_servers_information)
        tasks_information = standard_scaler.fit_transform(tasks_information)
        flattened_edge_servers_information = edge_servers_information.reshape(1, -1).ravel()  # 摊平
        flattened_tasks_information = tasks_information.reshape(1, -1).ravel()

        state = np.concatenate((flattened_edge_servers_information, flattened_tasks_information))  # 拼接

        probabilities = self.actor(torch.tensor(state, dtype=torch.float32).to(self.device))
        for i in range(len(probabilities)):  # 额外处理一下，防止出现特殊值
            prob = probabilities[i]
            if math.isinf(prob):
                probabilities[i] = 1
            elif math.isnan(prob) or prob < 0:
                probabilities[i] = 0
        # 检查列表中的所有数是否都为0，如果是，将每个数设置为1除以列表的长度
        if all(x == 0 for x in probabilities):  # 额外处理一下，防止出现全0
            probabilities = torch.tensor([1.0 / len(probabilities) for _ in probabilities], dtype=torch.float32)

        action = torch.multinomial(probabilities, 1, replacement=True)  # 采样

        if len(self.transition_dict['states']) > 0:
            self.transition_dict['next_states'].append(state)
        self.transition_dict['states'].append(state)
        self.transition_dict['actions'].append(action.item())
        self.transition_dict['rewards'].append(0)
        if len(self.bs.task_queue) == 0:
            end_state = state.copy()
            end_state[len(end_state) - 1] = 0
            end_state[len(end_state) - 2] = 0
            end_state[len(end_state) - 3] = 0
            self.transition_dict['next_states'].append(end_state)
            self.transition_dict['dones'].append(True)
        else:
            self.transition_dict['dones'].append(False)

        if action < len(self.bs.edge_servers):  # 选择edge_server
            return "es", self.bs.edge_servers[action]
        else:  # 选择bs
            if task.loopback > 0:
                return "bs", self.bs.bss[action - len(self.bs.edge_servers)]
            else:  # task已经不可以再loopback，随机选一个edge_server
                return "es", random.choice(self.bs.edge_servers)

    def update(self):
        if len(self.transition_dict['states']) > 0:  # 如果transition_dict为空就做update了
            # 下面首先填入结束奖励
            if len(self.bs.task_record) != 0:
                all_delay = 0.0
                all_energy_consumption = 0.0
                for finished_task in self.bs.task_record:
                    all_delay += finished_task.compute_delay + finished_task.transmission_delay
                    all_energy_consumption += finished_task.energy_consumption
                avg_delay = all_delay / len(self.bs.task_record)
                avg_energy_consumption = all_energy_consumption / len(self.bs.task_record)
                reward = 1.0 / avg_delay + 1.0 / avg_energy_consumption  # 奖励设置，后续需要加一个权重
            else:  # 如果基站自始至终没有处理任何任务，则它的奖励为0
                reward = 0.0

            self.transition_dict['rewards'][len(self.transition_dict['rewards']) - 1] = reward

            # 下面用trajectory做更新
            states = torch.tensor(self.transition_dict['states'], dtype=torch.float).to(self.device)
            actions = torch.tensor(self.transition_dict['actions']).view(-1, 1).to(self.device)
            rewards = torch.tensor(self.transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(self.transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(self.transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

            for _ in range(self.epochs):
                log_probs = torch.log(self.actor(states).gather(1, actions))
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def clear(self):
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
