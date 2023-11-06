import random

from greedy_controller import GreedyController
from ppo_controller import PPOController


class BaseStation:
    def __init__(self, position, radius, controller):
        self.position = position
        self.radius = radius
        self.task_queue = []
        self.task_queue_maxlength = 10
        self.task_record = []
        self.edge_servers = []
        self.controller = controller
        self.bss = []
        self.edge_server_data_rates = []
        self.bss_data_rates = []

    def receive(self, task, source):
        if len(self.task_queue) == self.task_queue_maxlength:  # 队满则拒绝接收任务
            return False
        if len(self.task_queue) == 0 and source != "client":  # 如果队空则拒绝接收来自其他基站转移的任务
            return False
        if source == "client":
            self.task_record.append(task)
        for i in range(len(self.task_queue)):
            if task.priority >= self.task_queue[i].priority:
                self.task_queue.insert(i, task)
                return True
        self.task_queue.append(task)
        return True

    def connect(self, bs):
        self.bss.append(bs)
        self.bss_data_rates.append(random.uniform(4, 24))  # 20-120Mbps均匀分布
        bs.bss.append(self)
        bs.bss_data_rates.append(random.uniform(4, 24))  # 20-120Mbps均匀分布

    def register(self, edge_server):
        self.edge_servers.append(edge_server)
        self.edge_server_data_rates.append(random.uniform(4, 24))  # 20-120Mbps均匀分布

    def schedule(self):
        if len(self.task_queue) > 0:
            task = self.task_queue.pop(0)
            choice, destination = self.controller.schedule(task, self.bss, self.edge_servers)
            if choice == "bs":  # 发往另一个基站
                if destination.receive(task, "bs"):  # 如果另一个基站成功接收了
                    task.transmission_delay += task.data_size / self.bss_data_rates[self.bss.index(destination)]
                else:  # 如果另一个基站因为队满而拒绝接收，用贪婪策略选择一个边缘服务器
                    _, destination = GreedyController().schedule(task, self.bss, self.edge_servers)
                    task.transmission_delay += task.data_size / self.edge_server_data_rates[
                        self.edge_servers.index(destination)]
                    destination.parallelism += 1
                    destination.tasks.append(task)
            else:  # 发往边缘服务器
                task.transmission_delay += task.data_size / self.edge_server_data_rates[
                    self.edge_servers.index(destination)]
                destination.parallelism += 1
                destination.tasks.append(task)

    def clear(self):
        self.task_queue = []
        self.task_record = []
        if isinstance(self.controller, PPOController):
            self.controller.clear()

    def init(self):
        if isinstance(self.controller, PPOController):
            self.controller.init(self)

    def learn(self):
        if isinstance(self.controller, PPOController):
            self.controller.update()

    def reset_controller(self, controller):
        self.controller = controller
