import random
from tqdm import tqdm
import time

from base_station import BaseStation
from base_station_choose_strategy_random import BSChooseRandomStrategy
from client import Client
from edge_server import EdgeServer
from graph import drawGraph
from greedy_controller import GreedyController
from ppo_controller import PPOController
from random_controller import RandomController

factor = 4
hidden_dim = 256
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.98
lmbda = 0.95
eps = 0.2
epochs = 10


bss = []
clients = []
edge_servers = []
for i in range(factor):  # 生成基站
    controller = PPOController(hidden_dim, actor_lr, critic_lr, gamma, lmbda, eps, epochs)
    bss.append(
        BaseStation((random.uniform(20, 80), random.uniform(20, 80)), random.uniform(30, 50), controller))
for i in range(0, len(bss)):  # 连接基站
    for j in range(i + 1, len(bss)):
        if random.random() < 0.3:
            bss[i].connect(bss[j])
for i in range(factor * 4):  # 边缘服务器
    edge_server = EdgeServer(random.uniform(20, 50), random.uniform(5000, 15000), 0.001,
                             (random.uniform(0, 100), random.uniform(0, 100)))
    edge_servers.append(edge_server)
    for bs in bss:  # 边缘服务器与基站相连
        if random.random() < 0.4:
            bs.register(edge_server)
for bs in bss:
    bs.init()


def clear():
    for bs in bss:
        bs.clear()
    for edge_server in edge_servers:
        edge_server.clear()


controller = ['ppo', 'random', 'greedy']

# 所有的基站和边缘服务器已经固定了，接下来采用不同的策略进行模拟，每种策略每个时隙的client生成和task生成都是随机的
for choice in controller:
    if choice == 'random':
        for bs in bss:
            bs.reset_controller(RandomController())
    elif choice == 'greedy':
        for bs in bss:
            bs.reset_controller(GreedyController())
    all_task_count = 0
    all_compute_delay = 0.0
    all_transmission_delay = 0.0
    all_energy_consumption = 0.0
    for item in tqdm(range(10), desc="Processing"):
        clear()
        clients = []
        for i in range(factor * 8):  # 生成客户端
            clients.append(Client((random.uniform(0, 100), random.uniform(0, 100)), BSChooseRandomStrategy()))
        for client in clients:  # 客户端产生任务并发送
            client.scan_all_base_station(bss)
            client.generate_task_and_send(2)

        while True:  # 基站调度任务
            for bs in bss:
                bs.schedule()
            isFinish = True
            for bs in bss:
                if len(bs.task_queue) > 0:
                    isFinish = False
                    break
            if isFinish:
                break

        for edge_server in edge_servers:  # 边缘服务器计算延迟、能耗
            edge_server.update_compute_delay_and_energy_consumption_for_all_tasks()

        for i in range(len(bss)):  # 统计数据
            bs = bss[i]
            bs.learn()
            compute_delay = 0.0
            transmission_delay = 0.0
            energy_consumption = 0.0
            n = len(bs.task_record)
            all_task_count += n
            for task in bs.task_record:
                compute_delay += task.compute_delay
                transmission_delay += task.transmission_delay
                energy_consumption += task.energy_consumption
            all_compute_delay += compute_delay
            all_transmission_delay += transmission_delay
            all_energy_consumption += energy_consumption

    print("================================" + choice + "================================")
    print("avg_compute_delay=" + str(all_compute_delay / all_task_count))
    print("avg_transmission_delay=" + str(all_transmission_delay / all_task_count))
    print("avg_energy_consumption=" + str(all_energy_consumption / all_task_count))

# all_task_count = 0
# all_compute_delay = 0
# all_transmission_delay = 0
# all_energy_consumption = 0
#
# for i in range(len(bss)):  # 统计数据
#     bs = bss[i]
#     compute_delay = 0.0
#     transmission_delay = 0.0
#     energy_consumption = 0.0
#     n = len(bs.task_record)
#     all_task_count += n
#     for task in bs.task_record:
#         compute_delay += task.compute_delay
#         transmission_delay += task.transmission_delay
#         energy_consumption += task.energy_consumption
#     all_compute_delay += compute_delay
#     all_transmission_delay += transmission_delay
#     all_energy_consumption += energy_consumption
#     print("bs#" + str(i) + " compute_delay=" + str(compute_delay / n))
#     print("bs#" + str(i) + " transmission_delay=" + str(transmission_delay / n))
#     print("bs#" + str(i) + " energy_consumption=" + str(energy_consumption / n))
#
# print("all_task_count=" + str(all_task_count))
# print("all_compute_delay=" + str(all_compute_delay))
# print("all_transmission_delay=" + str(all_transmission_delay))
# print("all_energy_consumption=" + str(all_energy_consumption))
#
# for bs in bss:
#     print(len(bs.task_record))
#
# drawGraph(clients, bss, edge_servers)
