import random

from task import Task


class Client:
    def __init__(self, position, bs_choose_strategy):
        self.position = position
        self.optional_base_station = []
        self.bs_choose_strategy = bs_choose_strategy
        self.tasks = []

    def scan_all_base_station(self, bss):
        for bs in bss:
            if (bs.position[0] - self.position[0]) ** 2 + (bs.position[1] - self.position[1]) ** 2 < bs.radius ** 2:
                self.optional_base_station.append(bs)

    def generate_task_and_send(self, n):
        for i in range(n):
            task = Task(random.uniform(5, 10), random.uniform(5, 20),
                        random.uniform(10, 90), random.randint(1, 5))
            bs = self.bs_choose_strategy.choose(self.optional_base_station, task)
            if bs is not None:
                success = self.send(task, bs)
                if success:
                    self.tasks.append(Task(random.uniform(5, 10), random.uniform(5, 20),
                                           random.uniform(10, 90), random.randint(1, 5)))

    def send(self, task, bs):
        return bs.receive(task, "client")  # 如果基站接收任务就返回true，如果基站因为优先队列满了而拒绝任务，则返回false
