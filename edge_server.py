class EdgeServer:
    def __init__(self, f, u, k, position):
        self.f = f  # cpu 速度，MIPS
        self.u = u  # gpu 速度，亿次浮点运算
        self.k = k  # 能耗系数
        self.position = position
        self.tasks = []
        self.parallelism = 0  # 并行数，用来衡量负载情况
        self.IO_conflict_factor = 0.1  # IO冲突因子

    def update_compute_delay_and_energy_consumption_for_all_tasks(self):
        if self.parallelism > 0:
            cpu_speed = self.f / self.parallelism
            gpu_speed = self.u / self.parallelism
            for task in self.tasks:
                task.compute_delay = (task.cpu_length / cpu_speed) * (
                        1 + self.IO_conflict_factor) ** self.parallelism + (task.gpu_length / gpu_speed)
                task.energy_consumption = self.k * task.cpu_length * cpu_speed * cpu_speed

    def clear(self):
        self.tasks = []
        self.parallelism = 0

    def avg_f(self):
        if self.parallelism == 0:
            return self.f
        else:
            return self.f / self.parallelism

    def avg_u(self):
        if self.parallelism == 0:
            return self.u
        else:
            return self.u / self.parallelism
