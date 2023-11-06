class Task:
    # 每个任务默认是一个0.5s的4k视频渲染+编码，编码格式采用h.264，编码比特率在10到20 Mbps之间均匀分布，60帧
    # 数据量：5-10兆比特 cpu长度：500万-2000万 gpu浮点运算次数：10兆次到90兆次 优先级：1-5
    def __init__(self, data_size, cpu_length, gpu_length, priority):
        self.compute_delay = 0
        self.transmission_delay = 0
        self.energy_consumption = 0
        self.data_size = data_size  # M比特
        self.cpu_length = cpu_length  # M指令周期数
        self.gpu_length = gpu_length  # Mflop
        self.priority = priority  # 优先级
        self.loopback = 1  # 还可以在基站间转移几次
