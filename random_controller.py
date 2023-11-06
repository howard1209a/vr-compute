import random

from controller import Controller


class RandomController(Controller):
    def __init__(self):
        pass

    def schedule(self, task, bss, edge_servers):
        if task.loopback > 0 and len(bss) > 0:
            if random.random() < 0.5:
                task.loopback -= 1
                return "bs", random.choice(bss)
            else:
                return "es", random.choice(edge_servers)
        else:
            return "es", random.choice(edge_servers)
