import random

from controller import Controller


class GreedyController(Controller):
    def __init__(self):
        pass

    def schedule(self, task, bss, edge_servers):
        fast_edge_server = edge_servers[0]
        for edge_server in edge_servers:
            if edge_server.parallelism == 0:
                parallelism = 1
            else:
                parallelism = edge_server.parallelism
            if fast_edge_server.parallelism == 0:
                fast_parallelism = 1
            else:
                fast_parallelism = fast_edge_server.parallelism
            if edge_server.f / parallelism > fast_edge_server.f / fast_parallelism:
                fast_edge_server = edge_server
        return "es", fast_edge_server
