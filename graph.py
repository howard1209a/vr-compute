import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def drawGraph(clients, bss, edge_servers):
    node_types = []
    adjacency_matrix = [[0 for _ in range(len(clients) + len(bss) + len(edge_servers))] for _ in
                        range(len(clients) + len(bss) + len(edge_servers))]
    node_positions = []
    for i in range(len(clients)):
        node_types.append(1)
    for i in range(len(bss)):
        node_types.append(2)
    for i in range(len(edge_servers)):
        node_types.append(3)
    for client in clients:
        for bs in client.optional_base_station:
            adjacency_matrix[clients.index(client)][len(clients) + bss.index(bs)] = 1
    for bs in bss:
        for edge_server in bs.edge_servers:
            adjacency_matrix[len(clients) + bss.index(bs)][
                len(clients) + len(bss) + edge_servers.index(edge_server)] = 1

    for client in clients:
        node_positions.append(client.position)
    for bs in bss:
        node_positions.append(bs.position)
    for edge_server in edge_servers:
        node_positions.append(edge_server.position)

    # 创建一个空图
    G = nx.Graph()

    # 获取邻接矩阵的行数和列数，以确定节点数
    num_nodes = len(adjacency_matrix)

    # 添加节点和节点属性
    for i in range(num_nodes):
        node_type = node_types[i]
        position = node_positions[i]
        G.add_node(i, type=node_type, pos=position)

    # 添加边
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 绘制图，根据节点属性设置节点的颜色
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [node_type for _, node_type in nx.get_node_attributes(G, 'type').items()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, cmap=plt.cm.get_cmap('tab10', 3))
    plt.title("Generated Graph with Node Types and Positions")
    plt.show()




