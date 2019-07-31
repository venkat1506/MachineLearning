import numpy as np
import pylab as plt
import networkx as nx

# Initializing points
points_list = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), (2, 3), (2, 7)]

goal = 7
mapping = {0: 'Start', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7-Destination'}
G = nx.Graph()
G.add_edges_from(points_list)
pos = nx.spring_layout(G, k=.5, center=points_list[2])
nx.draw_networkx_nodes(G, pos, node_color='g')
nx.draw_networkx_edges(G, pos, edge_color='b')
nx.draw_networkx_labels(G, pos)
plt.show()

NO_OF_POINTS = 8

# Inititlaizing R Matrix
R = np.matrix(np.ones(shape=(NO_OF_POINTS, NO_OF_POINTS)))
R *= -1

for point in points_list:
    print(point)
if point[1] == goal:
    R[point] = 150
else:
    R[point] = 0

if point[0] == goal:
    R[point[::-1]] = 150
else:
    # reverse of point
    R[point[::-1]] = 0


R[goal,goal]= 150
print(R)

Q = np.matrix(np.zeros([NO_OF_POINTS, NO_OF_POINTS]))

# The learning parameter
gamma = 0.8

initial_state = 1

