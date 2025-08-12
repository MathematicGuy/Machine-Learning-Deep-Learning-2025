import numpy as np

# a = np.array([6, 5])
# label = np.array([[-1],[-1],[1],[1],[1]])
# label_name = {-1: "trượt", 1:"đậu"}

# values = np.array([
#     [2, 8],
#     [3, 7],
#     [5, 6],
#     [6, 5],
#     [4, 6],
# ])
# data = np.hstack((values, label))
# print(data)

# k = 3
# dist = {} # list of nearest neighbor
# for i, p in enumerate(data):
#     dist[i] = [np.sqrt((a[0] - p[0])**2 + (a[1] - p[1])**2), p[2]]

# print(dist)
# sorted_dist = sorted(dist.values())[:k]
# print()

# final_class = 0
# for var in sorted_dist:
#     final_class += var[1]
#     print('sorted_dist:', var[0], "-", label_name.get(var[1]))

# if final_class <= 0:
#     print(f'{a} is trượt')
# else:
#     print(f'{a} is đậu')

a = np.array([3, 3.5, 2])
Clus  = np.array([[2,3,1.5], [1,2,1]])
for clu in Clus:
    print(np.square(np.sum((a - clu)**2)))