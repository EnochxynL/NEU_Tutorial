import os
import geopandas as gp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取路网文件
roads = gp.read_file("assets\sydney\sydney_roads_graph.shp")
roads = roads.to_crs(epsg=32756)
roads = roads[roads.geometry.type == 'LineString']
roads['length'] = roads.length
roads = roads.to_crs(epsg=4326)

# 计算每条线的起点和终点
roads['Start_pos'] = roads.geometry.apply(lambda x: x.coords[0])
roads['End_pos'] = roads.geometry.apply(lambda x: x.coords[-1])

# 创建唯一节点列表
s_points = pd.concat([roads.Start_pos, roads.End_pos], ignore_index=True)
s_points = s_points.drop_duplicates().reset_index(drop=True)

# 为每条线添加起点和终点的索引
df_points = pd.DataFrame(s_points, columns=['Start_pos'])
df_points['FNODE_'] = df_points.index
roads = pd.merge(roads, df_points, on='Start_pos', how='inner')
df_points = pd.DataFrame(s_points, columns=['End_pos'])
df_points['TNODE_'] = df_points.index
roads = pd.merge(roads, df_points, on='End_pos', how='inner')

# 准备节点数据
df_points.columns = ['pos', 'osmid']
df_points[['x', 'y']] = df_points['pos'].apply(pd.Series)
df_node_xy = df_points.drop('pos', axis=1)

# 创建 networkx.Graph
G = nx.Graph()

# 添加节点
for index, row in df_node_xy.iterrows():
    G.add_node(row['osmid'], x=row['x'], y=row['y'])

# 添加边
for index, row in roads.iterrows():
    G.add_edge(row['FNODE_'], row['TNODE_'], length=row['length'])

# 显示图形
plt.figure(figsize=(10, 10))

# 提取节点位置
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}

# 绘制图形
nx.draw(G, pos, node_size=1, node_color='blue', edge_color='gray', alpha=0.5)
plt.title('Sydney Road Network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()