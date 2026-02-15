# https://zhuanlan.zhihu.com/p/1919109773162423114
# https://zhuanlan.zhihu.com/p/624429248
# https://gitee.com/oathsign1996/shpfile-convert-to-network/
# https://www.jianshu.com/p/fb85afadde03
# https://networkx.org/documentation/stable/auto_examples/geospatial/extended_description.html
# https://osgeo.cn/pygis/others-pyshp.html

import os
import geopandas as gp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def shapefile_to_networkx(shapefile_path):
    """
    将shapefile转换为networkx.Graph格式
    
    参数:
    shapefile_path: str - shapefile文件路径
    
    返回:
    networkx.Graph - 路网图
    """
    # 读取路网文件
    roads = gp.read_file(shapefile_path)
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
    
    return G

def plot_network(G, title='Road Network'):
    """
    显示networkx.Graph格式的路网图
    
    参数:
    G: networkx.Graph - 路网图
    title: str - 图形标题
    """
    plt.figure(figsize=(10, 10))
    
    # 提取节点位置
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    
    # 绘制图形
    nx.draw(G, pos, node_size=1, node_color='blue', edge_color='gray', alpha=0.5)
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_network_with_map(G, title='Road Network on Map'):
    """
    在地图上显示networkx.Graph格式的路网图
    
    参数:
    G: networkx.Graph - 路网图
    title: str - 图形标题
    
    返回:
    folium.Map - 带有路网的地图实例
    """
    import folium
    
    # 计算地图中心坐标
    nodes = list(G.nodes(data=True))
    if not nodes:
        return folium.Map(location=[-33.8688, 151.2093], zoom_start=12)
    
    lats = [node[1]['y'] for node in nodes]
    lngs = [node[1]['x'] for node in nodes]
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)
    
    # 创建地图实例
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, title=title)
    
    # 添加路网边
    for u, v, data in G.edges(data=True):
        # 获取边的起点和终点坐标
        if 'x' in G.nodes[u] and 'y' in G.nodes[u] and 'x' in G.nodes[v] and 'y' in G.nodes[v]:
            # 创建边的坐标列表
            edge_coords = [
                (G.nodes[u]['y'], G.nodes[u]['x']),
                (G.nodes[v]['y'], G.nodes[v]['x'])
            ]
            # 添加边到地图
            folium.PolyLine(
                edge_coords, 
                tooltip=f"Length: {data.get('length', 'N/A'):.2f}m",
                color='blue',
                weight=2,
                opacity=0.5
            ).add_to(m)
    
    # 添加节点标记（可选，仅添加少量节点以避免地图混乱）
    sample_nodes = nodes[:100]  # 仅添加前100个节点
    for node_id, node_data in sample_nodes:
        if 'x' in node_data and 'y' in node_data:
            folium.Circle(
                location=[node_data['y'], node_data['x']],
                radius=5,
                color='#3186cc',
                fill=True,
                fill_color='#3186cc',
                fill_opacity=0.2
            ).add_to(m)
    
    return m

def plot_path_with_map(G, path, title='Path on Map'):
    """
    在地图上显示networkx.Graph格式的特定路径
    
    参数:
    G: networkx.Graph - 路网图
    path: list - 路径节点ID列表
    title: str - 图形标题
    
    返回:
    folium.Map - 带有路径的地图实例
    """
    import folium
    
    # 检查路径是否为空
    if not path:
        return folium.Map(location=[-33.8688, 151.2093], zoom_start=12, title=title)
    
    # 计算地图中心（基于路径节点）
    path_nodes = []
    for node_id in path:
        if node_id in G.nodes and 'x' in G.nodes[node_id] and 'y' in G.nodes[node_id]:
            path_nodes.append((G.nodes[node_id]['y'], G.nodes[node_id]['x']))
    
    if not path_nodes:
        return folium.Map(location=[-33.8688, 151.2093], zoom_start=12, title=title)
    
    # 计算中心坐标
    lats = [coord[0] for coord in path_nodes]
    lngs = [coord[1] for coord in path_nodes]
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)
    
    # 创建地图实例
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, title=title)
    
    # 添加路径到地图
    folium.PolyLine(
        path_nodes, 
        tooltip="Path",
        color='red',
        weight=4,
        opacity=0.8
    ).add_to(m)
    
    # 添加路径节点标记
    for i, (lat, lng) in enumerate(path_nodes):
        folium.Circle(
            location=[lat, lng],
            radius=8,
            color='#3186cc' if i == 0 else '#cc3131' if i == len(path_nodes) - 1 else '#31cc51',
            fill=True,
            fill_color='#3186cc' if i == 0 else '#cc3131' if i == len(path_nodes) - 1 else '#31cc51',
            fill_opacity=0.6,
            popup=f"Node {i}: ({lat:.6f}, {lng:.6f})"
        ).add_to(m)
    
    return m

if __name__ == '__main__':
    # 执行主程序
    shapefile_path = "assets\sydney\sydney_roads_graph.shp"
    G = shapefile_to_networkx(shapefile_path)
    plot_network(G, title='Sydney Road Network')
    
    # 使用folium在地图上显示路网
    print("\nGenerating map visualization...")
    print("Note: folium may require network access to display maps properly")
    map_obj = plot_network_with_map(G, title='Sydney Road Network on Map')
    
    # 保存地图为HTML文件
    map_html_path = "data\sydney_road_network_map.html"
    map_obj.save(map_html_path)
    print(f"\nMap visualization saved to: {map_html_path}")
    print("You can open this file in a web browser to view the interactive map")