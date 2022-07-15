import numpy as np
import open3d as o3d


def display_pcd(file_path):
    show_pcd(read_pcd(file_path))

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)

def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def show_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def show_pcds(pcd_np_arrs):
    pcd_list = []
    colors = [[0,1,1], [0,0.2,1], [0.6,0.2,0.9]]
    for idx, pcd_np in enumerate(pcd_np_arrs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.paint_uniform_color(colors[idx])
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list) # , width=3840, height=2160