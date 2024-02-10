from brightest_path_lib.algorithm import AStarSearch, NBAStarSearch
import numpy as np
import napari
from skimage import data
import tifffile
import time


def test_2D_image():
    # testing for 2D
    #twoDImage = data.cells3d()[30, 1]  # brighter image
    # twoDImage = data.cells3d()[30, 0] # darker image
    # start_point = np.array([0,192])
    # goal_point = np.array([198,9])

    #image = tifffile.imread('rr30a_s0_ch2.tif')[30]
    start_point = np.array([243, 292]) # (y,x)
    goal_point = np.array([128, 711]) # (y,x)

    # astar_search = AStarSearch(
    #     image,
    #     start_point,
    #     goal_point)
    # tic = time.perf_counter()
    # result = astar_search.search()
    # toc = time.perf_counter()
    # print(f"result: {result}")
    # print(f"Found brightest path in {toc - tic:0.4f} seconds")
    # print(f"path size: {len(result)}")
    # print(f"Number of nodes viewed: {astar_search.evaluated_nodes}")
    two_dim_image = np.array([[ 4496,  5212,  6863, 10113,  7055],
       [ 4533,  5146,  7555, 10377,  5768],
       [ 4640,  6082,  8452, 10278,  4543],
       [ 5210,  6849, 10010,  8677,  3911],
       [ 5745,  7845, 11113,  7820,  3551]])
    two_dim_start_point = np.array([0,0])
    two_dim_goal_point = np.array([4,4])

    three_dim_image = np.array([[[ 4496,  5212,  6863, 10113,  7055],
        [ 4533,  5146,  7555, 10377,  5768],
        [ 4640,  6082,  8452, 10278,  4543],
        [ 5210,  6849, 10010,  8677,  3911],
        [ 5745,  7845, 11113,  7820,  3551]],

       [[ 8868,  6923,  5690,  6781,  5738],
        [ 7113,  5501,  5216,  4789,  5501],
        [ 5833,  7160,  5928,  5596,  5406],
        [ 6402,  6259,  5501,  4458,  6449],
        [ 6117,  6022,  7160,  7113,  7066]]])
    three_dim_start_point = np.array([0,0,0])
    three_dim_goal_point = np.array([0,4,4])

    # nbastar_search = NBAStarSearch(
    #     image,
    #     start_point,
    #     goal_point)
    # nbastar_search = NBAStarSearch(
    #     two_dim_image,
    #     two_dim_start_point,
    #     two_dim_goal_point)
    three_dim_image = (three_dim_image/np.max(three_dim_image) * 255).astype(np.uint8)
    nbastar_search = NBAStarSearch(
        three_dim_image,
        three_dim_start_point,
        three_dim_goal_point)
    tic = time.perf_counter()
    result = nbastar_search.search()
    toc = time.perf_counter()
    print(f"result: {result}")
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(result)}")
    print(f"Number of nodes viewed: {nbastar_search.evaluated_nodes}")

    # viewer = napari.Viewer()
    # # viewer.add_image(twoDImage[:100, :250], colormap='magma')
    # viewer.add_image(image)
    # viewer.add_points(np.array([start_point, goal_point]), size=10, edge_width=1, face_color="red", edge_color="red")
    # viewer.add_points(result, size=10, edge_width=1, face_color="green", edge_color="green")
    # napari.run()

def test_3D_image():
    image = tifffile.imread('rr30a_s0_ch2.tif')
    start_point = np.array([30, 243, 292]) # (z,y,x)
    goal_point = np.array([30, 221, 434]) # (z,y,x)

    astar_search = AStarSearch(
        image,
        start_point,
        goal_point
    )

    tic = time.perf_counter()
    result = astar_search.search()
    toc = time.perf_counter()
    print(f"Found brightest path in {toc - tic:0.4f} seconds")
    print(f"path size: {len(result)}")

    # nbastar_search = NBAStarSearch(
    #     image,
    #     start_point,
    #     goal_point
    # )
    # tic = time.perf_counter()
    # result = nbastar_search.search()
    # toc = time.perf_counter()
    # print(f"Found brightest path in {toc - tic:0.4f} seconds")
    # print(f"path size: {len(result)}")

    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_points([start_point, goal_point], size=10, edge_width=1, face_color="red", edge_color="red")
    viewer.add_points(result, size=10, edge_width=1, face_color="green", edge_color="green")
    napari.run()

if __name__ == "__main__":
    test_2D_image()
