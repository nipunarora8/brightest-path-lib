from queue import Empty, Queue
from threading import Thread
import tifffile
import time
from typing import List

from brightest_path_lib.algorithm import AStarSearch, NBAStarSearch

from skimage import data
import numpy as np
import matplotlib.pyplot as plt


class AStarThread(Thread):
    def __init__(self,
        image : np.ndarray,
        start_point : np.ndarray,
        goal_point : np.ndarray,
        queue = None):
        super().__init__(daemon=True)
        self.queue = queue
        self.search_algorithm = AStarSearch(image, start_point=start_point, goal_point=goal_point, open_nodes=queue)
    
    def cancel(self):
        self.search_algorithm.is_canceled = True
    
    def run(self):
        """
        run A* tracing algorithm
        """
        print("Searching...")
        tic = time.perf_counter()
        self.search_algorithm.search()
        toc = time.perf_counter()
        print(f"Found brightest path in {toc - tic:0.4f} seconds")
        print(f"path size: {len(self.search_algorithm.result)}")
        print(f"Number of nodes viewed: {self.search_algorithm.evaluated_nodes}")
        print("Done")


class NBAStarThread(Thread):
    def __init__(self,
        image : np.ndarray,
        start_point : np.ndarray,
        goal_point : np.ndarray,
        queue = None):
        super().__init__(daemon=True)
        self.queue = queue
        self.search_algorithm = NBAStarSearch(image, start_point=start_point, goal_point=goal_point, open_nodes=queue)

    def cancel(self):
        self.search_algorithm.is_canceled = True

    def run(self):
        """
        run NBA* tracing algorithm
        """
        print("Searching...")
        tic = time.perf_counter()
        self.search_algorithm.search()
        toc = time.perf_counter()
        print(f"Found brightest path in {toc - tic:0.4f} seconds")
        print(f"path size: {len(self.search_algorithm.result)}")
        print(f"Number of nodes viewed: {self.search_algorithm.evaluated_nodes}")
        print("Done")


def _plot_image(image: np.ndarray, start: np.ndarray, end: np.ndarray):
    plt.imshow(image, cmap='gray')
    plt.plot(start[1], start[0],'og')
    plt.plot(end[1], end[0], 'or')
    plt.pause(0.001)


def _plot_points(points: List[np.ndarray], color, size, alpha=1.0):
    """Plot points

    Args:
        points: [(y,x)]
    """
    yPlot = [point[0] for point in points]
    xPlot = [point[1] for point in points]

    plt.scatter(xPlot, yPlot, c=color, s=size, alpha=alpha)
    plt.pause(0.0001)


def plot_brightest_path():
    # image = data.cells3d()[30, 0]
    # start_point = np.array([0,192]) # [y, x]
    # goal_point = np.array([198,9])

    image = tifffile.imread('/Users/vasudhajha/Documents/mapmanager/brightest-path-lib/brightest_path_lib/a-star-image.tif')
    start_point = np.array([188, 71]) # (y,x)
    goal_point = np.array([116, 415])
    #goal_point = np.array([128, 628])

    _plot_image(image, start_point, goal_point)

    queue = Queue()

    # search_thread = AStarThread(image, start_point, goal_point, queue)
    search_thread = NBAStarThread(image, start_point, goal_point, queue)
    search_thread.start()  # start the thread, internally Python calls tt.run()

    _updateInterval = 100  # wait for this number of results and update plot
    plotItems = []
    while search_thread.is_alive() or not queue.empty(): # polling the queue
        # if search_thread.search_algorithm.found_path:
        #     break

        try:
            item = queue.get(False)
            # update a matplotlib/pyqtgraph/napari interface
            plotItems.append(item)
            if len(plotItems) > _updateInterval:
                _plot_points(plotItems, 'c', 8, 0.3)
                plotItems = []

        except Empty:
            # Handle empty queue here
            pass


    if search_thread.search_algorithm.found_path:
        plt.clf()

        _plot_image(image, start_point, goal_point)

        _plot_points(search_thread.search_algorithm.result, 'y', 4, 0.5)
        

    # keep the plot up
    plt.show()

if __name__ == "__main__":
    plot_brightest_path()
