"""Scripts to measure the time it takes to trace

Requires
--------
pip install pandas, tifffile

"""

import os
import time
# import requests
from queue import Queue

import numpy as np
import pandas as pd
import tifffile

from typing import List

import matplotlib

from matplotlib import pyplot as plt

import brightest_path_lib
import brightest_path_lib.algorithm

class pmmTracing():
    def __init__(self):
        self.testVar = 222
        self.paths = []

        self.image = None

    def setImage(self, tifPath : str):
        print('python loading tifPath:', tifPath)
        self.image = tifffile.imread(tifPath)
        print('   image:', self.image.shape)

    def runTracing(self, startPnt : List[float], stopPnt : List[float]) -> List[int]:
        """Run brightest path tracing from start to stop.
        
        Parameters
        ----------
        startPnt, stopPnt
            Start and stop point to trace from and to.
        """
        
        algorithm = brightest_path_lib.algorithm.NBAStarSearch(self.image, startPnt, stopPnt)

        startTime = time.time()

        path = algorithm.search()
        path = np.array(path)
        
        stopTime = time.time()
        timeItTook = round(stopTime-startTime, 3)
        print('   timeItTook:', timeItTook)

        return path
    
aTracing = pmmTracing()

def runBrightestPath(tifPath, csvPath):
    
    print('myScript.runBrightestPath()')
    print('   tifPath:', tifPath)
    print('   csvPath:', csvPath)
    
    print('python loading csv with pandas csvPath:', csvPath)
    df = pd.read_csv(csvPath, header=1)

    #localFile = 'rr30a_s0_ch2_8_bit.tif'
    print('python loading tifPath:', tifPath)
    image = tifffile.imread(tifPath)
    print('   image:', image.shape)

    # reduce to one segmentID and just roiType 'controlPnt'
    segmentID = 0
    df = df[ df['segmentID']==segmentID]
    df = df[ df['roiType']=='controlPnt']

    # get (z, y, x) value)
    zyx = df[['z', 'y', 'x']].to_numpy()
    print('   zyx:', zyx.shape)  # (28,3)

    startTime0 = time.time()
    paths = []
    pathLengths = []
    times = []
    queueSize = []
    step = 1  # 2 to find path between every other control point

    queue = Queue()

    pointsToAnalyze = list(range(0, zyx.shape[0], step))
    for i in pointsToAnalyze:
        if i+step > zyx.shape[0]-2:
            break

        start = np.array(zyx[i,:])
        end = np.array(zyx[i+step,:])

        algorithm = brightest_path_lib.algorithm.NBAStarSearch(image, start, end, open_nodes=queue)

        startTime = time.time()

        path = algorithm.search()
        
        paths.append(path)

        path = np.array(path)

        pathLengths.append(path.shape[0])

        stopTime = time.time()
        timeItTook = round(stopTime-startTime, 3)
        times.append(timeItTook)

        queueSize.append(queue.qsize())
        # print('queue:', queue.qsize())

        # with printing we lose like 100 ms
        # print(f'   {i} time {timeItTook} start:{start} end:{end}')

    stopTime0 = time.time()
    print('   took:', round(stopTime0-startTime0,3), 'seconds')

    # plot
    maxImage = np.max(image, axis=0)
    zyx = zyx[pointsToAnalyze]
    plotFigure(maxImage, paths, zyx, pathLengths, times, queueSize)

def plotFigure(maxImage, paths, zyx, pathLengths, times,
               queueSize,  # list of queussize
               ):
    """Plot an image with brightest path
    Also plot time it took for each path.
    """

    # fig, axs = plt.subplots(2, 1, sharex=True)
    fig, axs = plt.subplots(2, 1, sharex=False)

    axs[0].imshow(maxImage, cmap='gray')

    # plot the brightest path
    for path in paths:
        yPlot = [point[1] for point in path]
        xPlot = [point[2] for point in path]
        axs[0].scatter(xPlot, yPlot, c='y', s=3, alpha=1)

    # plot the seed control points
    yPlot = [point[1] for point in zyx]
    xPlot = [point[2] for point in zyx]
    axs[0].scatter(xPlot, yPlot, c='c', s=16, alpha=1)

    # Turn off tick labels
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    # turn off tick marks
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    #time it takes is dependent on cost function,
    #not number of pnts in the path (e.g. pathLengths)

    # xPlot1 = xPlot[0:-2]
    xPlot1 = pathLengths
    xPlot1 = queueSize

    # trying to set the size, having problems
    # axs[1].scatter(xPlot2, times, s=pathLengths[0:-2])

    axs[1].plot(xPlot1, times, 'o')
    
    # axs[1].set(xlabel="Path Length (points)", ylabel="Time (Sec)")
    axs[1].set(xlabel="Voxels Visited", ylabel="Time (Sec)")
    # axs[1].set(xlabel="Start Position (point)", ylabel="Time (Sec)")

    # despine top and right
    axs[1].spines[['right', 'top']].set_visible(False)

    plt.show()

if __name__ == '__main__':
    tifPath = os.path.join('data', 'sample-3d.tif')
    csvPath = os.path.join('data', 'sample-3d.txt')
    runBrightestPath(tifPath, csvPath)