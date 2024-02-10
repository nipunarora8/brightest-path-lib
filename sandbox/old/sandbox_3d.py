import sys
import tifffile
import matplotlib.pyplot as plt

#TODO (cudmore): You need to define a logger fro your package
import logging
logger = logging.getLogger(__name__)
logging_level = logging.DEBUG
logger.setLevel(logging_level)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging_level)
formatter = logging.Formatter('%(levelname)7s - [%(module)s()] %(filename)s %(funcName)s() line:%(lineno)d -- %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def debug3d():
    path = '../PyMapManager-Data/one-timepoint/rr30a_s0_ch2.tif'
    
    _imgData = tifffile.imread(path)  # 3D
    logger.info(f'_imgData is shape: {_imgData.shape}')  # shape is (z,y,x)

    # load path (e.g. rr30a_s0_ch2.tif) into Fiji and manually ddetermine start/stop points
    startPnt = (32, 240, 355)  # (z,y,x)
    zStart = startPnt[0]

    goalPnt = (31, 215, 813)  # (z,y,x)
    zGoal = goalPnt[0]

    #
    # show these in start/goal in matplotlib
    # plt.imshow(_imgData[zStart])
    # plt.plot(startPnt[2], startPnt[1], 'og')  # (x,y)
    # plt.imshow(_imgData[zGoal])
    # plt.plot(goalPnt[2], goalPnt[1], 'or')  # (x,y)
    # plt.show()

    # run tracing
    queue = Queue()
    search_thread = AStarThread(_imgData, startPnt, goalPnt, queue)
    search_thread.start()

    _updateInterval = 256  # wait for this number of results and update plot
    plotItems = []
    while search_thread.is_alive() or not queue.empty(): # polling the queue
        if search_thread.search_algorithm.found_path:
            break

        try:
            item = queue.get(False)
            # update a matplotlib/pyqtgraph/napari interface
            plotItems.append(item)
            if len(plotItems) > _updateInterval:
                
                # Is plotItems[0] in (z,y,x) or (z,x,y) ???
                # print(f'after waiting for {_updateInterval} new values...')
                # print(f'  got plotItems[0] that looks like {plotItems[0]}')
                                
                plotItems = []

        except Empty:
            # Handle empty queue here
            pass

    if search_thread.search_algorithm.found_path:
        _result = search_thread.search_algorithm.result
        print(f'_result: {type(_result)}')

if __name__ == '__main__':
    debug3d()