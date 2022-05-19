#!/usr/bin/env python3
"""
plotting in matplotlib while receiving data
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor

# Dummy data generator
class DataGen():
    def __init__(self, xlim=[-10,10], ylim=[-10,10]) -> None:
        self.data = np.zeros(2)
        self.xlim = xlim
        self.ylim = ylim
        self.dt = 0.1

    def gen_data(self):
        # brownian motion / random walk
        while True:
            diff_vel = np.random.normal(0,1, size=2)
            self.data += diff_vel*self.dt
            # print(self.data)
            time.sleep(self.dt)

    def get_data(self):
        return self.data

class Plotter():
    def __init__(self, data_src, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), res=0.05) -> None:
        self.data_src = data_src
        fig, ax = plt.subplots(figsize=(10,10))
        self.fig = fig
        self.ax = ax
        
        self.res = res
        self.w = int(xlim[1] - xlim[0])
        self.h = int(ylim[1] - ylim[0])
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.grid = np.zeros((int((xlim[-1]-xlim[0])/res), int((ylim[-1]-ylim[0])/res)))
        self.title = ax.text(0.5,0.85, "--", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")

        self.ax_img = self.ax.imshow(self.grid.T, cmap=plt.get_cmap("binary"), extent=[
            xlim[0], xlim[1], ylim[0], ylim[1]], animated=True, vmin=0, vmax=1)

        self.animator = None

    def get_ij(self, x, y):
        i = int(np.floor((x + self.w)/(2*self.res)))
        j = int(np.floor((-y + self.h)/(2*self.res)))
        if i < 0 or i >= self.grid.shape[0]:
            raise ValueError('x out of bounds')
        if j < 0 or j >= self.grid.shape[1]:
            raise ValueError('y out of bounds')
        return i, j
    
    def get_probability(self, x, y):
        i, j = self.get_ij(x, y)
        return self.grid[i, j]

    def prob_update(self, i, j, prob):
        p0 = self.grid[i,j]
        alpha = 0.5
        p = alpha*p0 + (1 - alpha)*prob
        if p > 1:
            p = 1
        elif p < 0:
            p = 0
        self.grid[i, j] = p
        
    def update(self, frame):
        pos = self.data_src.get_data()
        i,j = self.get_ij(pos[0], pos[1])
        self.prob_update(i, j, 0.5)
        self.ax_img.set_data(self.grid)
        self.title.set_text("frame " + str(frame))
        # print("data set", pos)
        # print("i,j: ", [i,j])
    
    def run(self):
        if self.animator is None:
            self.animator = FuncAnimation(
                self.fig,
                self.update,
                frames=20,
                interval=100,
            )
            plt.show()

if __name__ == '__main__':
    # try:
    data_src = DataGen()
    plotter = Plotter(data_src)

    with ThreadPoolExecutor(max_workers=5) as tpe:
        print("running threads")
        tpe.submit(data_src.gen_data)
        plotter.run()

    # except Exception as e:
    #     print(e)

    