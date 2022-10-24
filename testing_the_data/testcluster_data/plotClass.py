from typing import Tuple
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from typing import *
import random

class PlotWrapper():
    def __init__(self, nbrOfPlt: int, setInches: Tuple = (16,9), dpi: int = 150):
        self.__nbrOfPlt = nbrOfPlt
        self.__setInches = setInches
        self.__dpi = dpi
        self.__fig, self.__axs = plt.subplots(self.__nbrOfPlt)
        self.__fig.set_size_inches(self.__setInches)
        self.__fig.set_dpi(self.__dpi)
        self.__cntPlots = []
        self.__cntScatter = []
        self.__axsCfg = []
        for i in range(self.__nbrOfPlt):
            self.__axsCfg.append('None')
            self.__cntPlots.append(-1)
            self.__cntScatter.append(-1)
        self.__colors = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan']

    def define_colors(self, colors: List) -> None:
        self__colors = colors

    def add_title(self, titles: Dict[str, str]) -> None:
        if 'subtitle' in titles:
            self.__fig.suptitle(titles['subtitle'])

        for i in range(self.__nbrOfPlt):
            key = f'ax-{i}'
            if key in titles:
                self.__axs[i].set_title(titles[key])

    def save_2_dir(self, path: str, name: str) -> None:
        if not os.path.isdir(path):
            logging.INFO('Invalid path')
            return
        plt.tight_layout()
        plt.savefig(f'{path}/{name}.png')

    def set_grid(self, axs: int = None) -> None:
        for i in range(self.__nbrOfPlt):
            self.__axs[i].grid()
    
    def set_legend(self, pos: str = None, axs: int = -1):
        if axs >= self.__nbrOfPlt:
            logging.INFO('Missmatch between axs and nbr of plots!')
        if pos is not None:
            self.pos = pos
            if axs > -1:
                self.__axs[axs].legend(loc=pos)
            else:
                for i in range(self.__nbrOfPlt):
                    self.__axs[i].legend(loc=pos)
            return
        for i in range(self.__nbrOfPlt):
            self.__axs[i].legend(loc='lower left',  bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", borderaxespad=0, ncol=3)

    def add_plot(self, posOfAxs: int, dataY: List, dataX: List = None, label: str = None) -> None:
        if self.__axsCfg[posOfAxs] == self.__scatter():
            logging.INFO("Invalid operation")       
            return 
        self.__axsCfg[posOfAxs] = self.__plot()
        
        if self.__cntPlots[posOfAxs] >= len(self.__colors):
            self.__cntPlots[posOfAxs]=0
            logging.INFO('Nbr of plots exceeded Nbr of colors!')
        else:
            self.__cntPlots[posOfAxs]+=1

        if dataX is None:
            dataX = np.linspace(0, len(dataY)-1, num=len(dataY))
        self.__axs[posOfAxs].plot(dataX, dataY, color=self.__colors[self.__cntPlots[posOfAxs]], label=label)

    def add_scatter(self, posOfAxs: int, dataY: List, dataX: List, label: str = None) -> None:
        if self.__axsCfg[posOfAxs] == self.__plot():
            logging.INFO("Invalid operation")       
            return 
        self.__axsCfg[posOfAxs] = self.__scatter()

        if self.__cntScatter[posOfAxs] >= len(self.__colors):
            self.__cntScatter[posOfAxs]=0
            logging.INFO('Nbr of scatter exceeded Nbr of colors!')
        else:
            self.__cntScatter[posOfAxs]+=1

        if dataX is None:
            dataX = np.linspace(0, len(dataY)-1, num=len(dataY))
        self.__axs[posOfAxs].scatter(dataX, dataY, color=self.__colors[self.__cntScatter[posOfAxs]], label=label)

    def __plot(self) -> str:
        return "Plot"
    def __scatter(self) -> str:
        return "Scatter"

if __name__ == '__main__':
    t = PlotWrapper(4)
    data = []
    for i in range (100):
        data.append(random.randint(0, 500))
    t.add_plot(0, data)
    title = {
        'subtitle' : 'Test',
        'ax-0' : 'Test',
        'ax-1' : 'Test3'
    }
    t.add_title(title)
    t.set_grid()
    t.save_2_dir(os.getcwd(), 'Test')