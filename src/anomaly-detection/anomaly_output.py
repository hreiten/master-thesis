import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np


from settings import UPPER_ANOMALY_THRESHOLD
from settings import LOWER_ANOMALY_THRESHOLD
from settings import WINDOW_LENGTH

class AnomalyOutput():
    def __init__(self, positions, figsize = (5,2)):
        plt.ion()
        self.fig = plt.figure(figsize = figsize)
        self.axs = [self.fig.add_subplot(i) for i in positions]

        self.fig.canvas.draw()
        self.fig.show()

    def update_plot(self, ax, x, y, labels, y_lim, x_lim):
        ax.set_xlim(x_lim[0], x_lim[1])
        if (ax.lines):
            for i in range(0, len(ax.lines)):
                ax.lines[i].set_xdata(x)
                ax.lines[i].set_ydata(y[i])
                
            
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            for i in range(0, len(y)):
                ax.plot(x, y[i], label=labels[i])
            ax.legend()
        
        self.fig.canvas.draw()

    def mark_anomalies_in_plot(self, ax, y, i):
        ax.fill_between(
            np.linspace(i - 1, i + 1, 5), 
            0, 
            1, 
            where = (
                y[-3] > UPPER_ANOMALY_THRESHOLD or 
                y[-2] > UPPER_ANOMALY_THRESHOLD or 
                y[-1] > UPPER_ANOMALY_THRESHOLD or
                y[-1] < LOWER_ANOMALY_THRESHOLD or
                y[-2] < LOWER_ANOMALY_THRESHOLD or
                y[-3] < LOWER_ANOMALY_THRESHOLD
            ),
            facecolor='red', 
            alpha=0.2, 
            linewidth = 10.2
        )
        
