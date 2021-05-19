import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os.path
import glob


filenames = glob.glob("plots/*27_temp_340*.png")
print(filenames)
with imageio.get_writer('plots/m27_t340.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)