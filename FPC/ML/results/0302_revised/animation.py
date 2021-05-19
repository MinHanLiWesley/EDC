import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os.path
import glob


filenames = glob.glob("plots/predict/*32_temp_320*.png")
print(filenames)
with imageio.get_writer('plots/predict/m32_t320.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)