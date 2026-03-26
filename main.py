import numpy as np
import matplotlib.pyplot as plt
import imaging
import os, sys


image_name = "DSC_1339_768x512_rggb"
image_raw = np.fromfile("images/" + image_name + ".raw", dtype=np.uint16)


