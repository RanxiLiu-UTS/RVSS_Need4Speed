import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from machinevisiontoolbox import Image

# car = Image.Read("/home/rxliu/git/RVSS_Need4Speed/data/train_starter/0000000.00.jpg")
# car.disp()
# plt.show(block=True)

# im = Image.Read("monalisa.png")
# im.disp()
# # mona.smooth(sigma=5).disp()
# Image.Hstack([im, im.smooth(sigma=5)]).disp()
# # plt.show(block=True)

# im = Image.Read("shark2.png")   # read a binary image of two sharks
im = Image.Read("multiblobs.png")
# im.disp()   # display it with interactive viewing tool

blobs = im.blobs()  # find all the white blobs
# print(blobs)

blobs.plot_box(color="g", linewidth=2)  # put a green bounding box on each blob
blobs.plot_centroid(label=True)  # put a circle+cross on the centroid of each blob

# labels = blobs.label_image()
# labels.disp(colormap="viridis", ncolors=len(blobs), colorbar=dict(shrink=0.8, aspect=20*0.8))

# blobs.dotfile(show=True)

plt.show(block=True)
