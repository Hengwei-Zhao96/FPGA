from img_io import read_img, write_img
import numpy as np
import PIL.Image as Image


color_board = [
                [0, 0, 0],
                [255, 0, 0],
                [255, 255, 255],
                [176, 48, 96],
                [255, 255, 0],
                [255, 127, 80],
                [0, 255, 0],
                [0, 205, 0],
                [0, 139, 0],
                [127, 255, 212]
        ]
        

def classmap_2_rgbmap(id):
    print(id)
    global color_board
    palette = np.asarray(color_board)
    h, w = 20,20

    rgb = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            rgb[i, j, :] = color_board[id]

    r = Image.fromarray(rgb[:, :, 0]).convert('L')
    g = Image.fromarray(rgb[:, :, 1]).convert('L')
    b = Image.fromarray(rgb[:, :, 2]).convert('L')

    rgb = Image.merge("RGB", (r, g, b))

    return rgb
    
for index in range(10):
    rgb = classmap_2_rgbmap(index)
    rgb.save("./temp/" + str(index) + ".png")