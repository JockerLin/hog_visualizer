import cv2
import numpy as np

def visualize_hog(img):
    m = cv2.imread(img, -1)
    m = cv2.resize(m, (600, 600))
    img = cv2.resize(m, (128, 128))
    scale = np.array(m.shape) / np.array(img.shape)
    y_sz = img.shape[0]
    x_sz = img.shape[1]

    if len(img.shape) == 4:
        for i in range(y_sz):
            for j in range(x_sz):
                if img[i,j,3] == 0:
                    for k in range(3):
                        img[i,j,k] = 255

    img = img[:,:,0:3].astype(np.uint8)

    hog = cv2.HOGDescriptor((y_sz,x_sz), (16,16), (8, 8), (8, 8), 9)
    hog_d = hog.compute(img)
    np.savetxt('test.csv', hog_d.reshape(-1,36), delimiter=',')
    hog_d = hog_d.reshape(int(128/8)-1,int(128/8)-1,9*4)
    midpoints = get_midpoints(img, 8, 16)

    draw_line(m, scale[0:2], hog_d, midpoints)
    cv2.imshow('img', m)
    cv2.waitKey(0)
    return


def draw_line(m, scale, val, midpoints):

    max_line_len = np.min(scale)
    for i in range(midpoints.shape[0]):
        for j in range(midpoints.shape[1]):
            for k in range(9):
                angle = np.pi * k/9 * np.pi/2
                unit = np.array((np.sin(angle), np.cos(angle)))
                line_len = val[i,j,k]
                if np.max(val[i,j,:]) != 0:
                    line_len = line_len / np.max(val[i,j,:])
                beg = midpoints[i,j,:] * scale
                end = beg + unit * line_len * max_line_len * 4
                cv2.line(m, tuple(beg.astype(np.uint32)), tuple(end.astype(np.uint32)), (255,0,0), 1)
                #end = beg - unit * line_len * max_line_len * 4
                #cv2.line(m, tuple(beg.astype(np.uint32)), tuple(end.astype(np.uint32)), (255,0,0), 1)


#            for k in range(9):
#                angle = np.pi * k/9 - np.pi/2
#                unit = np.array((-np.sin(angle), np.cos(angle)))
#                line_len = val[i,j,k] + val[i,j,(k+9)] + val[i,j,(k+18)] + val[i,j,(k+27)]
#                beg = midpoints[i,j,:] * scale
#                end = beg + unit * line_len * max_line_len * 4
#                cv2.line(m, tuple(beg.astype(np.uint32)), tuple(end.astype(np.uint32)), (255,0,0), 1)
#                end = beg - unit * line_len * max_line_len * 4
#                cv2.line(m, tuple(beg.astype(np.uint32)), tuple(end.astype(np.uint32)), (255,0,0), 1)

    return


def get_midpoints(img, cell_sz, block_sz):
    y_sz = np.floor(img.shape[0]/cell_sz) - (block_sz/cell_sz - 1)
    x_sz = np.floor(img.shape[1]/cell_sz) - (block_sz/cell_sz - 1)
    y_sz = y_sz.astype(np.uint8)
    x_sz = x_sz.astype(np.uint8)
    midpoints = np.zeros((y_sz, x_sz, 2))

    for i in range(y_sz):
        for j in range(x_sz):
            midpoints[i, j, 0] = (i+1) * block_sz / 2
            midpoints[i, j, 1] = (j+1) * block_sz / 2

    return midpoints


if __name__ == '__main__':
    visualize_hog('circle.png')
    #visualize_hog('test.jpeg')
    #visualize_hog('rin1.png')

