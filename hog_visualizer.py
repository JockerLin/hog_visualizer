import cv2
import numpy as np

def visualize_hog(img):
    '''
    A function to visualize the results of a HOG calculation on a given image

    Parameters:
    img : The location of the image file
    '''

    #TODO:
    #   Fix hog_shape to work with non-default cell_sz
    #   Fix calculation of cell to work with non-default block_sz (replace the 2)

    m = cv2.imread(img, -1)
    win_sz = (128, 128)
    cell_sz = np.array((8, 8))
    block_sz = np.array((16, 16))
    stride = np.array((8, 8))
    nbins = 9

    # Fix scaling so the maximum dimension is 600
    m_scale = 600 / np.max(m.shape)
    m = cv2.resize(m, None, fx=m_scale, fy=m_scale)

    # Resize the image for HOG extraction
    img = cv2.resize(m, win_sz)
    img_scale = np.array(m.shape[0:2]) / np.array(img.shape[0:2])

    # If there's an alpha channel, remove it and replace it with a white
    # background
    if img.shape[2] == 4:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,3] == 0:
                    for k in range(3):
                        img[i,j,k] = 255

    # hog.compute oddly doesn't work without this explicit type conversion
    img = img[:,:,0:3].astype(np.uint8)

    # Calculate the HOG features
    hog = cv2.HOGDescriptor((img.shape[0],img.shape[1]), 
                            tuple(block_sz), 
                            tuple(cell_sz), 
                            tuple(stride), 
                            nbins)
    hog_d = hog.compute(img, locations=[(0,0)])

    # Reshapes the HOG array to correspond to position on the image
    hog_shape = (((img.shape[0]/cell_sz[0] - block_sz[0]/cell_sz[0] + 1).astype(np.uint32),
                  (img.shape[1]/cell_sz[1] - block_sz[1]/cell_sz[1] + 1).astype(np.uint32),
                  (np.prod(block_sz/cell_sz) * nbins).astype(np.uint32)
                ))
    hog_d = hog_d.reshape(hog_shape)

    # Retrieves the cell data from each block. Outer loop consists of the HOG
    # descriptor blocks, while inner loops handle the cells within each block.
    # Each cell is renormalized, so only exceptionally strong gradients should
    # show strong lines.
    cell_val = np.zeros(((img.shape[0]/cell_sz[0]).astype(np.uint32),
                         (img.shape[1]/cell_sz[1]).astype(np.uint32),
                         nbins
                       ))     
    for i in range(hog_d.shape[0]):
        for j in range(hog_d.shape[1]):
            for ii in range((block_sz[0]/cell_sz[0]).astype(np.uint32)):
                for jj in range((block_sz[1]/cell_sz[1]).astype(np.uint32)):
                    cell = hog_d[i,j,((ii*2+jj)*nbins):((ii*2+jj+1)*nbins)]
                    if np.max(cell) != 0:
                        cell = cell / np.linalg.norm(cell)
                    cell_val[i+ii,j+jj,:] = cell

    # Draws the visualization on the original image
    for i in range(cell_val.shape[0]):
        for j in range(cell_val.shape[1]):
            for k in range(nbins):
                # Creates a unit vector corresponding to theta
                theta = -np.pi/2 + np.pi * (k/nbins)
                u_vec = np.array((np.sin(theta), -np.cos(theta)))

                # Creates the vector that will be drawn on the midpoint of the
                # cell
                vec = u_vec * cell_val[i,j,k] * np.min(cell_sz * img_scale[::-1]) / 2

                # Specifies the midpoint in beg, then sets the endpoints of
                # each line in end, based on the values in vec
                beg = (cell_sz/2 + np.array((i, j)) * cell_sz) * img_scale[::-1]
                end1 = beg + vec
                end2 = beg - vec
                beg = beg.astype(np.uint32)
                end1 = end1.astype(np.uint32)
                end2 = end2.astype(np.uint32)

                # Draws the lines
                if cell_val[i,j,k] == np.max(cell_val[i,j,:]):
                    cv2.line(m, tuple(beg), tuple(end1), (255,0,0), 1)
                    cv2.line(m, tuple(beg), tuple(end2), (255,0,0), 1)

    cv2.imshow('img', m)
    cv2.waitKey(0)

    return m


if __name__ == '__main__':
    visualize_hog('test.png')
    #visualize_hog('test.jpeg')
    #visualize_hog('rin1.png')

