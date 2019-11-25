import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_differential_filter():
    
    filter_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    filter_y = filter_x.transpose()
    return filter_x, filter_y


def filter_image(im, filter):

    im_filtered = np.zeros((im.shape[0],im.shape[1]))
    pad_size = (int)(filter.shape[0]/2)
    im = np.pad(im, ((pad_size,pad_size),(pad_size,pad_size)), 'constant')

    for x1 in range(im_filtered.shape[0]):
        for x2 in range(im_filtered.shape[1]):
            for x3 in range(filter.shape[0]):
                for x4 in range(filter.shape[1]):
                    im_filtered[x1,x2] += im[x1+x3,x2+x4]*filter[x3,x4]

    return im_filtered


def get_gradient(im_dx, im_dy):
    
    grad_mag = np.copy(im_dx)
    grad_angle = np.copy(im_dx)

    for x1 in range(grad_mag.shape[0]):
        for x2 in range(grad_mag.shape[1]):
            grad_mag[x1,x2] = math.sqrt(math.pow(im_dx[x1,x2],2)+math.pow(im_dy[x1,x2],2))
            grad_angle[x1,x2] = math.degrees(np.arctan2(im_dy[x1,x2],im_dx[x1,x2])) % 180;

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    
    ori_histo = np.zeros((int(grad_mag.shape[0]/cell_size),int(grad_mag.shape[1]/cell_size),6))

    for x1 in range(grad_mag.shape[0]-(grad_mag.shape[0]%cell_size)):
        for x2 in range(grad_mag.shape[1]-(grad_mag.shape[1]%cell_size)):
            bin_number = int(grad_angle[x1,x2]/15)
            if(bin_number == 0 or bin_number == 11 ):
                ori_histo[int(x1/cell_size),int(x2/cell_size),0] += grad_mag[x1,x2];
            elif(bin_number == 1 or bin_number == 2 ):
                ori_histo[int(x1/cell_size),int(x2/cell_size),1] += grad_mag[x1,x2];
            elif(bin_number == 3 or bin_number == 4 ):
                ori_histo[int(x1/cell_size),int(x2/cell_size),2] += grad_mag[x1,x2];
            elif(bin_number == 5 or bin_number == 6 ):
                ori_histo[int(x1/cell_size),int(x2/cell_size),3] += grad_mag[x1,x2];
            elif(bin_number == 7 or bin_number == 8 ):
                ori_histo[int(x1/cell_size),int(x2/cell_size),4] += grad_mag[x1,x2];
            else:
                ori_histo[int(x1/cell_size),int(x2/cell_size),5] += grad_mag[x1,x2];

    return ori_histo


def get_block_descriptor(ori_histo, block_size):

    ori_histo_normalized = np.zeros((ori_histo.shape[0]-block_size+1,ori_histo.shape[1]-block_size+1,6*block_size*block_size))

    for x1 in range(ori_histo_normalized.shape[0]):
        for x2 in range(ori_histo_normalized.shape[1]):
            sum = 0.001*0.001
            for x3 in range(x1,x1+block_size):
                for x4 in range(x2,x2+block_size):
                    for x5 in range(6):
                        sum += ori_histo[x3,x4,x5]*ori_histo[x3,x4,x5]
            sum = math.sqrt(sum)
            block_number = 0
            for x3 in range(x1,x1+block_size):
                for x4 in range(x2,x2+block_size):
                    for x5 in range(6):
                        ori_histo_normalized[x1,x2,x5+(block_number*6)] = ori_histo[x3,x4,x5]/sum
                    block_number += 1


    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    
    filter_x, filter_y = get_differential_filter()
    diff_x = filter_image(im,filter_x)
    diff_y = filter_image(im,filter_y)
    grad_mag,grad_ang = get_gradient(diff_x,diff_y)
    ori_histo = build_histogram(grad_mag,grad_ang,8)
    hog = get_block_descriptor(ori_histo,2)

    # visualize to verify
    visualize_hog(im, hog, 8, 2)
    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

if __name__=='__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)


