import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import random
import math


def find_match(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()

    x1_list = []
    x2_list = []
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    nbrs = NearestNeighbors(n_neighbors=2).fit(des2)
    dist,indices = nbrs.kneighbors(des1)
    
    for x in range(len(kp1)):
        if(dist[x][0]/dist[x][1] <= 0.7):
            x1_list.append([kp1[x].pt[0],kp1[x].pt[1]])
            x2_list.append([kp2[indices[x][0]].pt[0],kp2[indices[x][0]].pt[1]])

    x1 = np.asarray(x1_list)    
    x2 = np.asarray(x2_list)

    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):

    best_A = np.zeros((6,1))
    A = np.zeros((6,1))
    max_inliers = 0
    inliers = 0

    for i in range(ransac_iter):
        inliers = 0
        point_indices = random.sample(range(0, x1.shape[0]), 3)
        x1_points = x1[point_indices,:]
        x1_points = [[x1_points[0,0],x1_points[0,1],1,0,0,0],[0,0,0,x1_points[0,0],x1_points[0,1],1],
                    [x1_points[1,0],x1_points[1,1],1,0,0,0],[0,0,0,x1_points[1,0],x1_points[1,1],1],
                    [x1_points[2,0],x1_points[2,1],1,0,0,0],[0,0,0,x1_points[2,0],x1_points[2,1],1]] 
        x2_points = x2[[point_indices],:]
        x2_points = np.reshape(x2_points,(6,1))
        A = np.linalg.solve(x1_points,x2_points)
        A = np.reshape(A,(2,3))
        for j in range(x1.shape[0]):
            b = np.matmul(A,[[x1[j][0]],[x1[j][1]],[1]])
            if(np.linalg.norm([[x2[j][0]],[x2[j][1]]]-b) <= ransac_thr):
                inliers += 1
        if(inliers>=max_inliers):
            best_A = A
            max_inliers = inliers

    
    best_A = np.append(best_A,[[0,0,1]],axis=0)
    return best_A

def warp_image(img, A, output_size):
    
    img_warped = np.zeros((output_size[0],output_size[1]))

    for x in range(output_size[1]):
        for y in range(output_size[0]):
            point = np.matmul(A,[[x],[y],[1]])
            x_point = math.floor(point[0][0])
            y_point = math.floor(point[1][0])
            if((x_point>=img.shape[1] or x_point<0 or y_point>=img.shape[0] or y_point<0) == False):
                img_warped[y,x] = img[y_point,x_point]

    return img_warped


def align_image(template, target, A):

    template = template.astype('float')/255.0
    target = target.astype('float')/255.0
 
    diff_y,diff_x = np.gradient(template)
    diff_template = np.stack((diff_x,diff_y), axis=2)

    jacobian = np.zeros((diff_x.shape[0],diff_x.shape[1],2,6))
    for x in range(diff_x.shape[0]):
        for y in range(diff_x.shape[1]):
            current =  np.array([[y,x,1,0,0,0],[0,0,0,y,x,1]])
            jacobian[x][y] = current

    SDI = np.zeros((diff_x.shape[0],diff_x.shape[1],1,6))
    for x in range(diff_x.shape[0]):
        for y in range(diff_x.shape[1]):
            current =  np.matmul(diff_template[x][y],jacobian[x][y])
            SDI[x][y] = current

    all_hessian = np.zeros((diff_x.shape[0],diff_x.shape[1],6,6))
    for x in range(diff_x.shape[0]):
        for y in range(diff_x.shape[1]):
            current =  np.matmul(SDI[x][y].transpose(),SDI[x][y])
            all_hessian[x][y] = current

    H = np.zeros((6,6))
    for x in range(diff_x.shape[0]):
        for y in range(diff_x.shape[1]):
                    H += all_hessian[x,y]

    delta_P = 5
    errors= []
    while( np.linalg.norm(delta_P) > 0.001):
        img_warped = warp_image(target, A, template.shape)
        I_error = img_warped - template
        errors.append(np.linalg.norm(I_error))
        F = np.zeros((6,1))
        for x in range(I_error.shape[0]):
            for y in range(I_error.shape[1]):
                current =  np.matmul(SDI[x][y].transpose(),[[I_error[x][y]]])
                F += current
        delta_P = np.matmul(np.linalg.inv(H),F)
        delta_A = np.reshape(delta_P,(2,3)) 
        delta_A = np.append(delta_A,[[0,0,1]],axis=0)
        delta_A[0,0] += 1
        delta_A[1,1] += 1
        A = np.matmul(A, np.linalg.inv(delta_A))

    A_refined = A
    return A_refined,np.array(errors)


def track_multi_frames(template, img_list):
    
    A_list = []
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, 15, 1000)

    for x in range(len(img_list)):
        A,errors = align_image(template, img_list[x], A)
        A_list.append(A)
        template = warp_image(img_list[x], A, template.shape)

    return A_list

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    np.random.seed(641)
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, 15, 10000)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


