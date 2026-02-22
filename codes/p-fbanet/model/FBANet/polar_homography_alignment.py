import cv2
import os
import numpy as np
# from motion_selection import model_select

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

LR_patch_path = '/root/data/polar/polar_burst/test'
save_path = '/home/data1/dataset/RealBSR/RGB_aligned/train/LR_aligned'
save_gt_path = '/home/data1/dataset/RealBSR/RGB_aligned/train/GT'

def process_one_frame(i, im1, LR_patch_path, LR_list, save_LR_path):
    im2_path = '{}/{}/frame_{:02d}.png'.format(LR_patch_path, LR_list, i)
    im2 = cv2.imread(im2_path)

    if not os.path.exists(im2_path):
        logs = open('DRealBSR_test.txt', 'a')
        logs.write(im2_path)
        logs.write('\n')
        logs.close()
        return

    print("processing image {}".format(im2_path))

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    try:
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, i),
                    im2_aligned)

    except:
        cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, i), im2)
        print("An error occured when ECC not converge")

def process_one_image(LR_list):
    global LR_patch_path

    base_frame_path = '{}/{}/frame_00.png'.format(LR_patch_path, LR_list)

    im1 = cv2.imread(base_frame_path)

    save_LR_path = os.path.join(LR_patch_path, LR_list, 'aligned')
    if not os.path.exists(save_LR_path):
        os.makedirs(save_LR_path)

    with ThreadPoolExecutor(max_workers=16) as t:
        for i in range(1, 14):
            t.submit(lambda cxp:process_one_frame(*cxp),(i, im1, LR_patch_path, LR_list, save_LR_path))

    # if len(os.listdir(save_LR_path)) == 14:
    #     continue

    # cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x4_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, 0), im1)
    cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(save_LR_path, LR_number1, LR_number2, 0), im1)
    cv2.imwrite('{}/{}_MFSR_Sony_{:04d}_x4.png'.format(save_hr_path, LR_number1, LR_number2, 0), gt_img)


def main():
    pool = multiprocessing.Pool(16) # 每组图分配一个进程
    pool.map(process_one_image, os.listdir(LR_patch_path))

if __name__ == '__main__':
    main()