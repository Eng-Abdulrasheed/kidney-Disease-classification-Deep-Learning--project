from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
import os
import numpy as np
from PIL import Image
import cv2
from scipy.signal import convolve2d
import math
from cnnClassifier.entity.config_entity import (DataPreprocessingConfig)



class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def ED_LBP_Sliding_Matrix(self, I, P):
        padding_amount = 1
        I = np.pad(I, pad_width=padding_amount, mode='constant') #, constant_values=0
        K = (2**P) - 1
        C_list = self.C_list_calculate(P)
        u_fac_matrix = self.u_sliding_factor(I.astype(np.float32), P)
        slid_factor = np.zeros((u_fac_matrix.shape),np.float32)
        m, n = u_fac_matrix.shape
        ED_LBP = np.zeros(u_fac_matrix.shape, np.float32)
        ED_LBP_matrix = np.zeros((u_fac_matrix.shape),np.float32)
        K_matrix = np.ones(u_fac_matrix.shape).astype(np.float32) * K
        offsets = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0),(0,0)]
        count = 1
        for offset in offsets:
            row_offset, col_offset = offset
            sliding_matrix = I[row_offset:row_offset + m, col_offset:col_offset + n].astype(np.float32) - u_fac_matrix.astype(np.float32)
            slid_factor = np.maximum(sliding_matrix, 0).astype(np.float32)
            k_norm = K_matrix.astype(np.float32) - u_fac_matrix.astype(np.float32)
            k_norm_nonzero = np.where(k_norm == 0, 1e-10, k_norm)
            A_factor = np.where(k_norm != 0, slid_factor / k_norm_nonzero, 0)
            ED_LBP_matrix = (A_factor.astype(np.float32) * C_list[count - 1]) + np.ones(A_factor.shape).astype(np.float32)
            ED_LBP += np.where(sliding_matrix >= 0, 2 ** ((count - 1) * ED_LBP_matrix.astype(np.float32)), 0)
            count  = count + 1
        
        ED_LBP = np.where(ED_LBP > 255, 255, np.round(ED_LBP))
        return ED_LBP.astype(int)

    def C_list_calculate(self, P):
        C = []
        for count in range(1, 9):
            c_value = ((P - count) * (count - 1)) / math.floor(((P - 1) / 2)**2)
            C.append(c_value)
        return C

    def u_sliding_factor(self, image_channel, P):
        result = np.zeros(image_channel.shape,np.float32)
        window_size = (3, 3)
        kernel = np.ones(window_size, np.float32)
        kernel[1, 1] = 0
        kernel = kernel / (2 * P)
        
        kernel2 = np.zeros(window_size, np.float32)
        kernel2[1, 1] = 1
        kernel2 = kernel2 / 2
        
        convolution_matrix = cv2.filter2D(image_channel, -1, kernel) + cv2.filter2D(image_channel, -1, kernel2)
        result = convolution_matrix[1:-1, 1:-1]
        return result.astype(np.float32)
    def process_and_save_images(self):
        real_dir = self.config.unzip_dir_real
        fake_dir = self.config.unzip_dir_fake

        real_output_dir = self.config.real_process_imgs
        fake_output_dir = self.config.fake_process_imgs

        os.makedirs(real_output_dir, exist_ok=True)
        os.makedirs(fake_output_dir, exist_ok=True)

        for input_dir, output_dir in [(real_dir, real_output_dir), (fake_dir, fake_output_dir)]:
            for img_name in os.listdir(input_dir):
                img_path = os.path.join(input_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error reading image {img_path}")
                        continue

                    resized_face = cv2.resize(img, (512, 512))
                    face_image_array = np.array(resized_face)
                    image_rgb = cv2.cvtColor(face_image_array, cv2.COLOR_BGR2RGB)
                    ED_LBP_image = np.zeros((image_rgb.shape), np.int16)
                    for i in range(3):
                        ED_LBP_image[:, :, i] = self.ED_LBP_Sliding_Matrix(image_rgb[:, :, i].astype(np.int16), 8)

                    processed_img_path = os.path.join(output_dir, img_name)
                    # Use the corrected method for saving images
                    cv2.imwrite(processed_img_path, ED_LBP_image)

                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    
    # def process_and_save_images(self):
    #     os.makedirs(self.config.real_process_imgs, exist_ok=True)
    #     os.makedirs(self.config.fake_process_imgs, exist_ok=True)

    #     num_augmented_versions = 5  # Number of augmented versions to create for each image

    #     for input_dir, output_dir in [(self.config.unzip_dir_real, self.config.real_process_imgs), 
    #                                 (self.config.unzip_dir_fake, self.config.fake_process_imgs)]:
    #         for img_name in os.listdir(input_dir):
    #             img_path = os.path.join(input_dir, img_name)
    #             try:
    #                 img = cv2.imread(img_path)
    #                 if img is None:
    #                     print(f"Error reading image {img_path}")
    #                     continue

    #                 for i in range(num_augmented_versions):
    #                     augmented_img = img.copy()

    #                     if self.config.params_is_augmentation:
    #                         augmented_img = self.apply_manual_augmentation(augmented_img)

    #                     resized_img = cv2.resize(augmented_img, (512, 512))
    #                     img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    #                     ED_LBP_image = np.zeros(img_rgb.shape, np.int16)
    #                     for j in range(3):
    #                         ED_LBP_image[:, :, j] = self.ED_LBP_Sliding_Matrix(img_rgb[:, :, j].astype(np.int16), 8)

    #                     processed_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_processed_{i}.png")
    #                     cv2.imwrite(processed_img_path, ED_LBP_image)

    #             except Exception as e:
    #                 print(f"Error processing image {img_path}: {e}")
