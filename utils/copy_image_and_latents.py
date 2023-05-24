import os
import shutil

if __name__ == '__main__':
    src_base_path = '/home/andrei/Documents/licenta/dbs/generated_for_test'
    dst_base_path = '/home/andrei/Documents/licenta/face_seg_dataset/test'

    for dir_name in os.listdir(dst_base_path):
        src_image_path = os.path.join(src_base_path, f'{dir_name}.png')
        src_latent_path = os.path.join(src_base_path, f'{dir_name}_latent.npy')

        dst_dir_path = os.path.join(dst_base_path, dir_name)
        dst_image_path = os.path.join(dst_dir_path, f'{dir_name}.png')
        dst_latent_path = os.path.join(dst_dir_path, f'{dir_name}.npy')

        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_latent_path, dst_latent_path)