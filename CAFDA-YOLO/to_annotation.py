from astropy.io import fits
import os
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import cv2
from lxml import etree
from tqdm import tqdm
from fits_operator import fits_reproject
from fits_config import fits_config as config
from typing import List, Tuple, Union, Any
import argparse


def convert_2_pixel_coordinates(header, ra, dec):
    wcs = WCS(header)
    return wcs.all_world2pix([[ra, dec]], 1)


def read_ra_dec_from_csv(csv_path,line_num):
    df = pd.read_csv(csv_path, header=0, converters={
        'ra': float,
        'dec': float
    })
    return df['ra'].values[line_num], df['dec'].values[line_num]

def xml_save(save_path, folder, filename, width, height, depth, obj_name, left_up, right_down):
    root = etree.Element('annotation')
    folder_node = etree.SubElement(root, 'folder')
    folder_node.text = folder
    filename_node = etree.SubElement(root, 'filename')
    filename_node.text = filename
    source_node = etree.SubElement(root, 'source')
    database_node = etree.SubElement(source_node, 'database')
    database_node.text = 'bss'
    size_node = etree.SubElement(root, 'size')
    width_node = etree.SubElement(size_node, 'width')
    width_node.text = str(width)
    height_node = etree.SubElement(size_node, 'height')
    height_node.text = str(height)
    depth_node = etree.SubElement(size_node, 'depth')
    depth_node.text = str(depth)
    segmented_node = etree.SubElement(root, 'segmented')
    segmented_node.text = '0'
    object_node = etree.SubElement(root, 'object')
    name_node = etree.SubElement(object_node, 'name')
    name_node.text = obj_name
    pose_node = etree.SubElement(object_node, 'pose')
    pose_node.text = 'Unspecified'
    truncated_node = etree.SubElement(object_node, 'truncated')
    truncated_node.text = '0'
    difficult_node = etree.SubElement(object_node, 'difficult')
    difficult_node.text = '0'
    bndbox_node = etree.SubElement(object_node, 'bndbox')
    xmin_node = etree.SubElement(bndbox_node, 'xmin')
    xmin_node.text = str(left_up[0])
    ymin_node = etree.SubElement(bndbox_node, 'ymin')
    ymin_node.text = str(left_up[1])
    xmax_node = etree.SubElement(bndbox_node, 'xmax')
    xmax_node.text = str(right_down[0])
    ymax_node = etree.SubElement(bndbox_node, 'ymax')
    ymax_node.text = str(right_down[1])
    tree = etree.ElementTree(root)
    tree.write(os.path.join(save_path, filename.split('.npy')[0] + '.xml'), pretty_print=True,
               encoding='utf-8')


def create_bbox(img: np.ndarray, obj_x: int, obj_y: int) -> Tuple[List[int], List[Union[int, Any]]]:
    h, w = img.shape
    left_up_coord = [0, 0]
    right_down_coord = [h - 1, w - 1]
    # To Up
    for i in range(obj_x, 0, -1):  
        if img[i, obj_y] != 0:
            left_up_coord[0] = i
        else:
            break
    # To Down
    for i in range(obj_x, h - 1): 
        if img[i, obj_y] != 0:
            right_down_coord[0] = i
        else:
            break 
    # To Left
    for i in range(obj_y, 0, -1):
        if img[obj_x, i] != 0:
            left_up_coord[1] = i
        else:
            break
    # To Right
    for i in range(obj_y, w - 1):
        if img[obj_x, i] != 0:
            right_down_coord[1] = i
        else:
            break
    left_up_coord[0] -= 1  # ymin
    left_up_coord[1] -= 1  # xmin 
    right_down_coord[0] += 1  #  ymax
    right_down_coord[1] += 1  #  xmax
    right_down_coord[0] = min(right_down_coord[0] + 1, h - 1)
    right_down_coord[1] = min(right_down_coord[1] + 1, w - 1)
    return left_up_coord, right_down_coord


def to_annotation(fits_path: str, fits_name: str,  line_num: int,save_process_img: bool = True,
                  enable_exists_check: bool = True) -> str:
    if os.path.exists(os.path.join(fits_path, '{}.xml'.format(fits_name))) and os.path.exists(
            os.path.join(fits_path, 'stack.png')) and enable_exists_check:
        print('\033[1;32m {} already exists. Skip \033[0m'.format(os.path.join(fits_path, '{}.xml'.format(fits_name))))
    else:
        used_filter = config['used_filter']
        reproject_target_filter = config['reproject_target_filter']
        fits_path_arr = []
        target_fits_path = ''
        for filename in os.listdir(fits_path):
            if filename.endswith('.fits.bz2') and filename.split('-')[1].lower() in used_filter.replace(
                    reproject_target_filter, '').lower():
                fits_path_arr.append(os.path.join(fits_path, filename))
            if filename.endswith('.fits.bz2') and filename.split('-')[1].lower() == reproject_target_filter.lower():
                target_fits_path = os.path.join(fits_path, filename)
        print(target_fits_path)
        reprojected_img = fits_reproject({
            'target_fits_path': target_fits_path,
            'fits_without_target_path': fits_path_arr
        }, config['fit_reproject_a'], config['fit_reproject_n_samples'],
            config['fit_reproject_contrast'])
        stack_img = reprojected_img[0]
        for img in reprojected_img[1:]:
            stack_img = stack_img * 0.7 + img * 0.3
        stack_img = np.where(np.squeeze(stack_img) > 0.06, 1, 0)
        h, w = stack_img.shape

        # Generate bbox
        obj_ra, obj_dec = read_ra_dec_from_csv(r' your csv path ', line_num)
        header = fits.open(target_fits_path)[0].header
        obj_x, obj_y = convert_2_pixel_coordinates(header, obj_ra, obj_dec)[0]
        obj_x = int(obj_x)
        obj_y = int(obj_y)
        left_up, right_down = create_bbox(stack_img, obj_x, obj_y)

        # Save xml
        xml_save(fits_path, fits_name, '{}.png'.format(fits_name), w, h, 1, 'bss', left_up, right_down)

        # Plot
        if save_process_img:
            stack_img[left_up[0]:right_down[0], left_up[1]] = 1
            stack_img[left_up[0]:right_down[0], right_down[1]] = 1  
            stack_img[left_up[0], left_up[1]:right_down[1]] = 1    
            stack_img[right_down[0], left_up[1]:right_down[1]] = 1   
            stack_img[obj_x, :] = 1  
            stack_img[:, obj_y] = 1  
            cv2.imwrite(os.path.join(fits_path, 'stack.png'), stack_img * 255.0)

        return '{} {},{},{},{},0'.format(fits_path, left_up[0], left_up[1], right_down[0], right_down[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input fits file path')
    parser.add_argument('-p', '--path',help='Path to fits file')
    args = parser.parse_args()
    fits_parent_path = config['fits_parent_path'] if args.path is None else args.path
    obj_txt_path = config['obj_txt_path']
    obj_txt = []
    err_list = []
    with tqdm(total=len([filename for filename in os.listdir(fits_parent_path) if
                         os.path.isdir(os.path.join(fits_parent_path, filename))]), desc='Creating bbox...',
              ncols=100) as pbar:
        for filename in os.listdir(fits_parent_path):
            if os.path.isdir(os.path.join(fits_parent_path, filename)):
                pbar.set_description('Creating bbox: {}'.format(filename))
                try:
                    line_num = int(filename.split('_')[0])-1
                    annotation = to_annotation(os.path.join(fits_parent_path, filename), filename,line_num,
                                               enable_exists_check=False)
                    if annotation:
                        obj_txt.append(annotation)
                except Exception as e:
                    print('\033[1;31m [Error]: {} \033[0m'.format(os.path.join(fits_parent_path, filename)))
                    err_list.append(os.path.join(fits_parent_path, filename))
                    pass
                pbar.update(1)

    with open(os.path.join(obj_txt_path if args.path is None else args.path, 'obj.txt'), 'w') as f:
        for row in obj_txt:
            f.write(row + '\n')
        f.close()

    print(err_list)
