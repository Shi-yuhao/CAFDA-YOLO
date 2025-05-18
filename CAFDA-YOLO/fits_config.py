fits_config = {
    'fits_parent_path': '../sdss_detect',   # your fits data path
    
    'JPEG_save_path': '../dataset_example/dataset/VOCdevkit/VOC2007/JPEGImages',
    'txt_save_path': '../dataset_example/dataset/VOCdevkit/VOC2007/ImageSets/Main',
    'annotation_save_path': '../dataset_example/dataset/VOCdevkit/VOC2007/Annotations',
    'img_save_path': '../dataset_example/dataset/VOCdevkit/img',
    'window_size': 256,
    'window_number': 10,
    
    'used_filter': 'ugriz',
    'reproject_target_filter': 'g',
    'fit_reproject_a': 70,
    'fit_reproject_n_samples': 0,
    'fit_reproject_contrast': 0.06,

    'scan_score_threshold': 0.75
}
