import argparse
import os
import numpy as np
import SimpleITK as sitk
import pickle
from skimage.transform import resize


def get_bbox_from_mask(nonzero_mask, label_mask, spacing, axis=None, outside_value=0, ):
    if axis is None:
        axis = [0, 1, 2]
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    mask_voxel_label = np.where(label_mask != outside_value)
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, label_mask.shape[0], 0, label_mask.shape[1], 0, \
                                                           label_mask.shape[2]
    if 0 in axis:
        minzidx = int(np.min(mask_voxel_label[0]))
        maxzidx = int(np.max(mask_voxel_label[0])) + 1
    if 1 in axis:
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    if 2 in axis:
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1

    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    bbox_shape = [i[1] - i[0] for i in bbox]
    if np.min(bbox_shape) * 2 * spacing[3 - np.argmin(bbox_shape) - 1] < np.max(bbox_shape) * spacing[
        3 - np.argmax(bbox_shape) - 1]:
        min_axis = np.argmin(bbox_shape)
        pad_len = np.min(bbox_shape)
        bbox[min_axis][0] -= pad_len
        if bbox[min_axis][0] < 0:
            bbox[min_axis][0] = 0
        bbox[min_axis][1] += pad_len
        if bbox[min_axis][1] > nonzero_mask.shape[min_axis]:
            bbox[min_axis][1] = nonzero_mask.shape[min_axis]
    return bbox


def crop(data, seg, crop_size, margins=(0, 0, 0), crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    data_shape = data.shape
    data_dtype = data.dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = seg.shape
        seg_dtype = seg.dtype

    if not isinstance(crop_size, (tuple, list, np.ndarray)):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == dim, '裁剪尺寸维度与图像不符，请使用三维或者二维裁剪'

    if not isinstance(margins, (np.ndarray, tuple, list)):
        if margins is None:
            margins = 0
        margins = [margins] * dim

    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)

    if seg is not None:
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
    else:
        seg_return = None

    for b in range(data_shape[0]):
        if crop_type == 'center':
            lbs = [(data_shape[i + 2] - crop_size[i]) // 2 for i in range(dim)]
        elif crop_type == 'random':
            lbs = [np.random.randint(margins[i], data_shape[i + 2] - crop_size[i] - margins[i]) if data_shape[i + 2] -
                                                                                                   crop_size[i] -
                                                                                                   margins[i] > margins[
                                                                                                       i] else (
                                                                                                                       data_shape[
                                                                                                                           i + 2] -
                                                                                                                       crop_size[
                                                                                                                           i]) // 2
                   for i in range(dim)]
        else:
            raise NotImplementedError('没有这个方法%s' % crop_type)

        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])), abs(min(0, data_shape[d + 2] - (lbs[d] + crop_size[d])))] for d
                                  in range(dim)]

        ubs = [min(lbs[d] + crop_size[d], data_shape[d + 2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_croped = data[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_croped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_croped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_croped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_croped
            if seg_return is not None:
                seg_return[b] = seg_croped

    return data_return, seg_return




def get_bbox_from_mask_b(nonzero_mask, outside_value):
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    return bbox


def get_all_file_paths(folder_path, P='.pkl'):  # 返回一个按文件名排序的文件路径列表(一.npz为后缀)
    file_paths = []
    # 遍历文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹中的所有文件
        for file in files:
            if file.endswith(P):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return sorted(file_paths, key=lambda x: x.split(os.sep)[-1].split('.')[0])


def load_pickle(path):  # 加载 Pickle 文件
    return pickle.load(open(path, 'rb+'))


def save_pickle(obj, path):  # 保存Pickle 文件
    pickle.dump(obj, open(path, 'wb+'))


def resample_data(ori_array, ori_spacing,
                  target_spacing=None, only_z=False):  # 重新采样数据
    # shape c w h d
    # spacing_nnunet = [1.8532123022052305, 1.512973664256994, 1.512973664256994]
    # ori_spacing原始体素间距
    # target_spacing目标体素间距
    spacing = ori_spacing
    if target_spacing is None:
        #target_spacing = [2.2838839, 1.8709113, 1.8709113]
        target_spacing = [1.6, 1.2859375, 1.2859375]
        #target_spacing = [1.8, 0.8, 0.8]
    if only_z:
        target_spacing = [target_spacing[0], ori_spacing[0], ori_spacing[1]]
    ori_shape = ori_array.shape[1:]
    target_shape = [ori_spacing[len(ori_shape) - i - 1] * ori_shape[i] / target_spacing[i] // 1 for i in
                    range(len(ori_shape))]  # lyy  师兄修改后
    reshaped_data = []

    reshaped_ddd = ori_array[0]
    reshaped_data.append(resize(ori_array[0], target_shape, order=3, preserve_range=True)[None])

    reshaped_data.append(resize(ori_array[1], target_shape, order=0, preserve_range=True,
                                anti_aliasing=False)[None])
    #reshaped_data.append(resize(ori_array[-1], target_shape, order=0, preserve_range=True,
    #                            anti_aliasing=False)[None])
    return np.vstack(reshaped_data), target_spacing


def convert(nnunet_npy_paths, CT_paths, label_paths):
    '''

    :param pseudo_paths: the path of pseudo labels, all the file must be ending with '*.nii.gz'.
    :param nnunet_npy_paths: the path of nnunet's basepath/nnUNet_preprocessed/Task098_FLARE2023/nnUNetData_plans_v2.1_stage1, it depends on which of the nnunet data you are using
    :return:
    '''

    # 1. input data convert to npy
    #       converted by nnunet already
    # 2. pseudo label convert into npy
    CT_path_list = get_all_file_paths(CT_paths, '.nii.gz')
    label_path_list = get_all_file_paths(label_paths, '.nii.gz')
    npy_path_list = get_all_file_paths(nnunet_npy_paths)
    print(len(npy_path_list))
    print(len(label_path_list))
    #assert len(npy_path_list) == len(
    #    label_path_list), 'length for pseudo labels must be same as nii files'
    for npy_path, CT_path, label_path in zip(npy_path_list, CT_path_list, label_path_list):
        # load npy

        # if not os.path.exists(npy_path.replace('.npz', '.npy')):  #生产.npz对应的.npy文件路径，并判断是否存在
        #    images = np.load(npy_path, allow_pickle=True)
        # else:
        #    images = np.load(npy_path.replace('.npz', '.npy'), allow_pickle=True)

        # load nii
        # 使用 SimpleITK 库加载 NIfTI 文件（.nii 或 .nii.gz 格式）的图像数据，并将其转换为 NumPy 数组

        file_name = os.path.basename(npy_path)
        dir = os.path.dirname(label_path)
        label_path = os.path.join(dir, file_name[:-4] + '.nii.gz')
        dir1 = os.path.dirname(CT_path)
        CT_path = os.path.join(dir1, file_name[:-4] + '_0000.nii.gz')

        image = sitk.ReadImage(CT_path)
        spacing = image.GetSpacing()
        CT_array = sitk.GetArrayFromImage(image).astype(np.float32)
        label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)

        # data_return, seg_return = crop(CT_array,label_array,128,'random')

        # load info
        properties = load_pickle(npy_path)  # 生产.npz对应的.pkl路径并加载  #lyy 变成.pkl
        if CT_array.shape != label_array.shape:
            label_array = resize(label_array, CT_array.shape, order=0, preserve_range=True, anti_aliasing=False)
        # update the crop bbox
        #ddd = get_bbox_from_mask(CT_array > 0, label_array > 0, spacing, axis=None, outside_value=0, )
        crop_bbox = get_bbox_from_mask_b(CT_array > 0, outside_value=0)  ###########################
        #crop_bbox1 = get_bbox_from_mask_b((label_array) > 0, outside_value=0)
        sli = slice(crop_bbox[0][0], crop_bbox[0][1]), slice(crop_bbox[1][0], crop_bbox[1][1]), slice(crop_bbox[2][0],
                                                                                                      crop_bbox[2][1])

        # crop three arrays
        # properties['crop_bbox'] = crop_bbox  #lyy
        # 通过使用 slice 对象 sli 对 pseudo_array、CT_array 和 label_array 进行切片操作，从而只保留感兴趣的区域。
        CT_array = CT_array[sli]
        label_array = label_array[sli]
        properties['size_after_cropping'] = np.array(label_array.shape)

        # concate
        cropped_data = np.stack((CT_array, label_array), axis=0)

        # resample array
        properties['spacing_after_resampling'] = None  ############lyy
        resampled_data, current_spacing = resample_data(cropped_data, spacing,
                                                        properties['spacing_after_resampling'])
        properties['spacing_after_resampling'] = current_spacing

        # norm one  归一化处理
        ct_array = resampled_data[0].copy()
        if np.max(ct_array) < 1:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            percentile_95 = np.percentile(ct_array, 95)
            percentile_5 = np.percentile(ct_array, 5)
            std = np.std(ct_array)
            mn = np.mean(ct_array)
            ct_array = np.clip(ct_array, a_min=percentile_5, a_max=percentile_95).astype(
                np.float32)  # 将小于 percentile_5 的值设置为 percentile_5，将大于 percentile_95 的值设置为 percentile_95。
            ct_array = (ct_array - mn) / std
        else:
            print("#######################")
            ct_array = np.clip(ct_array, a_min=-160., a_max=240.).astype(np.float32)
            ct_array = (ct_array + 160.) / 400.
        resampled_data[0] = ct_array

        properties['size_after_resampling'] = np.array(resampled_data[0].shape)

        # save to npy
        np.save(npy_path.replace('.pkl', '.npy'), resampled_data)

        print('finish combine')

        # 3. location info add into pkl
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(1234)
        class_locs = {}
        all_classes = range(15)  #14
        #all_classes = range(2)
        for c in all_classes:
            if c == 0: 
                continue
            # kk = np.unique(resampled_data[2])
            # kk1 = np.unique(resampled_data[1])

            # ddd = np.unique(resampled_data[-1])
            # ddd1 = np.unique(resampled_data[-2])
            # all_locs = np.argwhere(resampled_data[-2 if c in resampled_data[-1] else -1] == c)  #原始
            all_locs = np.argwhere(resampled_data[-1] == c)
            #all_locs = np.argwhere(resampled_data[-2 if c in resampled_data[-2] else -1] == c)  # 返回满足指定条件的数组元素的坐标 lyy
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples,
                                     int(np.ceil(len(all_locs) * min_percent_coverage)))  # np.ceil()向上取整

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            print(c, target_num_samples)
            properties['class_locations'] = class_locs
        save_pickle(properties, npy_path)


import os
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_file(npy_path, CT_path, label_path):
    try:
        # load nii
        file_name = os.path.basename(npy_path)
        dir = os.path.dirname(label_path)
        label_path = os.path.join(dir, file_name[:-4] + '.nii.gz')
        dir1 = os.path.dirname(CT_path)
        CT_path = os.path.join(dir1, file_name[:-4] + '_0000.nii.gz')

        image = sitk.ReadImage(CT_path)
        spacing = image.GetSpacing()
        CT_array = sitk.GetArrayFromImage(image).astype(np.float32)
        label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)

        properties = load_pickle(npy_path)

        if CT_array.shape != label_array.shape:
            label_array = resize(label_array, CT_array.shape, order=0, preserve_range=True, anti_aliasing=False)

        crop_bbox = get_bbox_from_mask_b(CT_array > 0, outside_value=0)
        sli = slice(crop_bbox[0][0], crop_bbox[0][1]), slice(crop_bbox[1][0], crop_bbox[1][1]), slice(crop_bbox[2][0],
                                                                                                      crop_bbox[2][1])

        CT_array = CT_array[sli]
        label_array = label_array[sli]
        properties['size_after_cropping'] = np.array(label_array.shape)

        cropped_data = np.stack((CT_array, label_array), axis=0)

        properties['spacing_after_resampling'] = None
        resampled_data, current_spacing = resample_data(cropped_data, spacing, properties['spacing_after_resampling'])
        properties['spacing_after_resampling'] = current_spacing

        ct_array = resampled_data[0].copy()
        if np.max(ct_array) < 1:
            percentile_95 = np.percentile(ct_array, 95)
            percentile_5 = np.percentile(ct_array, 5)
            std = np.std(ct_array)
            mn = np.mean(ct_array)
            ct_array = np.clip(ct_array, a_min=percentile_5, a_max=percentile_95).astype(np.float32)
            ct_array = (ct_array - mn) / std
        else:
            ct_array = np.clip(ct_array, a_min=-160., a_max=240.).astype(np.float32)
            ct_array = (ct_array + 160.) / 400.
        resampled_data[0] = ct_array

        properties['size_after_resampling'] = np.array(resampled_data[0].shape)

        np.save(npy_path.replace('.pkl', '.npy'), resampled_data)

        num_samples = 10000
        min_percent_coverage = 0.01
        rndst = np.random.RandomState(1234)
        class_locs = {}
        all_classes = range(15)
        for c in all_classes:
            if c == 0:
                continue
            all_locs = np.argwhere(resampled_data[-1] == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            properties['class_locations'] = class_locs

        save_pickle(properties, npy_path)
        print(f'Finished processing {npy_path}')
    except Exception as e:
        print(f'Error processing {npy_path}: {e}')


def convert11(nnunet_npy_paths, CT_paths, label_paths):


    CT_path_list = get_all_file_paths(CT_paths, '.nii.gz')
    label_path_list = get_all_file_paths(label_paths, '.nii.gz')
    npy_path_list = get_all_file_paths(nnunet_npy_paths)

    #assert len(npy_path_list) == len(label_path_list), 'length for pseudo labels must be same as nii files'

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, npy_path, CT_path, label_path) for npy_path, CT_path, label_path in
                   zip(npy_path_list, CT_path_list, label_path_list)]

        for future in as_completed(futures):
            future.result()  # Ensure exceptions are raised

    print('All files have been processed.')


if __name__ == '__main__':
    argps = argparse.ArgumentParser()
    # -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
    argps.add_argument('nnunet_npy_path', type=str,
                       help='path for npy from nnunet, it must be convert by nnunet first!')
    argps.add_argument('image_tr_path', type=str, help='path for npy from FLARE imageTr')
    argps.add_argument('label_tr_path', type=str, help='path for npy from FLARE labelTr')

    arg_s = argps.parse_args()

    convert(nnunet_npy_paths=arg_s.nnunet_npy_path, CT_paths=arg_s.image_tr_path,
            label_paths=arg_s.label_tr_path)

# E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\pseudoTr E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUnet_preprocessed\Task096_FLARE\nnUNetData_plans_v2.1_stage1 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\imagesTr E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\labelsTr