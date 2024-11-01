import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pickle
from skimage.transform import resize
from scipy.ndimage import binary_dilation
from skimage.measure import label
import SimpleITK as sitk


def check_nii(r0, r1, properties):
    if isinstance(properties, str):
        spacing, origin, direction, size, unique, image_dtype, label_dtype, bbox = load_pickle(properties)
    if isinstance(properties, list):
        spacing = properties

    # data_shape b c w h d
    r0 = r0.reshape(r0.shape[1:])
    r1 = r1.reshape(r1.shape[1:])
    after_itk_image = sitk.GetImageFromArray(r0.squeeze().numpy())
    after_itk_image.SetSpacing(spacing)
    after_itk_image.SetOrigin(origin)
    after_itk_image.SetDirection(direction)
    after_itk_label = sitk.GetImageFromArray(r1.squeeze().numpy())
    after_itk_label.SetSpacing(spacing)
    after_itk_label.SetOrigin(origin)
    after_itk_label.SetDirection(direction)

    sitk.WriteImage(after_itk_image, 'after_itk_image.nii.gz')
    sitk.WriteImage(after_itk_label, 'after_itk_label.nii.gz')
    pass


def get_bbox_from_mask_move_bed(image_array, label_mask, spacing):
    # dilation
    selem = np.ones((3, 3, 3), dtype=bool)
    labels = binary_dilation(image_array, selem)
    # connection
    labels = label(label_image=labels, connectivity=2)

    connected_components = dict()
    for i in np.unique(labels):
        connected_components[i] = (np.sum(labels == i))

    retv = dict(sorted(connected_components.items(), key=lambda k: -k[1]))
    # max connection
    keep_labels = list(retv.keys())
    keep_labels = keep_labels[:2]
    for i in retv.keys():
        if i not in keep_labels:
            labels[labels == i] = 0
    labels[labels != 0] = 1

    bbox = get_bbox_from_mask(labels, label_mask, spacing)
    return bbox


# def get_bbox_from_mask_move_bed(image_array, label_mask, spacing):
#     # dilation
#     kernel = skimage.morphology.ball(2)
#     label_np = skimage.morphology.erosion(image_array, kernel)
#
#     region_volume = OrderedDict()
#     # cal connection in image
#     label_map, numregions = label(label_np == 1, return_num=True)
#     region_volume['num_region'] = numregions
#     total_volume = 0
#     max_region = 0
#     max_region_flag = 0
#     for l in range(1, numregions + 1):
#         region_volume[l] = np.sum(label_map == l)  # * volume_per_volume
#         if region_volume[l] > max_region:
#             max_region = region_volume[l]
#             max_region_flag = l
#         total_volume += region_volume[l]
#     post_label_np = label_np.copy()
#     post_label_np[label_map != max_region_flag] = 0
#     post_label_np[label_map == max_region_flag] = 1
#
#     kernel = skimage.morphology.ball(2)
#     img_dialtion = skimage.morphology.dilation(post_label_np, kernel)
#     bbox = get_bbox_from_mask(img_dialtion, label_mask, spacing)
#     return bbox


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


def resample_data(ori_array, ori_spacing,
                  target_spacing=None, only_z=False):
    # shape c w h d
    # spacing_nnunet = [1.8532123022052305, 1.512973664256994, 1.512973664256994]
    if target_spacing is None:
        target_spacing = [2.2838839, 1.8709113, 1.8709113]
    if only_z:
        target_spacing = [target_spacing[0], ori_spacing[0], ori_spacing[1]]
    ori_shape = ori_array.shape[1:]
    target_shape = [ori_spacing[len(ori_shape) - i - 1] * ori_shape[i] / target_spacing[i] // 1 for i in
                    range(len(ori_shape))]
    reshaped_data = []
    reshaped_data.append(resize(ori_array[0].astype(np.float32), target_shape, order=3)[None])
    reshaped_data.append(resize(ori_array[1].astype(np.float32), target_shape, order=0, preserve_range=True,
                                anti_aliasing=False)[None])
    reshaped_data.append(resize(ori_array[-1].astype(np.float32), target_shape, order=0, preserve_range=True,
                                anti_aliasing=False)[None])
    return np.vstack(reshaped_data), target_spacing


def load_pickle(path):
    return pickle.load(open(path, 'rb+'))


def save_pickle(obj, path):
    pickle.dump(obj, open(path, 'wb+'))


def isfile(path):
    return os.path.exists(path)


def create_matrix_rotation_x_3d(angle, rot_matrix):
    rotation_x = np.array(
        [[1, 0, 0],
         [0, np.cos(angle), -np.sin(angle)],
         [0, np.sin(angle), np.cos(angle)]]
    )
    if rot_matrix is None:
        return rotation_x
    return np.dot(rot_matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, rot_matrix):
    rotation_y = np.array(
        [[np.cos(angle), 0, np.sin(angle)],
         [0, 1, 0],
         [-np.sin(angle), 0, np.cos(angle)]]
    )
    if rot_matrix is None:
        return rotation_y
    return np.dot(rot_matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, rot_matrix):
    rotation_z = np.array(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle), np.cos(angle), 0],
         [0, 0, 1]]
    )
    if rot_matrix is None:
        return rotation_z
    return np.dot(rot_matrix, rotation_z)


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    # coords shape 3 patch_size
    # rot_matrix 旋转矩阵 shape 3 * 3
    # coords.reshape(3, -1).T * rot_matrix => shape:(-1, 3) => transpose => reshape
    # 这个*为什么就可以做到变化呢，可能是旋转矩阵的意义？
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, 'random')


def center_crop(data, crop_size, seg=None, margins=None):
    return crop(data, seg, crop_size, margins, 'center')


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


def get_range_val(value, rnd_type='uniform'):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == 'uniform':
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == 'normal':
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError('value 必须是长度为1或2的list/tuple')
        return n_val
    else:
        return value


def check_slice(image_arr):
    assert len(image_arr.shape) == 2
    plt.imshow(image_arr, cmap='gray')
    plt.show()
    plt.close()


def max_compoment(predict_array):
    predict_post = np.zeros_like(predict_array, dtype=predict_array.dtype)
    for organ in np.unique(predict_array):
        if organ == 0:
            continue
        copy = predict_array.copy()
        copy[predict_array != organ] = 0
        selem = np.ones((3, 3, 3), dtype=bool)
        # 一次膨胀 11个连通分量
        labels = binary_dilation(copy, selem)
        # 两次膨胀 10个连通分量
        # copy = morphology.dilation(copy, selem)

        labels = label(label_image=labels, connectivity=2)
        # print(np.unique(labels))
        connected_components = dict()
        for i in np.unique(labels):
            connected_components[i] = (np.sum(labels == i))

        retv = dict(sorted(connected_components.items(), key=lambda k: -k[1]))
        keep_labels = list(retv.keys())
        if organ == 14:
            keep_labels = keep_labels[:]
        else:
            keep_labels = keep_labels[:2]
        for i in retv.keys():
            if i not in keep_labels:
                labels[labels == i] = 0
        labels[labels != 0] = 1
        labels *= copy
        labels = labels.astype(predict_post.dtype)
        # predict_array[predict_array == organ] = 0
        predict_post += labels
    return predict_post
