import numpy as np
from torch import nn
import torch
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple, List
from torch import Tensor
import math

ACCURACY = "accuracy"
DICE = "dice"
F1 = "f1"
SENSITIVITY = "SENSITIVITY"
SPECIFICITY = "SPECIFICITY"
PRECISION = "PRECISION"
JS = "js"
EVALUATIONS = [ACCURACY, DICE, F1, SENSITIVITY, SPECIFICITY, PRECISION, JS]


class Valid_utils():

    def __init__(self, net, num_classes, patch_size):
        self.num_classes = num_classes
        self.net = net
        # self.net.cpu()

        self.patch_size = patch_size
        pass

    def predict_3D(self, data_x, do_mirror: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   step_size: Tuple[float, ...] = (7/8, 7/8, 7/8), patch_size: Tuple[int, ...] = None,
                   use_gaussian: bool = True,
                   ) -> Tuple[torch.Tensor, np.ndarray]:

        '''

        :param data_x:
        :param do_mirror:
        :param mirror_axes:
        :param step_size:
        :param patch_size:
        :param use_gaussian:
        :return: predict shape b x y z
        '''
        if patch_size is None:
            patch_size = self.patch_size
        data_x, slices_xyz = self.maybe_pad(image=data_x, new_shape=patch_size, mode='constant', kwargs=None,
                                            return_slicer=True, shape_must_be_divisible_by=None)
        data_x = torch.from_numpy(data_x).cuda()
        shape = data_x.shape[1:]
        print(shape)
        #data_x = data_x.cuda()
        do_crop = False
        resizer = None
        '''
        #原始方案，判断有没有器官标签
        if shape[0] > int(np.ceil(patch_size[0] + 3 / 4 * patch_size[0])):
        #if 1==0:
            print("do crop")
            do_crop = True
            target_step_sizes_in_voxels = [i * j for i, j in zip((1, 1, 1), patch_size)]
            num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                         zip(shape, target_step_sizes_in_voxels, patch_size)]
            dd = int(np.ceil((shape[0] - patch_size[0]) / patch_size[0]) + 1)
            if (patch_size[1] >= shape[1]):
                lb_x = 0
                ub_x = shape[1]
            else:
                s = shape[1] - patch_size[1]
                lb_x = math.ceil(shape[1] / 2) - math.ceil(patch_size[1] / 2) + math.floor(s / 3)
                ub_x = lb_x + patch_size[1]
            if (patch_size[2] > shape[2]):
                lb_y = 0
                ub_y = shape[2]
            else:
                s = shape[2] - patch_size[2]
                lb_y = math.ceil(shape[2] / 2) - math.ceil(patch_size[2] / 2) + math.floor(s / 3)
                ub_y = lb_y + patch_size[2]
            lb_z = 0
            is_start = False
            low = 0
            high = shape[0]
            for i in range(dd):
                i += 1
                ub_z = lb_z + patch_size[0]
                if i == dd:
                    ub_z = shape[0]
                    lb_z = shape[0] - patch_size[0]
                x = data_x[None, :, int(lb_z): int(ub_z), int(lb_x): int(ub_x), int(lb_y): int(ub_y)]
                pred = self.net(x)
                pred = pred[0]

                pred_label = torch.argmax(pred[:, :, :, :], 0)
                is_half_empty = torch.sum(pred_label) < 600  # 500
                if is_half_empty == True:
                    if is_start == False:
                        if torch.sum(pred_label[int((2 / 4) * patch_size[0]):, :, :]) < 300:
                            lb_z += patch_size[0]
                        else:
                            lb_z += patch_size[0] / 2

                        continue
                    else:
                        if torch.sum(pred_label[:int((2 / 4) * patch_size[0]), :, :]) < 300:
                            lb_z += patch_size[0] / 2
                        high = lb_z
                        break

                else:
                    if is_start == False:
                        is_start = True
                    if is_start == True:
                        low = lb_z - math.floor(patch_size[0] / 2)
                        
                    lb_z += patch_size[0]
        '''

        if shape[0] > int(np.ceil(patch_size[0] + 3 / 4 * patch_size[0])):
            # if 1==0:
            print("do crop")
            do_crop = True
            target_step_sizes_in_voxels = [i * j for i, j in zip((1, 1, 1), patch_size)]
            num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                         zip(shape, target_step_sizes_in_voxels, patch_size)]
            dd = int(np.ceil((shape[0] - patch_size[0]) / patch_size[0]) + 1)
            if (patch_size[1] >= shape[1]):
                lb_x = 0
                ub_x = shape[1]
            else:
                lb_x = 0
                ub_x = lb_x + patch_size[1]
            if (patch_size[2] > shape[2]):
                lb_y = 0
                ub_y = shape[2]
            else:
                lb_y = 0
                ub_y = lb_y + patch_size[2]
            lb_z = 0
            is_start = False
            low = 0
            high = shape[0]
            for i in range(dd):
                i += 1
                ub_z = lb_z + patch_size[0]
                if i == dd:
                    ub_z = shape[0]
                    lb_z = shape[0] - patch_size[0]
                x = data_x[None, :, int(lb_z): int(ub_z), int(lb_x): int(ub_x), int(lb_y): int(ub_y)]
                pred = self.net(x)
                pred = pred[0]
                pred_label = torch.argmax(pred[:, :, :, :], 0)

                # 获取肝脏区域的边界框
                indices = torch.where(pred_label==1)

                if len(indices[0]) < 1000:  # 确保找到标签为1的区域
                    lb_z += patch_size[0]
                    continue
                else:
                    low = lb_z - math.floor(patch_size[0] * 3 / 4)
                    high = lb_z + math.floor(patch_size[0] * 2)
                    break
            if low < 0:
                low = 0
            if high > shape[0]:
                high = shape[0]
            if (high - low) < patch_size[0]:
                k = patch_size[0] -(high - low)
                if (patch_size[0] - high) > k:
                    high = high + k
                else:
                    low = low - k
            resizer = (slice(0, 1), slice(int(low), int(high)), slice(0, shape[1]), slice(0, shape[2]))
            print(resizer)


            data_x = data_x[resizer]

        #data_x, slices_xyz = self.maybe_pad(image=data_x, new_shape=patch_size, mode='constant', kwargs=None,
        #                                    return_slicer=True, shape_must_be_divisible_by=None)
        #data_x = torch.from_numpy(data_x).cuda()
        data_shape = data_x.shape[1:]
        assert len(data_shape) == 3, 'shape 不对, 应该为 b, x, y, z'
        print(data_shape)
        strides = self._compute_steps_for_sliding_window(patch_size=patch_size, image_size=data_shape,
                                                         step_size=step_size)
        print(strides)
        gaussian_map = self._get_gaussian(patch_size=patch_size)
        gaussian_map = torch.from_numpy(gaussian_map).float().cuda()

        aggregated_result = torch.zeros([self.num_classes] + list(data_shape), dtype=torch.float32)
        aggregated_result_mask = torch.zeros([self.num_classes] + list(data_shape), dtype=torch.float32)
        # try:
        do_mirror = False
        is_empty = False
        sure = False
        times = 0
        num_window = 0
        for z in strides[0]:
            lb_z = z
            ub_z = z + patch_size[0]
            '''
            # 对该 xy 平面共有 9 个 window， 对正中央的 1 个 window 进行推理
            lb_z = z
            ub_z = z + patch_size[0]
            if sure == True:
                times += 1
                if times > 2:
                    sure = False
            else:
            # 看正中的 window
                if (patch_size[1] >= data_shape[1]):
                    lb_x = 0
                    ub_x = data_shape[1]
                else:
                    lb_x = math.ceil(data_shape[1] / 2) - math.ceil(patch_size[1] / 2)
                    ub_x = lb_x + patch_size[1]
                if (patch_size[2] > data_shape[2]):
                    lb_y = 0
                    ub_y = data_shape[2]
                else:
                    lb_y = math.ceil(data_shape[2] / 2) - math.ceil(patch_size[2] /2)
                    ub_y = lb_y + patch_size[2]
                result = self.do_mirror_maybe(data_x[None, :, lb_z: ub_z, lb_x: ub_x, lb_y: ub_y], do_mirror,
                                              mirror_axes, gaussian_map)
                num_window += 1
                #is_empty = torch.sum(torch.argmax(result, 0) > 0) < 1000
                is_half_empty = torch.sum(torch.argmax(result[:, :, int((3/4) * patch_size[0]):, :], 0) > 0) < 300  #500
                if is_half_empty  == True:
                    print("跳。。。。。。。。。。。。。。。。。。。")
                    continue
                else:
                    sure = True
            '''


            for x in strides[1]:
                lb_x = x
                ub_x = x + patch_size[1]
                for y in strides[2]:
                    lb_y = y
                    ub_y = y + patch_size[2]

                    result = self.do_mirror_maybe(data_x[None, :, lb_z: ub_z, lb_x: ub_x, lb_y: ub_y], do_mirror,
                                                  mirror_axes, gaussian_map)
                    aggregated_result[:, lb_z: ub_z, lb_x: ub_x, lb_y: ub_y] += result.cpu()
                    aggregated_result_mask[:, lb_z: ub_z, lb_x: ub_x, lb_y: ub_y] += gaussian_map.cpu()
                    del result
        # except RuntimeError:
        #     print('RuntimeError')

        # 通道数也加上
        slices_xyz = tuple([slice(0, aggregated_result.shape[i]) for i in
                            range(len(aggregated_result.shape) - (len(slices_xyz) - 1))] + slices_xyz[1:])
        if do_crop == False:
            aggregated_result = aggregated_result[slices_xyz]
            aggregated_result_mask = aggregated_result_mask[slices_xyz]

        aggregated_result /= aggregated_result_mask

        # aggregated_result = aggregated_result.argmax(0)
        del gaussian_map
        del aggregated_result_mask
        del data_x
        return aggregated_result, shape, resizer, do_crop, slices_xyz

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> \
            List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        #assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        #target_step_sizes_in_voxels = [i * 0.5 for i in patch_size]
        target_step_sizes_in_voxels = [i * j for i, j in zip(step_size,patch_size)]
        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                     zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def maybe_pad(self, image, new_shape, mode='constant', kwargs=None, return_slicer=False,
                  shape_must_be_divisible_by=None):
        image = image.cpu().numpy()
        if kwargs is None:
            kwargs = {'constant_values': 0}

        if new_shape is not None:
            old_shape = np.array(image.shape[-len(new_shape):])
        else:
            assert shape_must_be_divisible_by is not None
            assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
            new_shape = image.shape[-len(shape_must_be_divisible_by):]
            old_shape = new_shape

        num_axes_nopad = len(image.shape) - len(new_shape)

        new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

        if not isinstance(new_shape, np.ndarray):
            new_shape = np.array(new_shape)

        if shape_must_be_divisible_by is not None:
            if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
                shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
            else:
                assert len(shape_must_be_divisible_by) == len(new_shape)

            for i in range(len(new_shape)):
                if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                    new_shape[i] -= shape_must_be_divisible_by[i]

            new_shape = np.array(
                [new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in
                 range(len(new_shape))])

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

        if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
            res = np.pad(image, pad_list, mode, **kwargs)
        else:
            res = image

        if not return_slicer:
            return res
        else:
            pad_list = np.array(pad_list)
            pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
            slicer = list(slice(*i) for i in pad_list)
            return res, slicer
        # res = None
        # if kwargs is None:
        #     kwargs = {'constant_values': 0}
        # if patch_size is not None:
        #     old_shape = np.array(image.shape[-(len(patch_size)):])
        #
        # len_no_pad = len(image.shape) - len(patch_size)
        #
        # patch_size = np.array(patch_size)
        #
        # new_shape = [max(patch_size[i], old_shape[i]) for i in range(len(patch_size))]
        #
        # difference = new_shape - old_shape
        #
        # pad_l = difference // 2
        # pad_u = difference // 2 + difference % 2
        #
        # # pad_ = [[0, 0]] * len_no_pad + [list(*i) for i in zip(pad_l, pad_u)]
        # pad_image = np.array([[0, 0]] * len_no_pad + [list(i) for i in zip(pad_l, pad_u)])
        # pad_ = tuple([j for i in zip(pad_l[::-1], pad_u[::-1]) for j in i])
        #
        # if not all(list(pad_)):
        #     res = torch.nn.functional.pad(image, pad_, mode, value=0)
        # else:
        #     res = image
        #
        # pad_image[:, 1] = np.array(res.shape) - pad_image[:, 1]
        # slices_xyz = list(slice(*i) for i in pad_image)
        # return res, slices_xyz

    def do_mirror_maybe(self, x, do_mirror=True, mirror_axes=(0, 1, 2), mult_gaussian_map=None):
        # x shape b c x y z 
        result = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float)
        # x = torch.from_numpy(x).float()
        x = x.cuda()
        # mult_gaussian_map = torch.from_numpy(mult_gaussian_map).float()
        # mult_gaussian_map = mult_gaussian_map.cuda()
        result = result.cuda()

        if do_mirror:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.net(x)
                result += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.net(torch.flip(x, (4,)))
                result += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.net(torch.flip(x, (3,)))
                result += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.net(torch.flip(x, (4, 3)))
                result += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.net(torch.flip(x, (2,)))
                result += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.net(torch.flip(x, (4, 2)))
                result += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.net(torch.flip(x, (3, 2)))
                result += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.net(torch.flip(x, (4, 3, 2)))
                result += 1 / num_results * torch.flip(pred, (4, 3, 2))
        if mult_gaussian_map is not None:
            result[:, :] *= mult_gaussian_map
        del x
        return result[0]

    def _valid(self, loss_function):
        losss = list()
        EVALUATIONS = [DICE]
        table = np.zeros((self.num_classes, len(EVALUATIONS)))
        mask = np.ones((self.num_classes, 1))
        evaluations = None
        for k, v in self.val_dataset.items():
            npz = np.load(v)
            image, label = npz['data'], npz['seg']
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image).float()
            # shape x,y,z
            predict = self.predict_3D(image, patch_size=self.patch_size, do_mirror=True)
            label = torch.from_numpy(label).float().unsqueeze(0).cuda()
            predict = predict.unsqueeze(0)
            losss.append(loss_function(predict.cup(), label.cpu()))
            predict = predict.argmax(1)
            class_unique = torch.unique(label)
            for i in range(self.num_classes):
                if i == 0:
                    continue
                if i in class_unique:
                    evals = self.GetEvaluation(predict == float(i), label == float(i), EVALUATIONS)
                    mask[i] += 1
                else:
                    evals = [0 for _ in range(len(EVALUATIONS))]
                    # mask[i] -= 1
                table[i, :] += evals
        evaluations = table / mask
        return np.mean(evaluations[1:, -1]), np.mean(losss)

    @staticmethod
    def GetEvaluation(SR: Tensor, GT: Tensor, EVALS: list = EVALUATIONS):
        SR = SR.type(torch.int)
        GT = GT.type(torch.int)
        TP = ((SR == 1) * 1 + (GT == 1) * 1) == 2
        FN = ((SR == 0) * 1 + (GT == 1) * 1) == 2
        TN = ((SR == 0) * 1 + (GT == 0) * 1) == 2
        FP = ((SR == 1) * 1 + (GT == 0) * 1) == 2
        acc = None
        dice = None
        f1 = None
        sensitivity = None
        specificity = None
        precision = None
        js = None
        return_eval = list()
        for eval in EVALS:
            assert eval in EVALUATIONS
            if eval == ACCURACY:
                acc = float(torch.sum(TP + TN)) / \
                      (float(torch.sum(TP + FN + TN + FP)) + 1e-6)
                return_eval.append(acc)

            if eval == SENSITIVITY:
                sensitivity = float(torch.sum(TP)) / \
                              (float(torch.sum(TP + FN)) + 1e-6)
                return_eval.append(sensitivity)

            if eval == SPECIFICITY:
                specificity = float(torch.sum(TN)) / \
                              (float(torch.sum(TN + FP)) + 1e-6)
                return_eval.append(specificity)

            if eval == PRECISION:
                precision = float(torch.sum(TP)) / \
                            (float(torch.sum(TP + FP)) + 1e-6)
                return_eval.append(precision)

            if eval == F1:
                SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
                PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
                f1 = 2 * SE * PC / (SE + PC + 1e-6)
                return_eval.append(f1)

            if eval == JS:
                Inter = torch.sum((SR + GT) == 2)
                Union = torch.sum((SR + GT) >= 1)
                js = float(Inter) / (float(Union) + 1e-6)
                return_eval.append(js)

            if eval == DICE:
                Inter = torch.sum((SR + GT) == 2)
                dice = float(2 * Inter) / \
                       (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)
                return_eval.append(dice)

        return return_eval


if __name__ == '__main__':
    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (162, 529, 529), 0.5))
    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (162, 529, 529), 1))
    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (162, 529, 529), 0.1))

    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (60, 448, 224), 1))
    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (60, 448, 224), 0.5))

    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (30, 224, 224), 1))
    # print(Valid_utils._compute_steps_for_sliding_window((30, 224, 224), (30, 224, 224), 0.125))
    # import SimpleITK as sitk
    # import os
    # from save_nii import resample_image_v2
    # # print(Valid_utils._compute_steps_for_sliding_window((123, 54, 123), (246, 162, 369), 0.25))
    # paths = '/nas/ljc/datasets/Subtask1/TrainImage/train_0358_0000.npz'
    # paths_nii = '/nas/ljc/datasets/Subtask1/TrainImage/train_0358_0000.nii.gz'
    # array = np.load('/nas/ljc/datasets/Subtask1/TrainImage/train_0358_0000.npz')['seg']
    # array = np.expand_dims(array, axis=0)
    # image = sitk.ReadImage(paths_nii)
    # ori = image.GetOrigin()
    # spac = image.GetSpacing()
    # dir_ = image.GetDirection()
    # from dataloading import DataSet, base_path
    # s = Valid_utils(net=net, num_classes=5, dataset=DataSet(base_path))
    # with torch.no_grad():
    #     result = s.predict_3D(data_x=array, do_mirror=True, patch_size=(128,128,128))
    #     # result = result.argmax(1)
    #     result_image = sitk.GetImageFromArray(np.array(result).astype(np.int32))
    #     result_image.SetDirection(dir_)
    #     result_image.SetOrigin(ori)
    #     result_image.SetSpacing([0.688, 0.688, 2])
    #     result_image = resample_image_v2(result_image, spac, is_label=True)
    #     sitk.WriteImage(result_image, os.path.dirname(__file__) + '/guassian_map.nii.gz')
    #     pass
    pass
