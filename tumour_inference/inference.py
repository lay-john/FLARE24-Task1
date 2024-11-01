from nnunet.network_architecture.generic_UNet import Generic_UNet
from data_convert import load_pickle, resize, get_bbox_from_mask_b
import SimpleITK as sitk
from predict import Valid_utils
import os
from scipy.ndimage import binary_fill_holes
import numpy as np
import threading
import time
import queue
import concurrent.futures
import torch.nn as nn
import torch
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from skimage.measure import label
import argparse
from scipy import ndimage
import torch.nn.functional as F


def inner_outer_edge_diff(img_np_255, tumor_mask, not_edge=None, edge_diff_when_dont_find=1000):
    # 计算所有连通域的标签
    labeled_array, num_features = ndimage.label(tumor_mask)
    edge_diffs = []
    if num_features == 0:
        print("没有找到连通域")
        return tumor_mask > 0, edge_diffs # 返回原始的 tumor_mask

    # 存储每个连通区域的边缘差异


    # 处理每个连通区域
    for i in range(1, num_features + 1):
        # 获取当前连通区域
        current_region = (labeled_array == i)

        # 计算当前连通区域的体素数量
        num_voxels = np.sum(current_region)

        # 跳过体素数量小于 100 的连通区域
        if num_voxels < 100:
            tumor_mask[labeled_array == i] = 0
            continue

        # 对当前连通区域进行腐蚀和膨胀
        erosion_tumor = ndimage.binary_erosion(current_region, iterations=4)
        expand_tumor = ndimage.binary_dilation(current_region, iterations=4)

        # 计算当前连通区域的内部和外部边缘
        in_edge = current_region * (erosion_tumor == 0)
        out_edge = expand_tumor * (current_region == 0)

        # 如果提供了 not_edge，则将其应用于 in_edge 和 out_edge
        if not_edge is not None:
            in_edge *= (not_edge == 0)
            out_edge *= (not_edge == 0)

        # 计算边缘差异（如果找到边缘区域）
        if np.max(in_edge) > 0 and np.max(out_edge) > 0:
            in_edge_mean = np.mean(img_np_255[in_edge == 1])
            out_edge_mean = np.mean(img_np_255[out_edge == 1])
            edge_diff = int(in_edge_mean - out_edge_mean)
        else:
            edge_diff = edge_diff_when_dont_find

        # 将每个连通区域的边缘差异添加到列表中
        edge_diffs.append(edge_diff)

        # 根据边缘差异的绝对值更新 tumor_mask
        if  abs(edge_diff) < 15:
            tumor_mask[labeled_array == i] = 0
    print(edge_diffs)
    return tumor_mask, edge_diffs

def convert_to_gray(value, old_min=-160, old_max=240, new_min=0, new_max=255):
    normalized_value = (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return normalized_value

def keep_largest_connected_component(binary_image):
    # 1. 计算所有连通域的标签
    labeled_array, num_features = ndimage.label(binary_image)

    if num_features == 0:
        print("没有找到连通域")
        return binary_image

    # 2. 计算每个连通域的体素数量
    sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))

    # 3. 找到最大连通域的标签
    max_label = np.argmax(sizes)

    # 4. 创建一个新的二值图像，只保留最大连通域
    largest_component = (labeled_array == max_label)

    return largest_component


def resample_data(ori_array, ori_spacing,
                  target_spacing=None, only_z=False):
    # shape c w h d
    # spacing_nnunet = [1.8532123022052305, 1.512973664256994, 1.512973664256994]
    print("进入rasample")
    if target_spacing is None:
        target_spacing = [2.2838839, 1.8709113, 1.8709113]
    if only_z:
        target_spacing = [target_spacing[0], ori_spacing[0], ori_spacing[1]]
    ori_shape = ori_array[0].shape
    #target_shape = [ori_spacing[i] * ori_shape[i] / target_spacing[i] // 1 for i in range(len(ori_shape))]    #原始

    reshaped_data = []
    data = []
    ori_array = ori_array[0]

    print("ffffffffffffffffff")
    target_shape = [int(ori_spacing[len(ori_shape) - i - 1] * ori_shape[i] / target_spacing[i] // 1) for i in
                    range(len(ori_shape))]  # lyy  师兄修改后
    ########################
    ori_array = ori_array.astype(np.float32)
    data_torch = torch.tensor(ori_array).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    del ori_array
    resized_data = F.interpolate(data_torch, size=target_shape, mode='trilinear', align_corners=False)
    del data_torch
    resized_data = resized_data.squeeze(0).cpu().numpy().astype(np.float32)
    reshaped_data.append(resized_data)
    del resized_data



        #reshaped_data.append((resize(ori_array, target_shape, order=3, preserve_range=True)[None]).astype(np.float32))
        #del ori_array
    #reshaped_data.append(resize(ori_array[0], target_shape, order=3, preserve_range=True)[None])
    #reshaped_data.append(resize(ori_array[1], target_shape, order=0, preserve_range=True,
    #                            anti_aliasing=False)[None])
    #reshaped_data.append(resize(ori_array[-1], target_shape, order=0, preserve_range=True,
    #                            anti_aliasing=False)[None])
    print("出rasample")
    return np.vstack(reshaped_data), target_spacing



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
        if len(keep_labels) > 4:
            keep_labels = keep_labels[:4]
        for i in retv.keys():
            if i not in keep_labels:
                labels[labels == i] = 0
        labels[labels != 0] = 1
        labels *= copy
        labels = labels.astype(predict_post.dtype)
        # predict_array[predict_array == organ] = 0
        predict_post += labels
    return predict_post





class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
            module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class my_Trainner():
    def __init__(self):
        self.base_num_features = self.num_classes = self.net_num_pool_op_kernel_sizes = self.conv_per_stage \
            = self.net_conv_kernel_sizes = None
        self.stage = 1
        self.network = None
        self.initial_par()

    def initial_par(self):
        #plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16 = \
        #    load_pickle(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final_checkpoint.model.pkl')
        #                )[
        #        'init']
        plans = load_pickle(os.path.join(checkpoint, 'nnUNetPlansv2.1_plans_3D.pkl'))
        stage_plans = plans['plans_per_stage'][1]
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.base_num_features = plans['base_num_features']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            print(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

        if 'conv_kernel_sizes' not in stage_plans.keys():
            print(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

    def network_initial(self):
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        '''
        self.conv_per_stage = 1
        self.stage_num = 4
        self.base_num_features = 16
        self.max_num_features = 256

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]
        '''
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        #self.net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
        self.network = Generic_UNet(1, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        print(self.network)
    def load_best_checkpoint(self, fname):
        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        curr_state_dict_keys = list(self.network.state_dict().keys())
        new_state_dict = OrderedDict()
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)  # todo jichao params not match

    def get_network(self, fname):
        self.network_initial()
        self.load_best_checkpoint(
            fname=fname)
        return self.network


class MuiltThreadDataGenerator(object):
    def __init__(self, iter, produce_queue_number,save_path) -> None:
        self.iter = iter
        self.produce_queue_number = produce_queue_number
        self.output_queue = queue.Queue(1)
        self.save_path = save_path

    def _ini(self):
        process_thread = threading.Thread(target=self.process_data_thread)
        process_thread.start()
        self.predict_model()
        process_thread.join()
        return

    def preprocess_1(self, image_array, predict_final, ori_spacing):
        #identity = nii_path.split('/')[-1].split('_0000.nii.gz')[0]   #pre

        # crop array
        pp = image_array > 50
        pp = keep_largest_connected_component(pp)
        bbox = get_bbox_from_mask_b(pp, outside_value=0)
        del pp
        #bbox = get_bbox_from_mask_b(nonzero_mask, 0)
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        cropped_data = np.stack((image_array[resizer], np.zeros_like(image_array[resizer])), 0)
        del image_array
        cropped_shape = cropped_data.shape[1:]
        # resample array
        #target_spacing = [2.5, 0.859375, 0.859375]

        target_spacing = [2.5, 2.00922773, 2.00922773]
        #target_spacing = [1., 1., 1.]
        #target_spacing = [1., 0.80371094, 0.80371094]

        #target_spacing = [1., 0.78320301, 0.78320301]

        resampled_data, _ = resample_data(cropped_data, np.array(ori_spacing), target_spacing=target_spacing,
                                          only_z=False)
        ct_array = resampled_data[0].copy()

        # norm to one
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
        ori = torch.from_numpy(resampled_data[0:1])
        del resampled_data
        data = [ori, predict_final, resizer, cropped_shape, ori_spacing]
        del ori
        predict_final = self.predict_model_1(data)
        del data
        return predict_final



    def predict_model_1(self, data):

        start = time.time()
        # nii_path = '/home/ljc/code/FLARE2023/data/Task098_FLARE2023/imagesTs/FLARE23Ts_0001.nii.gz'
        ori, predict_final_shape, resizer, cropped_shape, ori_spacing = data
        torch.cuda.empty_cache()
        ori = ori.cuda().float()
        # predict
        s = Valid_utils(model, 2, patch_size)

        predict, mask = s.predict_3D(ori, do_mirror=False, mirror_axes=(0, 1, 2))

        del ori
        del mask
        # predict = predict.softmax(0).detach().cpu().squeeze().numpy()
        # save_name = os.path.join(self.save_path, '%s.npy' % identity)
        # np.save(save_name, predict)

        predict = predict.softmax(0).argmax(0).detach().cpu().squeeze().numpy()
        # predict = predict * mask
        # predict = max_compoment(predict)
        # re resample
        predict_resample = resize(predict, cropped_shape, order=0, preserve_range=True, anti_aliasing=False)
        del predict
        predict_final = np.zeros(predict_final_shape, dtype=np.uint8)
        predict_final[resizer] = predict_resample
        del predict_resample
        return predict_final





    def preprocess(self, nii_path):
        identity = nii_path.split('/')[-1].split('_0000.nii.gz')[0]   #pre

        #ss = nii_path.replace('\\', '/')
        #identity = ss.split('/')[-1].split('_0000.nii.gz')[0]
        image = sitk.ReadImage(nii_path)
        ori_spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        ori_size = np.array(image.GetSize())
        image_array = sitk.GetArrayFromImage(image)
        del image
        image_dtype = image_array.dtype

        #predict_final = np.zeros_like(image_array).astype(np.float32)
        predict_final_shape = image_array.shape
        #predict_box = self.preprocess_1(image_array, predict_final_shape, ori_spacing)


        #bbox1 = get_bbox_from_mask_b(predict_box > 0, outside_value=0)

        #print("get box")
        pp = image_array > 50
        pp = keep_largest_connected_component(pp)
        bbox = get_bbox_from_mask_b(pp, outside_value=0)

        # crop array



        #bbox = get_bbox_from_mask_b(nonzero_mask, 0)
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        cropped_data = np.stack((image_array[resizer], np.zeros_like(image_array[resizer])), 0)
        cropped_shape = cropped_data.shape[1:]
        # resample array
        target_spacing = [1.6, 1.2859375, 1.2859375]

        #target_spacing = [1.2, 0.96445313, 0.96445313]
        #target_spacing = [1., 1., 1.]
        #target_spacing = [1., 0.80371094, 0.80371094]

        #target_spacing = [1., 0.78320301, 0.78320301]

        resampled_data, _ = resample_data(cropped_data, np.array(ori_spacing), target_spacing=target_spacing,
                                          only_z=False)
        ct_array = resampled_data[0].copy()

        # norm to one
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
        del ct_array
        ori = torch.from_numpy(resampled_data[0:1])
        del resampled_data
        print('identity: %s, preprocessing done.' % identity)
        data = [ori, identity, origin, direction, ori_size, image_dtype, predict_final_shape, resizer, cropped_shape,
                ori_spacing]
        del ori
        self.output_queue.put(data, block=True)

    # 多线程处理数据的函数
    def process_data_thread(self):
        data_list = self.iter
        while True:
            if self.check_queue_empty():
                with concurrent.futures.ThreadPoolExecutor(self.produce_queue_number) as executor:
                    # 多线程处理数据
                    for data in data_list:
                        executor.submit(self.preprocess, data)
                # 添加结束标志到队列
                for _ in range(self.produce_queue_number):
                    self.output_queue.put('end', block=True)
                print("process end.")
                return

    def check_queue_empty(self):
        # 实现检查队列是否为空的逻辑
        # 根据您的实际实现可能需要调整
        return self.output_queue.qsize() == 0

    # 第二部分的模型预测函数
    def predict_model(self):
        end_count = 0.
        while True:
            try:
                start = time.time()
                data = self.output_queue.get()
                if data == 'end':
                    end_count += 1
                    print('processes end this thread')
                    if end_count == self.produce_queue_number:
                        break
                    continue
                # nii_path = '/home/ljc/code/FLARE2023/data/Task098_FLARE2023/imagesTs/FLARE23Ts_0001.nii.gz'
                ori, identity, origin, direction, ori_size, image_dtype, predict_final_shape, resizer, cropped_shape, ori_spacing = data
                torch.cuda.empty_cache()
                ori = ori.cuda().float()
                # predict
                s = Valid_utils(model, 15, patch_size)

                predict = s.predict_3D(ori, do_mirror=False, mirror_axes=(0, 1, 2))
                #predict = predict.softmax(0).detach().cpu().squeeze().numpy()
                #save_name = os.path.join(self.save_path, '%s.npy' % identity)
                #np.save(save_name, predict)

                predict = predict.softmax(0).argmax(0).detach().cpu().squeeze().numpy()
                #predict = predict * mask
                predict[predict != 14] = 0
                predict[predict == 14] = 1

                predict_resample = resize(predict, cropped_shape, order=0, preserve_range=True, anti_aliasing=False)
                del predict
                predict_final = np.zeros(predict_final_shape, dtype=np.uint8)
                predict_final[resizer] = predict_resample
                del predict_resample
                after_itk_label = sitk.GetImageFromArray(predict_final.astype(np.uint8))
                del predict_final
                after_itk_label.SetSpacing(ori_spacing)
                after_itk_label.SetOrigin(origin)
                after_itk_label.SetDirection(direction)
                if not os.path.exists(save_path):
                    os.makedirs(os.path.join(save_path))
                #dd = self.save_path
                save_name = os.path.join(self.save_path, '%s.nii.gz' % identity)
                sitk.WriteImage(after_itk_label, save_name)
                del after_itk_label


                print('%s : cost %d s' % (identity, (time.time() - start)))
            except queue.Empty:
                time.sleep(1)
                pass
        return





def inference():
    file_list = os.listdir(base_folder_name)
    file_list = [os.path.join(base_folder_name, file_) for file_ in file_list]

    DG = MuiltThreadDataGenerator(file_list, 1, save_path)

    with torch.no_grad():
        DG._ini()


if __name__ == '__main__':
    start = time.time()
    argps = argparse.ArgumentParser()
    # -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
    argps.add_argument('input_folder', type=str, help='input folder')
    argps.add_argument('output_folder', type=str,
                       help='output folder')
    argps.add_argument('checkpoint', type=str, help='checkpoint')
    arg_s = argps.parse_args()
    base_folder_name = arg_s.input_folder
    save_path = arg_s.output_folder
    checkpoint = arg_s.checkpoint
    # init model and predict utils
    p = load_pickle(os.path.join(checkpoint, 'nnUNetPlansv2.1_plans_3D.pkl'))
    print(p['plans_per_stage'][1]['patch_size'])
    patch_size = p['plans_per_stage'][1]['patch_size']
    #patch_size = [80, 160, 160]
    #patch_size = (80, 160, 160)
    #patch_size = (40, 224, 192)
    #patch_size = (96, 160, 160)
    model = my_Trainner().get_network(
        os.path.join(checkpoint, 'model_final_checkpoint.model'))
    model.cuda()
    model.eval()

    inference()
    print("final cost", time.time() - start)
    pass


#G:\FLARE2022\Testing E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\a_imagesTest E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\a_outTest E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#G:\FLARE2022\Training\FLARE22_LabeledCase50\images E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#F:\ljc\all_organ_labeled\images F:\ljc\all_organ_labeled\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2

#F:\test\FLARE23TestImg400\FLARE23TestImg400 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs3 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#F:\test\FLARE23TestImg400\FLARE23TestImg400 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs1
#E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task099_FLARE2023\imagesTs E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs1
#E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task099_FLARE2023\imagesTs E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\outTs3 E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task096_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_normal_2
#644.6536412239075   部分保留最大连通域
#703.2188038825989   都保留最大连通域


#200Test
#G:\FLARE2022\Testing E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_raw\nnUNet_raw_data\Task99_FLARE2023\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task101_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_1

#G:\FLARE2022\Tuning\aaaa G:\FLARE2022\Tuning\out E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task101_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_1



#G:\FLARE2022\Tuning\images G:\FLARE2022\Tuning\outout E:\graduate\medicalImage\nnUNet\nnUNet\nnUNetFrame\DATASET\nnUNet_trained_models\nnUNet\3d_fullres\Task101_FLARE\nnUNetTrainerV2__nnUNetPlansv2.1\fold_1