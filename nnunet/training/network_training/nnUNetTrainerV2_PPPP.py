import torch
import numpy as np
from torch.cuda.amp import autocast


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

from nnunet.training.dataloading.dataset_loading import DataLoader2D, DataLoader3D
from nnunet.training.dataloading.dataset_loading import *
from nnunet.training.dataloading.dataset_loading_class_first import DataLoader3DMultiMaskData

from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch



from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet,  \
    Generic_UNet_Teacher, Generic_UNet_Student
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

class my_Trainner():
    def __init__(self, checkpoint):
        self.base_num_features = self.num_classes = self.net_num_pool_op_kernel_sizes = self.conv_per_stage \
            = self.net_conv_kernel_sizes = None
        self.stage = 1
        self.network = None
        self.checkpoint = checkpoint
        self.initial_par()

    def initial_par(self):
        #plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16 = \
        #    load_pickle(os.path.join(os.path.dirname(__file__), 'checkpoints/model_final_checkpoint.model.pkl')
        #                )[
        #        'init']
        plans = load_pickle(os.path.join(self.checkpoint, 'nnUNetPlansv2.1_plans_3D.pkl'))
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
        #conv_op = SeparableConv3d
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
        self.network = Generic_UNet_Teacher(1, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

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


class nnUNetTrainerV2_PPPP(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 10
        self.initial_lr = 1e-2
        self.do_bg = False
        self.do_mean = True
        self.weight_k = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.weight_s = self.weight_k
        self.model_loss = nn.MSELoss(reduction='mean')

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()
            ############## lyy
            self.base_num_features = 16
            self.net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
            self.net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1], [1, 2, 2]]

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            print(weights)
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            #conv_op = SeparableConv3d
            conv_op = nn.Conv3d
            #conv_op = MedNeXtBlock
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d



        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}



        #self.conv_per_stage = 1
        #self.stage_num = 6
        #self.base_num_features = 32
        #self.max_num_features = 256

        #if len(self.net_conv_kernel_sizes) > self.stage_num:
        #    self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[len(self.net_conv_kernel_sizes) - self.stage_num:]
        #    self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[len(self.net_num_pool_op_kernel_sizes) - (self.stage_num - 1):]

        self.network = Generic_UNet_Student(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, None,
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)


        '''
        self.network = MedNeXt_2(
            in_channels=1,
            n_channels=32,
            n_classes=14,
            exp_r=[1, 1, 1, 1, 2, 2, 1, 1, 1],  # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,  # Can test kernel_size
            deep_supervision=True,  # Can be used to test deep supervision
            do_res=True,  # Can be used to individually test residual connection
            do_res_up_down=True,
            # block_counts = [2,2,2,2,2,2,2,2,2],
            block_counts=[1, 1, 2, 2, 3, 2, 2, 1, 1],
            checkpoint_style=None,
            dim='3d',
            grn=True)
        '''





        checkpoint = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        checkpoint = os.path.join(checkpoint, 'checkpoint_teacher')
        self.model = my_Trainner(checkpoint).get_network(
            os.path.join(checkpoint, 'model_final_checkpoint.model'))
        #self.activation = {}
        #self.model.conv_blocks_context[-2].register_forward_hook(get_activation('conv_blocks_context_second_last', self.activation))
        #self.model.conv_blocks_context[-1].register_forward_hook(get_activation('conv_blocks_context_last', self.activation))

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(self.model)

        if torch.cuda.is_available():
            #self.teacher_network.cuda()
            self.network.cuda()



        #self.teacher_network.inference_apply_nonlin = softmax_helper
        self.network.inference_apply_nonlin = softmax_helper
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        number = []
        matrix_weight = []
        for i in data_dict['keys']:
            # 将字符串转换为整数
            number.append(i)
        print(number)
        # [1,1.57402705,5.84469124,5.51327007,7.70764601,7.70393582,7.82239399,9.86036445,9.79212208,9.07417292,9.52126771,4.80561119,8.00863778,5.77692344]
        weight = torch.ones(size=(len(target), self.batch_size, self.num_classes)) * 1 / 14  # lyy 14  tumour 2
        weight_1 = torch.ones(size=(len(target), self.batch_size, self.num_classes)) * 1 / 14  # lyy 14  tumour 2
        weight_d = [1., 0.9743, 0.9743, 0.9743, 0.8528, 0.9559, 0.9559, 0.8280, 0.8286, 0.8340, 0.8122, 0.9237, 0.8204,
                    0.9291]
        # * np.array([1, 1.83380888, 12.33115005, 10.94324685, 25.16274108, 25.12324126, 26.42838704, 87.5953611, 82.48844657, 49.49789066, 66.5423167, 8.46811204, 28.6819111, 12.03332958])
        weight_d = np.array(weight_d)
        weight_d = weight_d / weight_d.sum()
        weight_d = torch.tensor(weight_d)
        weight_f = [1, 1.83380888, 12.33115005, 10.94324685, 25.16274108, 25.12324126, 26.42838704, 87.5953611,
                    82.48844657, 49.49789066, 66.5423167, 8.46811204, 28.6819111, 12.03332958]
        weight_f = np.array(weight_f)
        weight_f = weight_f / weight_f.sum()
        weight_f = torch.tensor(weight_f)

        weight_k = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        weight_k = weight_k / weight_k.sum()
        weight_k = torch.tensor(weight_k)

        weight_s = np.array([1, 1.1, 1.2, 1.2, 1.3, 1.4, 1.4, 2, 2, 1.6, 1.8, 1.2, 1.3, 1.2])
        weight_s = weight_s / weight_s.sum()
        weight_s = torch.tensor(weight_s)
        for l in range(len(target)):
            for b in range(target[l].shape[0]):
                if do_backprop:
                    '''
                    ########tumour
                    num_pos_samples = torch.sum(target[l][b] == 1)
                    num_neg_samples = torch.sum(target[l][b] == 0)
                    pos_weight = (num_neg_samples + 1e-3) / (num_pos_samples + 1e-3)
                    '''
                    if (number[b].startswith("FLARE22")):
                        weight[l, b] = weight_k
                        weight_1[l, b] = weight_s
                        #weight_1[l, b] = weight_k

                    else:
                        weight[l, b] = weight_d
                        #weight[l, b] = weight_k
                        #weight_1[l, b] = weight_k
                        weight_1[l, b] = weight_s
                    # weight[l, b] =  weight_k
                    # p = torch.tensor([1, pos_weight])
                    # p = p / p.sum()
                    # weight[l, b] = p
                else:

                    weight[l, b] = weight_k
                    weight_1[l, b] = weight_k


                # else:
                #    target[l][b][1] = target[l][b][0]
        # for l in range(len(target)):
        #     for b in range(target[l].shape[0]):
        #         target[l][b][1] = target[l][b][0]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        weight = maybe_to_torch(weight)
        weight_1 = maybe_to_torch(weight_1)
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            weight = to_cuda(weight)
            weight_1 = to_cuda(weight_1)
        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                with torch.no_grad():
                    teacher_out = self.model(data)
                
                output, student_out = self.network(data)
                #student_output = output[0]

                #for name, activation in self.activations.items():
                #    print(f"{name} output shape: {activation.shape}")

                del data
                #l = self.loss(output, target)
                if do_backprop:
                    l = self.loss(output, target, [weight, weight_1, self.epoch, 50]) + 0.5 * self.model_loss(student_out[0], teacher_out[0]) + 0.5 * self.model_loss(student_out[1], teacher_out[1])
                else:
                    l = self.loss(output, target, [weight, weight_1, self.epoch, 50])
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()




    def get_weight(self, properties):
        shape_weight = (len(properties), len(properties[0]['class_locations']) + 1)
        classes_list = [prop['classes'].astype(np.int16) for prop in properties]

        weight = torch.zeros(shape_weight)
        t = -1 if self.do_bg else 0
        for i in range(shape_weight[0]):
            classes = classes_list[i][classes_list[i] > t]
            weight[i][classes] = 1
        return weight

    def get_weight_2(self, properties):
        # shape_weight = (len(properties), len(properties[0]['class_locations']) + 1)
        # weight = torch.ones(shape_weight)
        # for i in range(shape_weight[0]):
        #    weight[i][14] = 2
        weight = torch.ones(14)
        dd = torch.tensor(np.array([0.99, 0.9743, 0.9743, 0.9743, 0.8528, 0.9559, 0.9559, 0.8280, 0.8286, 0.8340, 0.8122, 0.9237, 0.7904, 0.9291]))
        weight = dd
        #weight[14] = 2

        return weight

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size,
                                              self.batch_size,
                                              False, oversample_foreground_percent=self.oversample_foreground_percent,
                                              pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                               False,
                                               oversample_foreground_percent=self.oversample_foreground_percent,
                                               pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val





# 定义钩子函数
def get_activation(name, activations):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

