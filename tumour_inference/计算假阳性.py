import os
import nibabel as nib
import numpy as np


def load_nii_files_from_folder(folder_path):
    """
    从指定文件夹中加载所有.nii.gz文件

    参数:
    folder_path (str): 存储CT预测结果的文件夹路径

    返回:
    predictions (list of numpy arrays): 加载的每个CT片子的3D预测数组
    """
    predictions = []

    # 遍历文件夹中的所有.nii.gz文件
    for file_name in os.listdir(folder_path):
        print(file_name)
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(folder_path, file_name)
            # 使用nibabel加载.nii.gz文件
            nii_image = nib.load(file_path)
            # 将加载的图像数据转换为numpy数组
            prediction_data = nii_image.get_fdata()
            # 添加到预测列表中
            predictions.append(prediction_data)

    return predictions


def calculate_fpr_for_multiple_cts(predictions):
    """
    计算多张CT片子的假阳性率 (FPR)
    假设所有CT片子都是健康的，即真实标签全为0（无肿瘤）。

    参数:
    predictions (list of numpy arrays): 多张CT片子的模型预测结果，每个元素是一个3D数组

    返回:
    fpr (float): 假阳性率
    """
    # 初始化FP和TN
    total_FP = 0
    total_TN = 0
    t = 0
    # 遍历所有CT片子的预测结果
    for predicted in predictions:
        print("###############")
        # 假设真实标签全为0 (无肿瘤)
        ground_truth = np.zeros_like(predicted)

        # 累积 False Positive (FP) 和 True Negative (TN)
        total_FP += np.sum((predicted == 1) & (ground_truth == 0))  # False Positives
        total_TN += np.sum((predicted == 0) & (ground_truth == 0))  # True Negatives
        s = np.unique(predicted)
        if(len(np.unique(predicted)) > 1):
            t = t+1
    # 计算总体的假阳性率 FPR
    fpr = total_FP / (total_FP + total_TN) if (total_FP + total_TN) > 0 else 0
    print(t / 40)
    return fpr


# 指定存储CT预测结果的文件夹路径
folder_path = r"F:\Release-FLARE24-T1\Validation-Public\noLesion"

# 从文件夹加载所有.nii.gz文件
predictions = load_nii_files_from_folder(folder_path)

# 计算多张CT片子的假阳性率
fpr = calculate_fpr_for_multiple_cts(predictions)
print(f"假阳性率: {fpr:.4f}")
