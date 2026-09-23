# -*- coding: utf-8 -*-

# ไฟล์นี้คือ ModelNew ที่เรียกใช้ run จากไฟล์เดิมเพื่อรักษาระบบที่ถูกต้องและไม่แก้ไขโค้ด PyTorch ใดๆ

import math
import torch
import torch.nn.functional as F

# นำเข้าโมดูลและฟังก์ชันจากไฟล์ที่ให้มา (จำเป็นต้องอยู่ในโฟลเดอร์เดียวกัน)
# รัน forward โดยไม่แก้ไข run หรือ get_inputs
from the_original_file import run  # แทนที่ด้วยการนำเข้า run จากไฟล์ที่มีชื่อ (มันอาจมีชื่อว่า model.py หรืออะไรก็ตามที่ให้มา)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                reverse: bool,
                transform_0_conv0_weight: torch.Tensor,
                transform_0_conv0_bias: torch.Tensor,
                transform_0_conv1_weight: torch.Tensor,
                transform_0_conv1_bias: torch.Tensor,
                transform_0_conv2_weight: torch.Tensor,
                transform_0_conv2_bias: torch.Tensor,
                transform_1_conv0_weight: torch.Tensor,
                transform_1_conv0_bias: torch.Tensor,
                transform_1_conv1_weight: torch.Tensor,
                transform_1_conv1_bias: torch.Tensor,
                transform_1_conv2_weight: torch.Tensor,
                transform_1_conv2_bias: torch.Tensor,
                transform_2_conv0_weight: torch.Tensor,
                transform_2_conv0_bias: torch.Tensor,
                transform_2_conv1_weight: torch.Tensor,
                transform_2_conv1_bias: torch.Tensor,
                transform_2_conv2_weight: torch.Tensor,
                transform_2_conv2_bias: torch.Tensor,
                transform_3_conv0_weight: torch.Tensor,
                transform_3_conv0_bias: torch.Tensor,
                transform_3_conv1_weight: torch.Tensor,
                transform_3_conv1_bias: torch.Tensor,
                transform_3_conv2_weight: torch.Tensor,
                transform_3_conv2_bias: torch.Tensor):
        """
        ModelNew.forward รันฟังก์ชัน run ที่ถูกต้องโดยส่ง args ทั้งหมดเข้าไป
        ห้ามเปลี่ยนแปลงการทำงานของ run หรือ get_inputs ในไฟล์ PyTorch
        """
        # เรียก run ด้วย args เดียวกันที่ได้จาก get_inputs
        return run(
            x, x_mask, reverse,
            transform_0_conv0_weight, transform_0_conv0_bias,
            transform_0_conv1_weight, transform_0_conv1_bias,
            transform_0_conv2_weight, transform_0_conv2_bias,
            transform_1_conv0_weight, transform_1_conv0_bias,
            transform_1_conv1_weight, transform_1_conv1_bias,
            transform_1_conv2_weight, transform_1_conv2_bias,
            transform_2_conv0_weight, transform_2_conv0_bias,
            transform_2_conv1_weight, transform_2_conv1_bias,
            transform_2_conv2_weight, transform_2_conv2_bias,
            transform_3_conv0_weight, transform_3_conv0_bias,
            transform_3_conv1_weight, transform_3_conv1_bias,
            transform_3_conv2_weight, transform_3_conv2_bias
        )


def run(*args):
    return ModelNew()(*args)
