import colorsys
import os
import time
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
import pandas as pd
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

'''
训练自己的数据集必看注释！
'''
#---------------------------------------------------#
#   最多元素
#---------------------------------------------------#
collections_list = [] #定义统计列表
df = pd.DataFrame(data=np.zeros((1, 6)),
    columns=['危害鸟种', '置信度','先验框个数','涉鸟故障类型','风险等级','防治措施'],  #行
    index=np.linspace(1, 1, 1, dtype=int))  #列  

def most_frequent(lst):
    dict = {}
    count, itm = 0, ''
    for item in reversed(lst):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count:
            count, itm = dict[item], item
    return itm
      
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/ep100-loss0.062-val_loss0.046.pt',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   所使用的YoloV5的版本。s、m、l、x
        #------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.2,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        ChineseName = { 'HG' : '黑鹳', 'DFBG' : '东方白鹳', 'DB' : '大鸨'  , 'BL' : '白鹭', 'DS' : '戴胜',
                'CL' : '池鹭' , 'HS' : '红隼', 'HLLN': '黑领椋鸟', 'SGLN' : '丝光椋鸟', 'BG' : '八哥',
                'HXQ' : '灰喜鹊', 'XQ' : '喜鹊', 'DZWY' : '大嘴乌鸦' , 'DDJ' : '大杜鹃', 'ZJBJ' : '珠颈斑鸠',
                'BTB' : '白头鹎', 'HZHL' : '黑枕黄鹂', 'JY' : '家燕', 'HY' : '鸿雁', 'CEX' : '长耳鸮',
                'NBL' : '牛背鹭', 'BPL' : '白琵鹭', 'TJ' : '秃鹫', 'SY' : '松鸦', 'HZLQ' : '红嘴蓝鹊',
                'HWBL': '虎纹伯劳', 'BHWQ' : '北红尾鸲', 'HM' : '画眉', 'DTE' : '大天鹅', 'CMY' : '赤麻鸭',
                'QBMY' : '翘鼻麻鸭', 'LTY' : '绿头鸭', 'PTLC' : '普通鸬鹚', 'JYTH' : '卷羽鹈鹕', 'HSJ' : '黑水鸡',
                'PTCN' : '普通翠鸟', 'FTMJ' : '凤头麦鸡', 'PTYO' : '普通燕鸥', 'GYG' : '冠鱼狗', 'HTLZMN' : '灰头绿啄木鸟'}
        
        Chi_EngName = { 'HG' : '黑鹳(Ciconia nigra)', 'DFBG' : '东方白鹳(Ciconia boyciana)', 'DB' : '大鸨(Otis tarda)'  , 'BL' : '白鹭(Egretta garzetta)', 'DS' : '戴胜(Upupa epops)',
                'CL' : '池鹭(Ardeola bacchus)' , 'HS' : '红隼(Falco tinnunculus)', 'HLLN': '黑领椋鸟(Sturnus nigricollis)', 'SGLN' : '丝光椋鸟(Sturnus sericeus)', 'BG' : '八哥(Acridotheres cristatellus)',
                'HXQ' : '灰喜鹊(Cyanopica cyana)', 'XQ' : '喜鹊(Pica pica)', 'DZWY' : '大嘴乌鸦(Corvus macrorhynchos)' , 'DDJ' : '大杜鹃(Cuculus canorus)', 'ZJBJ' : '珠颈斑鸠(Streptopelia chinensis)',
                'BTB' : '白头鹎(Pycnonotus sinensis)', 'HZHL' : '黑枕黄鹂(Oriolus chinensis)', 'JY' : '家燕(Hirundo rustica)', 'HY' : '鸿雁(Anser cygnoides)', 'CEX' : '长耳鸮(Asio otus)',
                'NBL' : '牛背鹭(Bubulcus ibis)', 'BPL' : '白琵鹭(Platalea leucorodia)', 'TJ' : '秃鹫(Aegypius monachus)', 'SY' : '松鸦(Garrulus glandarius)', 'HZLQ' : '红嘴蓝鹊(Urocissa erythrorhyncha)',
                'HWBL': '虎纹伯劳(Lanius tigrinus)', 'BHWQ' : '北红尾鸲(Phoenicurus auroreus)', 'HM' : '画眉(Garrulax canorus)', 'DTE' : '大天鹅(Cygnus cygnus)', 'CMY' : '赤麻鸭(Tadorna ferruginea)',
                'QBMY' : '翘鼻麻鸭(Tadorna tadorna)', 'LTY' : '绿头鸭(Anas platyrhynchos)', 'PTLC' : '普通鸬鹚(Phalacrocorax carbo)', 'JYTH' : '卷羽鹈鹕(Pelecanus crispus)', 'HSJ' : '黑水鸡(Gallinula chloropus)',
                'PTCN' : '普通翠鸟(Alcedo atthis)', 'FTMJ' : '凤头麦鸡(Vanellus vanellus)', 'PTYO' : '普通燕鸥(Sterna hirundo)', 'GYG' : '冠鱼狗(Megaceryle lugubris)', 'HTLZMN' : '灰头绿啄木鸟(Picus canus)'}
        
        Measure = {'HG'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'DFBG' : '挡鸟类、引鸟类、驱鸟类装置', 
                   'DB'   : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'BL'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'DS'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'CL'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HS'   : '挡鸟类、引鸟类、驱鸟类装置', 
                   'HLLN' : '防鸟刺，防鸟挡板，防鸟盒，防鸟护套，人工栖鸟平台，人工鸟巢，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'SGLN' : '防鸟刺，防鸟挡板，防鸟盒，防鸟护套，人工栖鸟平台，人工鸟巢，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'BG'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'HXQ'  : '防鸟刺，防鸟挡板，防鸟盒，防鸟护套，人工栖鸟平台，人工鸟巢，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'XQ'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟护套，人工栖鸟平台，人工鸟巢，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'DZWY' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'DDJ'  : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'ZJBJ' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'BTB'  : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HZHL' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'JY'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HY'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'CEX'  : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'NBL'  : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'BPL'  : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'TJ'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'SY'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HZLQ' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'HWBL' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'BHWQ' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HM'   : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'DTE'  : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'CMY'  : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'QBMY' : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'LTY'  : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'PTLC' : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'JYTH' : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HSJ'  : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置',
                   
                   'PTCN' : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'FTMJ' : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'PTYO' : '防鸟护套，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'GYG'  : '防鸟刺，防鸟挡板，防鸟盒，防鸟针板，防鸟罩，防鸟护套，防鸟拉线，人工鸟巢，人工栖鸟平台，旋转式风车、反光镜等惊鸟装置，声、光、电等驱鸟装置', 
                   'HTLZMN' : '旋转式风车、反光镜等惊鸟装置 声、光、电等驱鸟装置'}
        HarmRank = { 'HG' : '高危', 'DFBG' : '高危', 'DB' : '高危'  , 'BL' : '高危', 'DS' : '高危',
                        'CL' : '高危' , 'HS' : '高危', 'HLLN': '高危', 'SGLN' : '高危', 'BG' : '高危',
                        'HXQ' : '高危', 'XQ' : '高危', 'DZWY' : '高危' , 'DDJ' : '高危', 'ZJBJ' : '高危',
                        'BTB' : '高危', 'HZHL' : '高危', 'JY' : '高危', 'HY' : '高危', 'CEX' : '高危',
                    
                        'NBL' : '微害', 'BPL' : '微害', 'TJ' : '微害', 'SY' : '微害', 'HZLQ' : '微害',
                        'HWBL': '微害', 'BHWQ' : '微害', 'HM' : '微害', 'DTE' : '微害', 'CMY' : '微害',
                        'QBMY' : '微害', 'LTY' : '微害', 'PTLC' : '微害', 'JYTH' : '微害', 'HSJ' : '微害',
                        'PTCN' : '微害', 'FTMJ' : '微害', 'PTYO' : '微害', 'GYG' : '微害', 'HTLZMN' : '微害'}
        
        ProblemTpye = { 'HG' : '鸟粪闪络、鸟体短接、鸟巢短路', 'DFBG' : '鸟粪闪络、鸟巢短路、鸟体短接', 'DB' : '鸟体短接'  , 'BL' : '鸟粪闪络、鸟巢短路', 'DS' : '鸟粪闪络',
                        'CL' : '鸟粪闪络、鸟巢短路' , 'HS' : '鸟粪闪络', 'HLLN': '鸟巢短路', 'SGLN' : '鸟巢短路', 'BG' : '鸟粪闪络、鸟巢短路',
                        'HXQ' : '鸟巢短路、鸟啄复合绝缘子', 'XQ' : '鸟巢短路、鸟啄复合绝缘子', 'DZWY' : '鸟粪闪络、鸟巢短路、鸟啄复合绝缘子' , 'DDJ' : '鸟粪闪络', 'ZJBJ' : '鸟粪闪络、鸟体短接、鸟巢短路',
                        'BTB' : '鸟粪闪络', 'HZHL' : '鸟粪闪络', 'JY' : '鸟粪闪络', 'HY' : '鸟粪闪络、鸟体短接', 'CEX' : '鸟粪闪络、鸟体短接',
                        'NBL' : '鸟粪闪络、鸟巢短路', 'BPL' : '鸟体短接', 'TJ' : '鸟体短接、鸟粪闪络', 'SY' : '鸟粪闪络', 'HZLQ' : '鸟粪闪络',
                        'HWBL': '鸟粪闪络', 'BHWQ' : '鸟粪闪络', 'HM' : '鸟粪闪络', 'DTE' : '鸟体短接', 'CMY' : '鸟体短接',
                        'QBMY' : '鸟体短接', 'LTY' : '鸟体短接', 'PTLC' : '鸟体短接', 'JYTH' : '鸟体短接', 'HSJ' : '鸟体短接',
                        'PTCN' : '鸟粪闪络', 'FTMJ' : '鸟体短接', 'PTYO' : '鸟体短接', 'GYG' : '鸟粪闪络', 'HTLZMN' : '鸟啄类'}
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32') #标签
            top_conf    = results[0][:, 4] * results[0][:, 5]   #置信度
            top_boxes   = results[0][:, :4]
            #此处存在循环 num=len(np.unique(data))
            # st.subheader(':balloon:种类：{}     :balloon:个数：{}'.format(top_label, top_conf))
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)] #标签名
            box             = top_boxes[i]
            score           = top_conf[i]   #置信度
            
            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
    ## 简单显示预测结果 ##
        
        # st.title(':baby_chick:预测结果')
    ## 表格 
#         df = pd.DataFrame(data=np.zeros((len(np.unique(top_label)), 6)),
#             columns=['危害鸟种', '置信度','先验框个数','涉鸟故障类型','风险等级','防治措施'],  #行
#             index=np.linspace(1, len(np.unique(top_label)), len(np.unique(top_label)), dtype=int))  #列  

        top_label_class, top_label_num= np.unique(top_label,return_counts=True)
        top_label_class, top_label_index = np.unique(top_label,return_index=True)
        
        
        for i, c in list(enumerate(np.unique(top_label))):    #将矩阵添加索引（键值对）
            Predicted_LableClass = self.class_names[int(c)]  #数字->标签
            collections_list.append(Predicted_LableClass) #添加元素
            # link_Wiki = 'https://en.wikipedia.org/wiki/' + \
            #     predicted_class.lower().replace(' ', '_')  # 故障鸟种超链接Wiki百科
#             link_Baidu = 'https://baike.baidu.com/item/' + \
#                 ChineseName[Predicted_LableClass].replace(' ', '_')  # 故障鸟种超链接百度百科
            
#             df.iloc[i,0] = f'<a href="{link_Baidu}" target="_blank">{Chi_EngName[Predicted_LableClass]}</a>'   #标签->中文名
#             # 显示识别故障鸟种置信度
#             df.iloc[i, 1] = top_conf[top_label_index[i]]
#             list(enumerate(np.unique(top_label_num)))
#             df.iloc[i,2] = top_label_num[i]
#             df.iloc[i,3] = f'<a target="_blank">{ProblemTpye[Predicted_LableClass]}</a>'
#             df.iloc[i,4] = f'<a target="_blank">{HarmRank[Predicted_LableClass]}</a>'
#             df.iloc[i,5] = f'<a target="_blank">{Measure[Predicted_LableClass]}</a>'
#             st.write(df.to_html(escape=False), unsafe_allow_html=True) #显示表格
        
        Predicted_LableClass = most_frequent(collections_list)
        link_Baidu = 'https://baike.baidu.com/item/' + \
            ChineseName[Predicted_LableClass].replace(' ', '_')  # 故障鸟种超链接百度百科
        df.iloc[0,0] = f'<a href="{link_Baidu}" target="_blank">{Chi_EngName[Predicted_LableClass]}</a>'   #标签->中文名
        # 显示识别故障鸟种置信度
        df.iloc[0, 1] = top_conf[top_label_index[0]]
        list(enumerate(np.unique(top_label_num)))
        df.iloc[0,2] = top_label_num[0]
        df.iloc[0,3] = f'<a target="_blank">{ProblemTpye[Predicted_LableClass]}</a>'
        df.iloc[0,4] = f'<a target="_blank">{HarmRank[Predicted_LableClass]}</a>'
        df.iloc[0,5] = f'<a target="_blank">{Measure[Predicted_LableClass]}</a>'
        return image

    def show_df(self):
#         st.write(collections_list)  
        st.write(df.to_html(escape=False), unsafe_allow_html=True) #显示表格
    def reload_col_list(self):
        collections_list.clear()
#         st.write(collections_list)  
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
