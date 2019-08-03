from aim.utils import DetectionModel, Evaluator, VideoInference

import argparse
import os
import torch

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.coco_dataset import CocoDetection
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import PredictionTransform


class PyTorchSSD(DetectionModel):

    def __init__(self, net_type, class_num, trained_model, with_mult=1.0, nms_method='hard'):
        if net_type == 'vgg16-ssd':
            self.net = create_vgg_ssd(class_num, is_test=True)
            self.config = vgg_ssd_config
        elif net_type == 'mb1-ssd':
            self.net = create_mobilenetv1_ssd(class_num, is_test=True)
            self.config = mobilenetv1_ssd_config
        elif net_type == 'mb1-ssd-lite':
            self.net = create_mobilenetv1_ssd_lite(class_num, is_test=True)
            self.config = mobilenetv1_ssd_config
        elif net_type == 'sq-ssd-lite':
            self.net = create_squeezenet_ssd_lite(class_num, is_test=True)
            self.config = squeezenet_ssd_config
        elif net_type == 'mb2-ssd-lite':
            self.net = create_mobilenetv2_ssd_lite(class_num, width_mult=width_mult, is_test=True)
            self.config = mobilenetv1_ssd_config
        else:
            logging.fatal('The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.')

        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load(trained_model)
        self.net = self.net.to(self.DEVICE)

        if net_type == 'vgg16-ssd':
            self.predictor = create_vgg_ssd_predictor(self.net, nms_method=nms_method, device=self.DEVICE)
        elif net_type == 'mb1-ssd':
            self.predictor = create_mobilenetv1_ssd_predictor(self.net, nms_method=nms_method, device=self.DEVICE)
        elif net_type == 'mb1-ssd-lite':
            self.predictor = create_mobilenetv1_ssd_lite_predictor(self.net, nms_method=nms_method, device=self.DEVICE)
        elif net_type == 'sq-ssd-lite':
            self.predictor = create_squeezenet_ssd_lite_predictor(self.net,nms_method=nms_method, device=self.DEVICE)
        elif net_type == 'mb2-ssd-lite':
            self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, nms_method=nms_method, device=self.DEVICE)
        else:
            logging.fatal('The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.')
        
    def __call__(self, *args, **kwargs):
        boxes, labels, probs = self.predictor.predict(args[0])
        results = []
        for box, label, prob in zip(boxes, labels, probs):
            obj = dict()
            obj['category_id'] = label.item() - 1 # Remove BACKGROUND
            obj['score'] = prob.item()
            x = box[0].item()
            y = box[1].item()
            w = box[2].item() - x
            h = box[3].item() - y
            obj['bbox'] = [x, y, w, h]
            results.append(obj)
        return results

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Video.')
    parser.add_argument('--net_type', default='vgg16-ssd', help='The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.')
    parser.add_argument('--trained_model', type=str)
    parser.add_argument('--dataset', type=str, help='The root directory of the COCO dataset.')
    parser.add_argument('--nms_method', type=str, default='hard')
    parser.add_argument('--mb2_width_mult', default=1.0, type=float, help='Width Multiplifier for MobilenetV2')
    parser.add_argument('--threshold', type=int, default=0.4)
    parser.add_argument('--crop', type=str, default='0:0:0:0')
    parser.add_argument('--input', default='test.mp4', type=str)
    parser.add_argument('--output', default='output.mp4', type=str)
    args = parser.parse_args()

    dataset = CocoDetection(os.path.join(args.dataset, 'val'), os.path.join(args.dataset, 'val.json'))
    model = PyTorchSSD(args.net_type, len(dataset.class_names), args.trained_model, with_mult=args.mb2_width_mult, nms_method=args.nms_method)

    crop = [int(c) for c in args.crop.split(':')]
    video_maker = VideoInference(os.path.join(args.dataset, 'val.json'), args.input, args.output, args.threshold, dict(x=crop[2], y=crop[3], w=crop[0], h=crop[1]))
    video_maker.run(model)
