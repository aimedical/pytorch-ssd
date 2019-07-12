from aim.utils import DetectionModel, Evaluator, BBoxDrawing

import argparse
import os
import torch

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.coco_dataset import CocoDetection


class PyTorchSSD(DetectionModel):

    def __init__(self, net_type, class_num, trained_model, with_mult=1.0, nms_method='hard'):
        if net_type == 'vgg16-ssd':
            self.net = create_vgg_ssd(class_num, is_test=True)
        elif net_type == 'mb1-ssd':
            self.net = create_mobilenetv1_ssd(class_num, is_test=True)
        elif net_type == 'mb1-ssd-lite':
            self.net = create_mobilenetv1_ssd_lite(class_num, is_test=True)
        elif net_type == 'sq-ssd-lite':
            self.net = create_squeezenet_ssd_lite(class_num, is_test=True)
        elif net_type == 'mb2-ssd-lite':
            self.net = create_mobilenetv2_ssd_lite(class_num, width_mult=width_mult, is_test=True)
        else:
            logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        
    def __call__(self, *args, **kwargs):
        boxes, labels, probs = self.predictor.predict(args[0])
        results = []
        image_id = kwargs['image_id']
        for box, label, prob in zip(boxes, labels, probs):
            obj = dict()
            obj['image_id'] = image_id
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
    parser = argparse.ArgumentParser(description="SSD Evaluation on COCO Dataset.")
    parser.add_argument('--net_type', default="vgg16-ssd",
                        help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument("--trained_model", type=str)
    parser.add_argument("--dataset", type=str, help="The root directory of the COCO dataset.")
    parser.add_argument("--nms_method", type=str, default="hard")
    parser.add_argument("--output_dir", default="output", type=str, help="The directory name of the detection result.")
    parser.add_argument("--output", default="bbox.json", type=str, help="The filename of the detection result.")
    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')
    args = parser.parse_args()

    dataset = CocoDetection(os.path.join(args.dataset, 'val'), os.path.join(args.dataset, 'val.json'))
    model = PyTorchSSD(args.net_type, len(dataset.class_names), args.trained_model, with_mult=args.mb2_width_mult, nms_method=args.nms_method)

    evaluator = Evaluator(os.path.join(args.dataset, 'val.json'), args.output)
    data_loader = torch.utils.data.DataLoader(dataset)
    evaluator.run(model, data_loader)

    bbox_drawing = BBoxDrawing()
    bbox_drawing.run(args.output_dir, os.path.join(args.dataset, 'val'), os.path.join(args.dataset, 'val.json'), args.output)
