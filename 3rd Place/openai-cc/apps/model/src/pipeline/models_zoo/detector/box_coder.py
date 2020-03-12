'''Encode object boxes and labels.'''
import math
import torch
import itertools
import numpy as np


from .ssd_utils import box_iou, box_nms, change_box_order, class_independent_decode, class_dependent_decode, same_box, meshgrid


class SSDBoxCoder:
    def __init__(self, ssd_model, ignore_threshold, iou_threshold, boxes_format='pascal_voc'):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.default_boxes = self._get_default_boxes()
        self.ignore_threshold = ignore_threshold
        self.iou_threshold = iou_threshold
        self.boxes_format = boxes_format
        self.input_size = ssd_model.input_size

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]

                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        '''

        if len(boxes) == 0:
            return (
                torch.zeros(self.default_boxes.shape, dtype=torch.float32),
                torch.zeros((self.default_boxes.shape[0],), dtype=torch.long)
            )

        default_boxes = self.default_boxes.clone()
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious: torch.Tensor = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        max_anchor_iou, max_iou_object_index = torch.max(ious, 1)
        cls_targets = labels[max_iou_object_index] + 1
        cls_targets[max_anchor_iou < self.ignore_threshold] = 0  # Background
        cls_targets[(max_anchor_iou >= self.ignore_threshold) & (max_anchor_iou < self.iou_threshold)] = -1  # Ignored

        loc_targets = boxes[max_iou_object_index]
        loc_targets = change_box_order(loc_targets, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        loc_xy = (loc_targets[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:]
        loc_wh = torch.log(loc_targets[:, 2:] / default_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        return loc_targets, cls_targets


    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45, class_independent_nms=False):

        """Decode predicted loc/cls back to real box locations and class labels.
        Args:
          multi_bboxes: (tensor) predicted loc, sized [#anchors, 4].
          multi_labels: (tensor) predicted conf, sized [#anchors, #classes].
          score_threshold: (float) threshold for object confidence score.
          nms_threshold: (float) threshold for box nms.
          class_independent_nms: (bool).

        Returns:
          bboxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj, ].
        """

        xy = loc_preds[:,:2] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = loc_preds[:,2:].exp() * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        decode = class_independent_decode if class_independent_nms else class_dependent_decode
        return decode(box_preds, cls_preds, score_thresh, nms_thresh)



    def decode_patches(self, loc_preds, cls_preds, scheme, patch_size, score_thresh=0.6, nms_thresh=0.45, include_thresh=0.65, class_independent_nms=False):
        num_classes = cls_preds.size(2)
        decode = class_independent_decode if class_independent_nms else class_dependent_decode

        nums = [i[0]*i[1] for i in scheme]
        scheme_nums = np.concatenate([[i]*n for i, n in enumerate(nums)])

        predictions = []

        for j in range(loc_preds.shape[0]):
            xy = loc_preds[j][:,:2] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
            wh = loc_preds[j][:,2:].exp() * self.default_boxes[:,2:]
            box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

            predictions.append(decode(box_preds, cls_preds[j], score_thresh, nms_thresh))

        prev_sch = None
        tile_num = 0
        all_boxes, all_labels, all_scores = [], [] ,[]
        for i, pred in enumerate(predictions):
            scheme_num = scheme_nums[i]
            sch = scheme[scheme_num]
            if sch != prev_sch:
                prev_sch = sch
                tile_num = 0
            else:
                tile_num += 1

            ny, nx, inter = sch
            h = int(patch_size * ny - inter * (ny - 1))
            w = int(patch_size * nx - inter * (nx - 1))

            boxes, labels, scores = pred
            if boxes.shape[0] > 0:
                y = (tile_num // nx) + 1
                x = (tile_num % nx) + 1
                y1, y2, x1, x2 = (y-1)*patch_size-(y-1)*inter,  (y)*patch_size-(y-1)*inter ,  (x-1)*patch_size-(x-1)*inter,  (x)*patch_size-(x-1)*inter
                boxes = (boxes + torch.Tensor(np.array([x1, y1, x1, y1]))) / torch.Tensor(np.array([w,h,w,h]))
                all_boxes.append(boxes)
                all_labels.append(labels)
                all_scores.append(scores)

        if len(all_boxes) > 0:
            all_boxes = torch.cat(all_boxes)
            all_labels = torch.cat(all_labels)
            all_scores = torch.cat(all_scores)
        else:
            all_boxes =  torch.tensor([], dtype=torch.float)
            all_labels = torch.tensor([], dtype=torch.long)
            all_scores = torch.tensor([], dtype=torch.float)


        if len(all_boxes) > 0:
            final_boxes, final_labels, final_scores = [], [] ,[]
            for cl in range(num_classes-1):
                keep_cl = all_labels == cl
                if keep_cl.sum() > 0:
                    keep = box_nms(all_boxes[keep_cl], all_scores[keep_cl], threshold=0.3)
                    keep2 = same_box(all_boxes[keep_cl][keep], thr=include_thresh)
                    final_boxes.append(all_boxes[keep_cl][keep][keep2])
                    final_labels.append(all_labels[keep_cl][keep][keep2])
                    final_scores.append(all_scores[keep_cl][keep][keep2])
            return torch.cat(final_boxes), torch.cat(final_labels), torch.cat(final_scores)
        else:
            return torch.tensor([], dtype=torch.float), \
                   torch.tensor([], dtype=torch.long), \
                   torch.tensor([], dtype=torch.float)
