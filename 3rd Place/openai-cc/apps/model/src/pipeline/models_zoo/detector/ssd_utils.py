import torch


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)

def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes

def box_select(boxes, xmin, ymin, xmax, ymax):
    '''Select boxes in range (xmin,ymin,xmax,ymax).

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) selected boxes, sized [M,4].
      (tensor) selected mask, sized [N,].
    '''
    mask = (boxes[:,0]>=xmin) & (boxes[:,1]>=ymin) \
         & (boxes[:,2]<=xmax) & (boxes[:,3]<=ymax)
    boxes = boxes[mask,:]
    return boxes, mask

def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]

    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def same_box(bboxes, thr=0.65):
    lt = torch.max(bboxes[:,None,:2], bboxes[:,:2])  # [N,M,2]
    rb = torch.min(bboxes[:,None,2:], bboxes[:,2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])  # [N,]
    area2 = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])  # [M,]

    include = inter / (area1[:,None])

    for i in range(bboxes.size(0)):
        include[i, i] = 0

    keep = torch.ones(bboxes.size(0))
    while True:
        values_1, i  = include.max(1)
        value, j = values_1.max(0)
        if value > thr:
            keep[j] = 0
            include[j,:] = 0.
            include[i[j], j] = 0.
        else:
            break

    return keep.type(torch.ByteTensor)

def box_nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0]
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.tensor(keep, dtype=torch.long)


def class_independent_decode(box_predictions, multi_labels, score_threshold, nms_threshold):
    scores, labels = torch.max(multi_labels, dim=1)

    mask = (scores > score_threshold) & (labels > 0)
    bboxes = box_predictions[mask]
    scores = scores[mask]
    labels = labels[mask] - 1
    multi_labels = multi_labels[mask]
    if len(bboxes):
        keep = box_nms(bboxes, scores, nms_threshold)
        return bboxes[keep], labels[keep], scores[keep], multi_labels[keep]
    else:
        return torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)


def class_dependent_decode(box_predictions, multi_labels, score_threshold, nms_threshold):
    bboxes = []
    labels = []
    scores = []
    num_classes = multi_labels.size(1)
    for i in range(num_classes - 1):
        # class i corresponds to (i + 1) column
        score = multi_labels[:, i + 1]
        mask = score > score_threshold
        if not mask.any():
            continue
        box = box_predictions[mask]
        score = score[mask]

        keep = box_nms(box, score, nms_threshold)
        bboxes.append(box[keep])
        labels.append(torch.empty_like(keep).fill_(i))
        scores.append(score[keep])
    if len(bboxes):
        return torch.cat(bboxes, 0), torch.cat(labels, 0), torch.cat(scores, 0)
    else:
        return torch.tensor([], dtype=torch.float), \
               torch.tensor([], dtype=torch.long), \
               torch.tensor([], dtype=torch.float)
