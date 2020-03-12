import os
import matplotlib.pyplot as plt
import numpy as np

from .utils import extract_item, mask_map


def draw_plot(path, id, data=None, pred_label=None):
    def _make_title(true_label, pred_label):
        true_label = 'N/A' if not true_label else true_label
        pred_label = 'N/A' if not pred_label else pred_label

        return f"Predicted: {pred_label}\nTrue: {true_label}"

    item, tiff_path = extract_item(
        path=path,
        id=id,
        data=data
    )

    image = mask_map(
        item=item,
        tiff_path=tiff_path
    )

    image = np.ma.transpose(image, [1, 2, 0])
    title = _make_title(item['roof_material'], pred_label)

    plt.figure()
    plt.title(title)
    plt.imshow(image)

    img_path = os.path.join(path, 'pics')
    if not os.path.exists(img_path):
        os.makedirs(img_path, exist_ok=True)

    plt.savefig(os.path.join(img_path, id + '.png'))



