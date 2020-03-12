import os
import json
import rasterio
from rasterio.mask import mask


def load_data(path):
    if not os.path.exists(os.path.join(path, 'train_labels.json')) or \
            not os.path.exists(os.path.join(path, 'test_labels.json')):
        raise Exception(f'Not found labels in {path}')

    with open(os.path.join(path, 'train_labels.json')) as fp:
        train = json.load(fp)

    with open(os.path.join(path, 'test_labels.json')) as fp:
        test = json.load(fp)

    return train, test


def load_tiff(tiff_path):
    tiff = rasterio.open(tiff_path)
    return tiff


def extract_item(path, id, data=None):
    item = None
    tiff_path = None

    def _find_element_by_id(d, id):
        return [x for x in d if x['id'] == id]

    if data is None:
        data = load_data(path)
    train, test = data

    for current_element in train + test:
        item = _find_element_by_id(current_element['features'], id)
        if len(item):
            tiff_path = os.path.join(path, current_element['image_path'])
            break

    if not len(item):
        raise Exception(f'Id {id} did not found')

    return item[0], tiff_path


def mask_map(item, tiff_image=None, tiff_path=None):
    polygon = {
        'type': 'Polygon',
        'coordinates': item['coordinates']
    }

    if tiff_path is not None:
        tiff_image = load_tiff(tiff_path)

    out_image, out_transform = mask(tiff_image, [polygon], crop=True, filled=False)
    return out_image
