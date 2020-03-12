import os
import uuid
import json
import numpy as np
from pathlib import Path
from youtrain.callbacks import ModelSaver, TensorBoard, Callbacks, Logger


def generate_hash():
    return uuid.uuid4().hex[:8]


def save_report(path, epoch_report):
    path = os.path.join(path, 'report.json')
    if os.path.exists(path):
        with open(path, 'r') as fp:
            report = json.load(fp)
    else:
        report = {}
        report.update(epoch_report)

    with open(path, 'w') as fp:
        json.dump(report, fp)


def create_callbacks(name, dumps, name_save, monitor_metric):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    save_dir = Path(dumps['path']) / name
    callbacks = Callbacks([
        Logger(log_dir),
        ModelSaver(
            checkpoint=True,
            metric_name=monitor_metric,
            save_dir=save_dir,
            save_every=1,
            save_name=name_save,
            best_only=True,
            threshold=0.5),
    ])
    return callbacks


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

