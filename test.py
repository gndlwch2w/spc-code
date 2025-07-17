import re
import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from medpy import metric
from scipy.ndimage import zoom
from loguru import logger
from torch.utils.data import DataLoader
from networks import net_factory
from dataloaders import BaseDataSets

np.bool = bool
np.set_printoptions(precision=4, suppress=True)

METRIC_LIST = ['dice','asd']
ACDC_IDX2CLS = ['BG', 'Myo', 'LV', 'RV']
ACDC_COLORMAP = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0]
]
SYNAPSE_IDX2CLS = ['BG', 'Aorta', 'GB', 'KL', 'KR', 'Liver', 'PC', 'SP', 'SM']
SYNAPSE_COLORMAP = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [63, 208, 244],
    [241, 240, 234],
]

def calculate_metric_per_case(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() > 0 and pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, asd
    elif gt.sum() == 0 and pred.sum() == 0:
        return 1, 0
    else:
        return 0, 0
    
def save_image_label(volume, label, save_dir, alpha, colormap):
    """Save the volume and label as images."""
    volume = (volume * 255).astype(np.uint8)
    label = label.astype(np.uint8)
    for d in range(volume.shape[0]):
        x, y = volume[d], label[d]
        if not y.any():
            continue
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        overlay = x.copy()
        for i, color in enumerate(colormap, 1):
            mask = (y == i)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            color_layer = np.ones_like(x, dtype=np.uint8) * np.array(color, dtype=np.uint8).reshape(1, 1, 3)
            overlay = np.where(mask_3ch, (alpha * color_layer + (1 - alpha) * overlay).astype(np.uint8), overlay)
        cv2.imwrite(os.path.join(save_dir, f'{d}.png'), overlay)

def test_single_volume(image, label, model, classes, patch_size, gt_save_dir, pred_save_dir, alpha, colormap):
    """Test a single volume and save the results."""
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if len(output) > 1:
                output = output[0]
            out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    save_image_label(image, label, gt_save_dir, alpha, colormap)
    save_image_label(image, prediction, pred_save_dir, alpha, colormap)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_per_case(prediction == i, label == i))
    return np.array(metric_list)  # [c, 2]

def test(output_dir, base_dir, idx2cls, colormap, num_classes, patch_size):
    """Test the model on the dataset."""
    log_id = logger.add(os.path.join(output_dir, 'test.log'), level='INFO')

    # model
    ckpt = glob(os.path.join(output_dir, f'*_best_model.pth'))[0]
    net_type = re.findall('(.*)?_best_model\.pth', os.path.basename(ckpt))[0]
    checkpoint = torch.load(ckpt, map_location='cpu')
    model = net_factory(net_type, in_channels=1, num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    # data
    test_dataset = BaseDataSets(base_dir=base_dir, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True
    )

    total_metrics = 0.0
    for case in tqdm(test_loader):
        casename = case['casename'][0]
        gt_save_dir = os.path.join(output_dir, 'gt', casename)
        os.makedirs(gt_save_dir, exist_ok=True)
        pred_save_dir = os.path.join(output_dir, 'pred', casename)
        os.makedirs(pred_save_dir, exist_ok=True)
        metrics = test_single_volume(
            image=case['image'],
            label=case['label'],
            model=model,
            classes=num_classes,
            patch_size=patch_size,
            gt_save_dir=gt_save_dir,
            pred_save_dir=pred_save_dir,
            alpha=0.5,
            colormap=colormap
        )
        logger.info(f'{casename}: \n {metrics}')
        total_metrics += metrics

    avg_metrics = total_metrics / len(test_dataset)  # [c, 2]
    for cls in range(1, num_classes):
        for i, metric_name in enumerate(METRIC_LIST):
            logger.info(f'{idx2cls[cls]} mean {metric_name}: {avg_metrics[cls - 1, i]:.4f}')

    print(avg_metrics)
    performances = np.mean(avg_metrics, axis=0)
    print(performances)
    for i, metric_name in enumerate(METRIC_LIST):
        logger.info(f'Mean {metric_name}: {performances[i]:.4f}')

    logger.info("Testing Finished")
    logger.remove(log_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory of the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--patch_size', type=list, default=[256, 256], help='Patch size for testing')

    args = parser.parse_args()
    if "ACDC" in args.base_dir:
        test(
            output_dir=args.output_dir,
            base_dir=args.base_dir,
            idx2cls=ACDC_IDX2CLS,
            colormap=ACDC_COLORMAP,
            num_classes=4,
            patch_size=args.patch_size
        )
    elif "Synapse" in args.base_dir:
        test(
            output_dir=args.output_dir,
            base_dir=args.base_dir,
            idx2cls=SYNAPSE_IDX2CLS,
            colormap=SYNAPSE_COLORMAP,
            num_classes=9,
            patch_size=args.patch_size
        )
    else:
        raise ValueError("Unsupported dataset. Please specify a valid base directory.")
