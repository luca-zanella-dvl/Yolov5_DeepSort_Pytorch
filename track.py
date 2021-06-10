import sys

sys.path.insert(0, "./yolov5")

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from helpers.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from iou.iou_tracker import IOUTracker
# from iou_kf.iou_kf_tracker import IOUKFTracker
from sort.sort import Sort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import json
import constants
import csv


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = [int(i) for i in box]
        top = y1
        left = x1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [left, top, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs
    

def bbox_rel(*xyxy):
    """ " Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = "{}{:d}".format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )
    return img


def detect(cfg, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = (
        cfg.OUTPUT,
        cfg.SOURCE,
        cfg.WEIGHTS,
        cfg.VIEW_IMG,
        cfg.SAVE_TXT,
        cfg.IMAGE_SIZE
    )
    webcam = (
        source == "0"
        or source.startswith("rtsp")
        or source.startswith("http")
        or source.endswith(".txt")
    )

    # Initialize
    device = select_device(cfg.DEVICE)

    # Create output folder
    exp_id = cfg.EXP_ID
    out = os.path.join(out, "exp_" + exp_id)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)["model"].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != "cpu" else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + "/results.txt"

    seen_vid_paths = {}

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if path not in seen_vid_paths:
            seen_vid_paths[path] = True
            if cfg.TRACKER == constants.DEEP_SORT:
                # initialize deepsort
                tracker = DeepSort(
                    cfg.DEEP_SORT.REID_CKPT,
                    max_dist=cfg.DEEP_SORT.MAX_DIST,
                    min_confidence=cfg.DEEP_SORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEP_SORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEP_SORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEP_SORT.MAX_AGE,
                    n_init=cfg.DEEP_SORT.N_INIT,
                    nn_budget=cfg.DEEP_SORT.NN_BUDGET,
                    use_cuda=True,
                )
            elif cfg.TRACKER == constants.IOU:
                sigma_iou = 1 - cfg.IOU.MAX_IOU_DISTANCE
                # initialize iou
                tracker = IOUTracker(
                    sigma_l=cfg.IOU.MIN_CONFIDENCE,
                    sigma_iou=sigma_iou,
                )
            elif cfg.TRACKER == constants.SORT:
                # initialize sort
                tracker = Sort(
                    min_confidence=cfg.SORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.SORT.MAX_IOU_DISTANCE,
                    max_age=cfg.SORT.MAX_AGE,
                    n_init=cfg.SORT.N_INIT,
                )

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=cfg.AUGMENT)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            cfg.CONF_THRES,
            cfg.IOU_THRES,
            classes=cfg.CLASSES,
            agnostic=cfg.AGNOSTIC_NMS,
        )
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s

            s += "%gx%g " % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to iou/sort/deepsort tracker
                outputs = tracker.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    if save_txt:
                        fieldnames = [
                            "frame",
                            "id",
                            "bb_left",
                            "bb_top",
                            "bb_width",
                            "bb_height",
                            "conf",
                            "x",
                            "y",
                            "z",
                        ]

                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_left = tlwh_bbox[0]
                            bbox_top = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                                writer.writerow({
                                    "frame": frame_idx + 1,
                                    "id": identity,
                                    "bb_left": float(bbox_left),
                                    "bb_top": float(bbox_top),
                                    "bb_width": float(bbox_w),
                                    "bb_height": float(bbox_h),
                                    "conf": -1,
                                    "x": -1,
                                    "y": -1,
                                    "z": -1,
                                })
            else:
                tracker.increment_ages()

            # Print time (inference + NMS)
            print("%sDone. (%.3fs)" % (s, t2 - t1))

            # Stream results
            view_img = cfg.VIEW_IMG
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print("saving img!")
                if dataset.mode == "images":
                    cv2.imwrite(save_path, im0)
                else:
                    print("saving video!")
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        fps = cfg.FPS
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*cfg.FOURCC),
                            fps,
                            (w, h),
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        print("Results saved to %s" % os.getcwd() + os.sep + out)
        if platform == "darwin":  # MacOS
            os.system("open " + save_path)

    print("Done. (%.3fs)" % (time.time() - t0))


def main(config):
    cfg = get_config(config)
    cfg.IMAGE_SIZE = check_img_size(cfg.IMAGE_SIZE)
    print(cfg)

    with torch.no_grad():
        detect(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)