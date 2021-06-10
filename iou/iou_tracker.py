import numpy as np
import torch

from .detection import Detection
from .track import Track
from .iou_matching import iou


class IOUTracker(object):
    """
    Parameters
    ----------
    sigma_l : float
        Low detection threshold.
    sigma_h : float
        High detection threshold.
    sigma_iou : float
        IOU threshold.
    t_min : int
        Minimum track length in frames.

    """
    # def __init__(self, sigma_l, sigma_h, sigma_iou, t_min):
    def __init__(self, sigma_l, sigma_iou):
        self.sigma_l = sigma_l
        # self.sigma_l = sigma_l  # min_confidence
        # self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        # self.t_min = t_min

        self.tracks = []
        self._next_id = 1

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf)
            for i, conf in enumerate(confidences)
            if conf > self.sigma_l
        ]

        for track in self.tracks:
            track.increment_age()

            if len(detections) > 0:
                # get det with highest iou
                bbox = track.to_tlwh()
                candidates = np.asarray([d.tlwh for d in detections])
                best_match_idx = np.argmax(iou(bbox, candidates))
                best_match = detections[best_match_idx]

                if iou(bbox, np.array([best_match.tlwh]))[0] >= self.sigma_iou:
                    track.update(best_match)
                    # remove best matching detection from detections
                    detections = np.delete(detections, best_match_idx)

            if track.time_since_update > 0:
                # if track was not updated
                track.mark_missed()

        for detection in detections:
            self._initiate_track(detection)

        self.tracks = [t for t in self.tracks if t.is_active()]

        # output bbox identities
        outputs = []
        for track in self.tracks:
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs

    def _initiate_track(self, detection):
        xyah = detection.to_xyah()
        conf = detection.confidence
        self.tracks.append(Track(xyah, self._next_id, conf))
        self._next_id += 1

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2