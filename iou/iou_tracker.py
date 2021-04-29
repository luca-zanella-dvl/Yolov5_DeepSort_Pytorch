import numpy as np
import torch

from .linear_assignment import min_cost_matching
from .iou_matching import iou_cost
from .detection import Detection
from .track import Track


class IOUTracker(object):
    def __init__(self, min_confidence=0.3, n_init=3, max_iou_distance=0.7, max_age=70):
        self.min_confidence = min_confidence  # sigma_l
        self.n_init = n_init  # t_min
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age  # ttl

        self.tracks = []
        self._next_id = 1

    def update(self, bbox_xywh, confidences, ori_img):
        self.height, self.width = ori_img.shape[:2]
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf)
            for i, conf in enumerate(confidences)
            if conf > self.min_confidence
        ]

        # update tracker
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # output bbox identities
        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _match(self, detections):
        # Associate tracks using IOU
        (
            matches,
            unmatched_tracks,
            unmatched_detections,
        ) = min_cost_matching(
            iou_cost, self.max_iou_distance, self.tracks, detections
        )
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        xyah = detection.to_xyah()
        self.tracks.append(Track(xyah, self._next_id, self.n_init, self.max_age))
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

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h