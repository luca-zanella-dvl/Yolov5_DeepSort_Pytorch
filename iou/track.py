class TrackState:
    Active = 1
    Finished = 2

class Track:
    """
    A single target track with state space `(x, y, a, h)`, where `(x, y)` 
    is the center of the bounding box, `a` is the aspect ratio and `h` is 
    the height.

    Parameters
    ----------
    xyah : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    track_id : int
        A unique track identifier.
    max_score : float
        The maximum detection score.

    Attributes
    ----------
    xyah : ndarray
        The current bounding box.
    track_id : int
        A unique track identifier.
    max_score : float
        The maximum detection score.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    """
    def __init__(self, xyah, track_id, max_score):
        self.xyah = xyah
        self.track_id = track_id
        self.max_score = max_score

        self.time_since_update = 0
        self.state = TrackState.Active
        
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.xyah[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):
        self.time_since_update += 1

    def update(self, detection):
        self.xyah = detection.to_xyah()
        self.time_since_update = 0
        self.max_score = max(self.max_score, detection.confidence)

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        self.state = TrackState.Finished

    def is_active(self):
        """Returns True if this track is active."""
        return self.state == TrackState.Active

    def is_finished(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Finished