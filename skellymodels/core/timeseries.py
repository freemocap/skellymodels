"""
Timeseries model for scalar quantities tracked over time.

A Timeseries is a 1D signal: one value per timestamp.
Examples: x-position of a single keypoint, a joint angle, angular speed.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator


class Timeseries(BaseModel):
    """
    A single scalar quantity tracked over time.

    Fields:
        name: identifier for this signal (e.g. "right_elbow.x", "speed")
        timestamps: (N,) array of time values in seconds
        values: (N,) array of scalar measurements
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str
    timestamps: NDArray[np.float64]
    values: NDArray[np.float64]

    @model_validator(mode="after")
    def _validate_lengths(self) -> "Timeseries":
        if self.timestamps.ndim != 1:
            raise ValueError(
                f"timestamps must be 1-dimensional, got ndim={self.timestamps.ndim}"
            )
        if self.values.ndim != 1:
            raise ValueError(
                f"values must be 1-dimensional, got ndim={self.values.ndim}"
            )
        if self.timestamps.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"timestamps length {self.timestamps.shape[0]} != "
                f"values length {self.values.shape[0]}"
            )
        if self.timestamps.shape[0] < 1:
            raise ValueError("Timeseries must have at least 1 sample")
        return self

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Total duration in seconds (0.0 if only one sample)."""
        if self.n_frames < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def mean_dt(self) -> float:
        """Mean time step in seconds (0.0 if only one sample)."""
        if self.n_frames < 2:
            return 0.0
        return self.duration / (self.n_frames - 1)

    def differentiate(self) -> "Timeseries":
        """
        Compute time derivative using central differences (interior)
        and forward/backward differences (endpoints).
        """
        n = len(self.values)
        if n < 2:
            raise ValueError("Cannot differentiate a Timeseries with fewer than 2 samples")

        derivative = np.empty(n, dtype=np.float64)

        # Forward difference at start
        dt_start = self.timestamps[1] - self.timestamps[0]
        if dt_start == 0.0:
            raise ValueError("Zero time step between timestamps[0] and timestamps[1]")
        derivative[0] = (self.values[1] - self.values[0]) / dt_start

        # Central differences for interior
        for i in range(1, n - 1):
            dt = self.timestamps[i + 1] - self.timestamps[i - 1]
            if dt == 0.0:
                raise ValueError(f"Zero time span between timestamps[{i-1}] and timestamps[{i+1}]")
            derivative[i] = (self.values[i + 1] - self.values[i - 1]) / dt

        # Backward difference at end
        dt_end = self.timestamps[-1] - self.timestamps[-2]
        if dt_end == 0.0:
            raise ValueError(f"Zero time step between timestamps[{n-2}] and timestamps[{n-1}]")
        derivative[-1] = (self.values[-1] - self.values[-2]) / dt_end

        return Timeseries(
            name=f"d({self.name})/dt",
            timestamps=self.timestamps,
            values=derivative,
        )

    def interpolate(self, target_timestamps: NDArray[np.float64]) -> "Timeseries":
        """Linearly interpolate to new timestamps."""
        interpolated = np.interp(target_timestamps, self.timestamps, self.values)
        return Timeseries(
            name=self.name,
            timestamps=target_timestamps,
            values=interpolated,
        )

    def __str__(self) -> str:
        return f"Timeseries(name={self.name!r}, n_frames={self.n_frames})"

    def __repr__(self) -> str:
        return self.__str__()
