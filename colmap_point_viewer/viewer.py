"""Python API for the COLMAP point cloud Jupyter viewer."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import numpy as np


_VIEWER_HTML = (
    files("colmap_point_viewer")
    .joinpath("static/viewer.html")
    .read_text(encoding="utf-8")
)

_VIEWER_JS = (
    files("colmap_point_viewer")
    .joinpath("static/viewer.js")
    .read_text(encoding="utf-8")
)


def _as_float32_array(value: Any, name: str) -> np.ndarray:
    """Convert supported tensor/array/list inputs to a float32 NumPy array."""
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()

    try:
        return np.asarray(value, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - re-raise with the public argument name.
        raise TypeError(f"{name} must be convertible to a float32 NumPy array") from exc


def _validate_shape(array: np.ndarray, expected_tail: tuple[int, ...], name: str) -> None:
    if array.ndim != len(expected_tail) + 1 or tuple(array.shape[1:]) != expected_tail:
        expected = "[N, " + ", ".join(str(dim) for dim in expected_tail) + "]"
        if name == "c2ws":
            expected = "[M, 4, 4]"
        raise ValueError(f"{name} must have shape {expected}; got {array.shape}")


def _normalize_height(height: int | str) -> str:
    if isinstance(height, int):
        if height <= 0:
            raise ValueError("height must be positive")
        return f"{height}px"
    if isinstance(height, str) and height.strip():
        return height
    raise ValueError("height must be a positive int or a non-empty CSS size string")


def _create_widget(**kwargs: Any):
    """Create the anywidget instance lazily so importing the package has no side effects."""
    try:
        import anywidget
        import traitlets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ColmapPointCloudViewer requires the optional dependency 'anywidget'. "
            "Install it in the notebook environment before creating the viewer."
        ) from exc

    class _ColmapPointCloudWidget(anywidget.AnyWidget):
        _esm = _VIEWER_JS
        _html = _VIEWER_HTML

        pc = traitlets.List(trait=traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
        pc_color = traitlets.List(trait=traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
        c2ws = traitlets.List(default_value=[]).tag(sync=True)
        height = traitlets.Unicode(default_value="600px").tag(sync=True)
        point_size = traitlets.Float(default_value=0.01).tag(sync=True)
        background = traitlets.Unicode(default_value="#20242b").tag(sync=True)
        H = traitlets.Int().tag(sync=True)
        W = traitlets.Int().tag(sync=True)
        fx = traitlets.Float().tag(sync=True)
        fy = traitlets.Float().tag(sync=True)
        cx = traitlets.Float().tag(sync=True)
        cy = traitlets.Float().tag(sync=True)
        html_template = traitlets.Unicode(default_value=_VIEWER_HTML).tag(sync=True)

    return _ColmapPointCloudWidget(**kwargs)


class ColmapPointCloudViewer:
    """Display a COLMAP-style point cloud and camera poses in a Jupyter cell."""

    def __init__(
        self,
        pc: Any,
        pc_color: Any,
        c2ws: Any,
        H: int,
        W: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        *,
        height: int | str = 600,
        point_size: float = 0.01,
        background: str = "#20242b",
        max_points: int | None = None,
    ):
        pc_array = _as_float32_array(pc, "pc")
        color_array = _as_float32_array(pc_color, "pc_color")
        c2w_array = _as_float32_array(c2ws, "c2ws")

        _validate_shape(pc_array, (3,), "pc")
        _validate_shape(color_array, (3,), "pc_color")
        _validate_shape(c2w_array, (4, 4), "c2ws")

        if color_array.shape[0] != pc_array.shape[0]:
            raise ValueError(
                "pc_color must contain one RGB color per point; "
                f"got {color_array.shape[0]} colors for {pc_array.shape[0]} points"
            )

        if max_points is not None:
            if max_points <= 0:
                raise ValueError("max_points must be positive when provided")
            if pc_array.shape[0] > max_points:
                indices = np.linspace(0, pc_array.shape[0] - 1, max_points, dtype=np.int64)
                pc_array = pc_array[indices]
                color_array = color_array[indices]

        if color_array.size and float(np.nanmax(color_array)) > 1.0:
            color_array = color_array / np.float32(255.0)
        color_array = np.clip(color_array, 0.0, 1.0).astype(np.float32, copy=False)

        self.widget = _create_widget(
            pc=pc_array.astype(np.float32, copy=False).tolist(),
            pc_color=color_array.tolist(),
            c2ws=c2w_array.astype(np.float32, copy=False).tolist(),
            H=int(H),
            W=int(W),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            height=_normalize_height(height),
            point_size=float(point_size),
            background=str(background),
        )

    def show(self) -> None:
        """Render the widget in the current Jupyter output cell."""
        from IPython.display import display

        display(self.widget)
