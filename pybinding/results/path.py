"""Processing and presentation of the path in the reciprocal space"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Optional, Union, List, Iterable
from matplotlib.patches import FancyArrow

from ..utils import with_defaults, pltutils

__all__ = ['make_path', 'Path', 'make_area', 'Area', 'AbstractArea']


class Path(np.ndarray):
    """A ndarray which represents a path connecting certain points

    Attributes
    ----------
    point_indices : List[int]
        Indices of the significant points along the path. Minimum 2: start and end.
    point_labels : Optional[List[str]]
        Labels for the significant points along the path.
    """
    def __new__(cls, array: ArrayLike, point_indices: List[int], point_labels: Optional[List[str]] = None):
        obj = np.asarray(array).view(cls)
        assert len(point_indices) >= 2
        obj.point_indices = point_indices
        obj.point_labels = point_labels
        return obj

    def _default_points(self, obj):
        default_indices = [0, obj.shape[0] - 1] if len(obj.shape) >= 1 else []
        default_labels = [str(i) for i in default_indices]
        return default_indices, default_labels

    def __array_finalize__(self, obj):
        if obj is None:
            return
        default_indices, default_labels = self._default_points(obj)
        self.point_indices = getattr(obj, 'point_indices', default_indices)
        self.point_labels = getattr(obj, 'point_labels', default_labels)

    def __reduce__(self):
        r = super().__reduce__()
        state = r[2] + (self.point_indices, self.point_labels,)
        return r[0], r[1], state

    # noinspection PyMethodOverriding,PyArgumentList
    def __setstate__(self, state):
        if len(state) == 7:
            self.point_indices, self.point_labels = state[-2:]
            state_out = state[:-2]
        else:
            self.point_indices = state[-1]
            self.point_labels = None
            state_out = state[:-1]
        super().__setstate__(state_out)

    @property
    def points(self) -> np.ndarray:
        """Significant points along the path, including start and end"""
        return self[self.point_indices]

    @property
    def is_simple(self) -> bool:
        """Is it just a simple path between two points?"""
        return len(self.point_indices) == 2

    def as_1d(self) -> np.ndarray:
        """Return a 1D representation of the path -- useful for plotting

        For simple paths (2 points) the closest 1D path with real positions is returned.
        Otherwise, an `np.arange(size)` is returned, where `size` matches the path. This doesn't
        have any real meaning, but it's something that can be used as the x-axis in a line plot.

        Examples
        --------
        >>> np.allclose(make_path(-2, 1, step=1).as_1d().T, [-2, -1, 0, 1])
        True
        >>> np.allclose(make_path([0, -2], [0, 1], step=1).as_1d().T, [-2, -1, 0, 1])
        True
        >>> np.allclose(make_path(1, -1, 4, step=1).as_1d().T, [0, 1, 2, 3, 4, 5, 6, 7])
        True
        """
        if self.is_simple:
            if len(self.shape) == 1:
                return self
            else:  # return the first axis with non-zero length
                return self[:, np.flatnonzero(np.diff(self.points, axis=0))[0]]
        else:
            if len(self.shape) == 1:
                return np.append([0], np.sqrt((np.diff(self, axis=0) ** 2)).cumsum())
            else:
                return np.append([0], np.sqrt((np.diff(self, axis=0) ** 2).dot(np.ones((self.shape[1], 1)))).cumsum())

    def plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None,
             **kwargs) -> FancyArrow:
        """Quiver plot of the path

        Parameters
        ----------
        point_labels : List[str]
            Labels for the :attr:`.Path.points`.
        ax : Optional[plt.Axes]
            The axis to plot on.
        **kwargs
            Forwarded to :func:`~matplotlib.pyplot.arrow`.
        """
        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')
        # TODO: plot in 3D
        default_color = pltutils.get_palette('Set1')[1]
        kwargs = with_defaults(kwargs, scale=1, zorder=2, lw=1.5, color=default_color,
                               name=None, head_width=0.08, head_length=0.2)

        out = pltutils.plot_vectors(np.diff(self.points, axis=0), self.points[0:], ax=ax, **kwargs)

        self.decorate_plot(point_labels, ax)
        return out

    def decorate_plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None) -> None:
        if ax is None:
            ax = plt.gca()
        ax.autoscale_view()
        pltutils.add_margin(0.5, ax=ax)
        pltutils.despine(trim=True, ax=ax)

        if point_labels is None:
            point_labels = self.point_labels

        if point_labels:
            for k_point, label in zip(self.points, point_labels):
                ha, va = pltutils.align(*(-k_point))
                pltutils.annotate_box(label, k_point * 1.05, fontsize='large',
                                      ha=ha, va=va, bbox=dict(lw=0), ax=ax)


def make_path(k0: ArrayLike, k1: ArrayLike, *ks: Iterable[ArrayLike], step: float = 0.1,
              point_labels: Optional[List[str]] = None) -> Path:
    """Create a path which connects the given k points

    Parameters
    ----------
    k0, k1, *ks
        Points in k-space to connect.
    step : float
        Length in k-space between two samples. Smaller step -> finer detail.
    point_labels : Optional[List[str]]
        The labels for the points.

    Examples
    --------
    >>> np.allclose(make_path(0, 3, -1, step=1).T, [0, 1, 2, 3, 2, 1, 0, -1])
    True
    >>> np.allclose(make_path([0, 0], [2, 3], [-1, 4], step=1.4),
    ...             [[0, 0], [1, 1.5], [2, 3], [0.5, 3.5], [-1, 4]])
    True
    """
    k_points = [np.atleast_1d(k) for k in (k0, k1) + ks]
    if not all(k.shape == k_points[0].shape for k in k_points[:1]):
        raise RuntimeError("All k-points must have the same shape")

    k_paths = []
    point_indices = [0]
    for k_start, k_end in zip(k_points[:-1], k_points[1:]):
        num_steps = int(np.linalg.norm(k_end - k_start) // step)
        # k_path.shape == num_steps, k_space_dimensions
        k_path = np.array([np.linspace(s, e, num_steps, endpoint=False)
                           for s, e in zip(k_start, k_end)]).T
        k_paths.append(k_path)
        point_indices.append(point_indices[-1] + num_steps)
    k_paths.append(k_points[-1])

    return Path(np.vstack(k_paths), point_indices, point_labels)


class Area(Path):
    """A ndarray which represents a area connecting certain points

    Attributes
    ----------
    point_indices : List[int]
        Indices of the significant points along the path. Minimum 2: start and end.
    point_labels : Optional[List[str]]
        Labels for the significant points along the path.
    """
    def __new__(cls, array: ArrayLike, point_indices: Optional[Union[List[List[int]], List[int]]] = None,
                point_labels: Optional[List[str]] = None):
        assert np.ndim(array) >= 3, "The area should be at least a 2D area of a 1D k-space"
        len_x, len_y = np.shape(array)[:2]
        if point_indices is None:
            point_indices = [0, int(len_x * len_y) - 1]
        if np.ndim(point_indices) >= 2:
            point_indices = np.array(point_indices, dtype=int).reshape(-1, 2)
            idx_x, idx_y = np.transpose(point_indices)
            point_indices = np.arange(len_x * len_y).reshape((len_x, len_y))[idx_x, idx_y]
        return super().__new__(cls, array, point_indices, point_labels)

    def _default_points(self, obj):
        default_indices = [0, np.prod(obj.shape[:2]) - 1] if len(obj.shape) == 2 else super()._default_points(obj)[0]
        default_labels = [str(i) for i in default_indices]
        return default_indices, default_labels

    @property
    def points(self) -> np.ndarray:
        """Significant points along the path, including start and end"""
        return self[[int(idx % self.shape[0]) for idx in self.point_indices],
        [int(idx // self.shape[0]) for idx in self.point_indices]]

    def plot(self, point_labels: Optional[List[str]] = None, ax: Optional[plt.Axes] = None,
             **kwargs) -> FancyArrow:
        if ax is None:
            ax = plt.gca()
        out = super().plot(point_labels, ax, **kwargs)
        ax.scatter(self[:, :, 0], self[:, :, 1])
        return out


class AbstractArea:
    """Abstract class to implement features to interact with the Area"""

    def __init__(self, k_area: Area):
        self.k_dims = np.shape(k_area)
        self.k_path: Optional[Path] = None  # to be overloaded
        self.data: Optional[np.ndarray] = None  # to be overloaded

    def karea_to_kpath(self, k_area: Area) -> Path:
        return Path(
            np.atleast_1d(k_area.reshape(np.prod(self.k_dims[:2]), -1)),
            k_area.point_indices,
            k_area.point_labels
        )

    def area_to_list(self, data: ArrayLike) -> np.ndarray:
        data_size = [self.k_dims[0] * self.k_dims[1]]
        data = np.atleast_2d(data)
        for ds in data.shape[2:]:
            data_size.append(ds)
        return data.reshape(data_size)

    def list_to_area(self, data: ArrayLike) -> np.ndarray:
        data_size = [self.k_dims[0], self.k_dims[1]]
        data = np.atleast_1d(data)
        for ds in data.shape[1:]:
            data_size.append(ds)
        return np.swapaxes(data.reshape(data_size), 0, 1)

    @property
    def k_area(self) -> Area:
        return Area(
            self.list_to_area(self.k_path),
            self.k_path.point_indices,
            self.k_path.point_labels
        )


def make_area(k0: ArrayLike, k1: ArrayLike, k_origin: Optional[ArrayLike] = None, step: float = 1,
              point_labels: Optional[List[str]] = None,) -> Area:
    """Create an area of k-point between k0 and k1, starting from k_origin.

    Parameters
    ----------
    k0, k1
        Point to which the area goes
    k_origin
        Point from which the area begins, default = [0, 0]
    step : float
        Length in k-space between two samples.
    point_indices : Union[List[str], List[List[str]]
        The indices that are spcial. If 1D, the N-th position in the array.flatten() will be taken.
    point_labels : List[str]
        The labels for the chosen special points.
        If None, the default values from `pb.results.Area` will be used.
    """
    k0, k1 = [np.atleast_1d(k) for k in (k0, k1)]
    if k_origin is None:
        k_origin = np.zeros(np.shape(k0))
    else:
        k_origin = np.atleast_1d(k_origin)
    if not all(k.shape == k_origin.shape for k in (k0, k1, k_origin)):
        raise RuntimeError("All k-points must have the same shape")
    num_steps = [int(np.linalg.norm(k - k_origin) // step) for k in (k0, k1)]
    subdivs = [np.linspace(0, 1, num_step) for num_step in num_steps]
    k_x, k_y = np.meshgrid(*subdivs)
    k_points = k_x[:, :, np.newaxis] * k0[np.newaxis, np.newaxis, :]
    k_points += k_y[:, :, np.newaxis] * k1[np.newaxis, np.newaxis, :]
    k_points += k_origin[np.newaxis, np.newaxis, :]
    point_indices = [0, k_points.shape[1] - 1, (k_points.shape[0] - 1) * k_points.shape[1],
                     k_points.shape[0] * k_points.shape[1] - 1]
    return Area(k_points, point_indices, point_labels)
