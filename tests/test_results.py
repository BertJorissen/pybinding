import pytest

import pickle

import numpy as np
import pybinding as pb
from pybinding.repository import graphene


#TODO: tests for bands, bands.dos, dos (add/subtract), plotting func, wavefunction, fat-bands, disentangle
@pytest.fixture(scope='module')
def model():
    return pb.Model(graphene.monolayer(), pb.rectangle(1))


def test_eigenvalues():
    eigh = np.arange(-1, 3)
    prob = np.linspace(0, .5, 4)
    prob /= np.sum(prob)
    eigenvalues = pb.results.Eigenvalues(eigh, prob)
    assert np.all(eigenvalues.indices == np.arange(4))


def test_path():
    k_begin, k_end = [-1], [2]
    point_labels = [r"$\nu$", r"$\theta$"]
    k_path_path = pb.results.Path(np.linspace(k_begin, k_end, 15), [0, 14], point_labels=point_labels)
    assert np.all(k_path_path == np.linspace(k_begin, k_end, 15))
    assert k_path_path.point_labels == point_labels
    assert np.all(k_path_path.as_1d() == np.linspace(k_begin[0], k_end[0], 15))
    assert k_path_path.is_simple
    assert np.all(k_path_path.point_indices == [0, 14])
    assert np.all(k_path_path.points == np.array([k_begin, k_end]))
    assert np.all(k_path_path.shape == (15, 1))

    k_path_make = pb.results.make_path(k_begin, k_end, step=0.2, point_labels=point_labels)
    assert np.all(k_path_path == k_path_make)
    assert k_path_path.point_labels == k_path_make.point_labels
    assert np.all(k_path_path.as_1d() == k_path_make.as_1d())
    assert k_path_path.is_simple == k_path_make.is_simple
    assert np.all(k_path_path.point_indices == k_path_path.point_indices)
    assert np.all(k_path_path.points == k_path_make.points)
    assert np.all(k_path_path.shape == k_path_make.shape)

    k_begin, k_end = [-1, 0, 1], [-1, 3, 4]
    k_path_path = pb.results.Path(np.vstack((np.linspace(-1, -1, 16), np.linspace(0, 3, 16), np.linspace(1, 4, 16))).T,
                                  [0, 15])
    assert np.all(k_path_path == np.vstack((np.linspace(-1, -1, 16), np.linspace(0, 3, 16), np.linspace(1, 4, 16))).T)
    assert k_path_path.point_labels is None
    assert np.all(k_path_path.as_1d() == np.linspace(0, 3, 16))
    assert k_path_path.is_simple
    assert np.all(k_path_path.point_indices == [0, 15])
    assert np.all(k_path_path.points == np.array([k_begin, k_end]))
    assert np.all(k_path_path.shape == (16, 3))

    k_path_make = pb.results.make_path(k_begin, k_end, step=0.27)
    assert np.all(k_path_path == k_path_make)
    assert k_path_path.point_labels == k_path_make.point_labels
    assert np.all(k_path_path.as_1d() == k_path_make.as_1d())
    assert k_path_path.is_simple == k_path_make.is_simple
    assert np.all(k_path_path.point_indices == k_path_path.point_indices)
    assert np.all(k_path_path.points == k_path_make.points)
    assert np.all(k_path_path.shape == k_path_make.shape)

    k_begin, k_mid, k_end = -1, 10, 2
    k_path_path = pb.results.Path(np.hstack((np.linspace(k_begin, k_mid, 11, endpoint=False),
                                             np.linspace(k_mid, k_end, 9))),
                                  [0, 11, 19])
    assert np.all(k_path_path == np.hstack((np.linspace(k_begin, k_mid, 11, endpoint=False),
                                            np.linspace(k_mid, k_end, 9))))
    assert k_path_path.point_labels is None
    assert np.all(k_path_path.as_1d() == np.hstack((np.linspace(0, 11, 11, endpoint=False), np.linspace(11, 19, 9))))
    assert not k_path_path.is_simple
    assert np.all(k_path_path.point_indices == [0, 11, 19])
    assert np.all(k_path_path.points == np.array([k_begin, k_mid, k_end]))
    assert np.all(k_path_path.shape == (20,))

    k_path_make = pb.results.make_path(k_begin, k_mid, k_end, step=0.95)[:, 0]
    assert np.all(k_path_path == k_path_make)
    assert k_path_path.point_labels == k_path_make.point_labels
    assert np.all(k_path_path.as_1d() == k_path_make.as_1d())
    assert k_path_path.is_simple == k_path_make.is_simple
    assert np.all(k_path_path.point_indices == k_path_path.point_indices)
    assert np.all(k_path_path.points == k_path_make.points)
    assert np.all(k_path_path.shape == k_path_make.shape)

    k_begin, k_mid, k_end = [-1, 0, 1], [-1, 0, 2], [1, 1, 4]
    k_path_path = pb.results.Path(np.vstack((
        np.vstack((np.linspace(-1, -1, 1, endpoint=False), np.linspace(0, 0, 1, endpoint=False),
                   np.linspace(1, 2, 1, endpoint=False))).T,
        np.vstack((np.linspace(-1, 1, 6), np.linspace(0, 1, 6), np.linspace(2, 4, 6))).T)),
        [0, 1, 6])
    assert np.all(k_path_path == np.vstack((
        np.vstack((np.linspace(-1, -1, 1, endpoint=False), np.linspace(0, 0, 1, endpoint=False),
                   np.linspace(1, 2, 1, endpoint=False))).T,
        np.vstack((np.linspace(-1, 1, 6), np.linspace(0, 1, 6), np.linspace(2, 4, 6))).T)))
    assert k_path_path.point_labels is None
    assert np.all(np.abs(k_path_path.as_1d() - np.hstack((np.linspace(0, 1, 1, endpoint=False),
                                                          np.linspace(1, 4, 6)))) < 1e-6)
    assert not k_path_path.is_simple
    assert np.all(k_path_path.point_indices == [0, 1, 6])
    assert np.all(k_path_path.points == np.array([k_begin, k_mid, k_end]))
    assert np.all(k_path_path.shape == (7, 3))
    k_path_make = pb.results.make_path(k_begin, k_mid, k_end, step=0.6)
    assert np.all(k_path_path == k_path_make)
    assert k_path_path.point_labels == k_path_make.point_labels
    assert np.all(k_path_path.as_1d() == k_path_make.as_1d())
    assert k_path_path.is_simple == k_path_make.is_simple
    assert np.all(k_path_path.point_indices == k_path_path.point_indices)
    assert np.all(k_path_path.points == k_path_make.points)
    assert np.all(k_path_path.shape == k_path_make.shape)


def test_area():
    k_origin, k_left, k_right = np.array([-1, 2, 3]), np.array([-1, 0, 1]), np.array([1, 1, -1])
    k_array = np.array(
        [[[-1.        ,  2.        ,  3.        ],
          [-2.        ,  2.        ,  4.        ]],
         [[-0.66666667,  2.33333333,  2.66666667],
          [-1.66666667,  2.33333333,  3.66666667]],
         [[-0.33333333,  2.66666667,  2.33333333],
          [-1.33333333,  2.66666667,  3.33333333]],
         [[ 0.        ,  3.        ,  2.        ],
          [-1.        ,  3.        ,  3.        ]]]
    )
    k_area_area = pb.results.Area(k_array)
    k_area_make = pb.results.make_area(k_left, k_right, k_origin)
    assert np.all(np.abs(k_area_area - k_area_make) < 1e-6)

def test_sweep():
    x0 = np.arange(3)
    y0 = np.arange(-1, 2)
    data0 = np.arange(9).reshape((3, 3))
    sweep = pb.results.Sweep(
        x0, y0, data0, tags=dict(b=1, c=2),
        labels=dict(title="test sweep", x="$\\alpha$", y=r"$\beta$ (eV)", data=r"$\gamma$")
    )

    assert sweep._plain_labels == dict(title="test sweep", x="alpha", y="beta (eV)", data="gamma")

    xgrid, ygrid = sweep._xy_grids()
    assert np.all(xgrid == [[v] * 3 for v in x0])
    assert np.all(ygrid == [y0] * 3)

    tmp = sweep[np.ix_([0, 1], [1, 2])]
    assert np.all(tmp.x == [[0], [1]]) and np.all(tmp.y == [0, 1])
    assert np.all(tmp.data == [[1, 2], [4, 5]])

    tmp = sweep.cropped(x=[0, 1])
    assert np.all(tmp.x == [0, 1]) and np.all(tmp.y == sweep.y)
    assert np.all(tmp.data == [[0, 1, 2], [3, 4, 5]])

    tmp = sweep.cropped(y=[0, 1])
    assert np.all(tmp.x == sweep.x) and np.all(tmp.y == [0, 1])
    assert np.all(tmp.data == [[1, 2], [4, 5], [7, 8]])

    tmp = sweep.cropped(x=[0, 1], y=[0, 1])
    assert np.all(tmp.x == [0, 1]) and np.all(tmp.y == [0, 1])
    assert np.all(tmp.data == [[1, 2], [4, 5]])

    tmp = sweep.mirrored(axis='x')
    assert np.all(tmp.x == [-2, -1, 0, 1, 2])
    assert np.all(tmp.data == [[6, 7, 8], [3, 4, 5], [0, 1, 2], [3, 4, 5], [6, 7, 8]])

    tmp = sweep.interpolated(mul=[2, 2])
    assert tmp.x.shape[0] == 6
    assert tmp.y.shape[0] == 6
    assert tmp.data.shape == (6, 6)

    s, x = sweep._slice_x(1.2)
    assert np.all(s == [3, 4, 5]) and x == 1
    s, y = sweep._slice_y(0.4)
    assert np.all(s == [1, 4, 7]) and y == 0


def test_spatial_map(model):
    system = model.system
    zeros = np.linspace(-10, 10, system.num_sites)

    spatial_map = model.structure_map(zeros).spatial_map

    assert system.x.data == spatial_map.x.data
    assert system.y.data == spatial_map.y.data
    assert system.z.data == spatial_map.z.data
    assert system.sublattices.data == spatial_map.sublattices.data

    tmp = spatial_map[spatial_map.sublattices == 'A']
    assert len(spatial_map.x) == 2 * len(tmp.x)

    tmp = spatial_map.cropped(x=(-0.1, 0.1), y=(0, 0.1))
    assert len(tmp.x) == 1

    tmp = spatial_map.clipped(0, 10)
    assert np.all(tmp.data >= 0)


def test_structure_map(model):
    system = model.system
    zeros = np.zeros_like(system.x)

    spatial_map = pb.results.SpatialMap(zeros, system.positions, system.sublattices)
    structure_map = system.with_data(zeros)

    assert pytest.fuzzy_equal(spatial_map.data, structure_map.spatial_map.data)
    assert pytest.fuzzy_equal(spatial_map.positions, structure_map.spatial_map.positions)
    assert pytest.fuzzy_equal(spatial_map.sublattices, structure_map.spatial_map.sublattices)

    tmp = structure_map[structure_map.x < 0.05]
    assert structure_map.hoppings.nnz == 41
    assert tmp.hoppings.nnz == 21
    assert tmp.hoppings.tocsr().data.mapping == model.system.impl.hopping_registry.name_map


def test_structure_map_plot(compare_figure):
    import matplotlib.pyplot as plt

    model = pb.Model(graphene.monolayer(), pb.rectangle(0.8))
    structure_map = model.structure_map(model.system.x * model.system.y)

    with compare_figure() as chk:
        structure_map.plot(site_radius=(0.03, 0.05))
        plt.gca().set_aspect("equal", "datalim")
    assert chk.passed


def test_path_pickle():
    k_path = pb.make_path(0, 1, 2)
    s = pickle.dumps(k_path)
    loaded = pickle.loads(s)

    assert np.all(k_path == loaded)
    assert k_path.point_indices == loaded.point_indices
