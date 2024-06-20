Wavefunction calculations
=========================

.. meta::
   :description: Using the eigenvectors in the tight-binding calculation

This section will introduce :class:`.Wavefunction` which can be used to calculate effects that are connected
to the properties in the eigenvectors, such as disentangling the bands, calculating the projected DOS, the
spatial DOS and the Berry curvature.

.. note::
    For some of these properties, access to the :class:`.Model`-object is needed.
    As a it is not possible to store a :class:`.Model`, save the required data during the calculation.

Berry curvature
Hamiltonian matrix is created.

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


Berry
-----

.. plot::
    :context:
    :alt: Berry phase calculation for MoS2


    from pybinding.repository.group6_tmd import monolayer_3band
    lat = monolayer_3band(name="MoS2")
    wfc_area = pb.solver.lapack(pb.Model(lat, pb.translational_symmetry())).calc_wavefunction_area(
    pb.make_area(*lat.reciprocal_vectors(), step=.1)
    )
    berry = pb.Berry(wfc_area)
    series_area = berry.calc_berry()
    series_area.area_plot()
    lat.plot_brillouin_zone()
    plt.show()

et voila
