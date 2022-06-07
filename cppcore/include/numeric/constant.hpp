#pragma once
#include <complex>
#include "dense.hpp"

namespace cpb { namespace constant {
    // imaginary one
    constexpr std::complex<cpb::CartesianX> i1(0, 1);
    // the omnipresent pi
    constexpr cpb::CartesianX pi = 3.14159265358979323846f;
    // electron charge [C]
    constexpr cpb::CartesianX e = 1.602e-19f;
    // reduced Planck constant [eV*s]
    constexpr cpb::CartesianX hbar = 6.58211899e-16f;
    // electron rest mass [kg]
    constexpr cpb::CartesianX m0 = 9.10938188e-31f;
    // vacuum permittivity [F/m == C/V/m]
    constexpr cpb::CartesianX epsilon0 = 8.854e-12f;
    // magnetic flux quantum (h/e)
    constexpr cpb::CartesianX phi0 = 2 * pi*hbar;
    // Boltzmann constant
    constexpr cpb::CartesianX kb = 8.6173303e-5f;
}} // namespace cpb::constant
