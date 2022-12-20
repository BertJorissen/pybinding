#pragma once
#include <complex>
#include "dense.hpp"

namespace cpb { namespace constant {
    // imaginary one
    constexpr std::complex<cpb::CartesianX> i1(0, 1);
    // the omnipresent pi
    constexpr cpb::CartesianX pi = 3.14159265358979323846;
    // electron charge [C]
    constexpr cpb::CartesianX e = 1.602176634e-19;
    // reduced Planck constant [eV*s]
    constexpr cpb::CartesianX hbar = 6.58211957e-16;
    // electron rest mass [kg]
    constexpr cpb::CartesianX m0 = 9.1093837015e-31;
    // vacuum permittivity [F/m == C/V/m]
    constexpr cpb::CartesianX epsilon0 = 8.85418781762039e-12;
    // magnetic flux quantum (h/e)
    constexpr cpb::CartesianX phi0 = 2 * pi*hbar;
    // Boltzmann constant
    constexpr cpb::CartesianX kb = 8.617333262e-5;
}} // namespace cpb::constant
