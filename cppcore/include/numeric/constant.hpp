#pragma once
#include <complex>

namespace cpb { namespace constant {
    // imaginary one
    constexpr std::complex<double> i1(0, 1);
    // the omnipresent pi
    constexpr double pi = 3.14159265358979323846f;
    // electron charge [C]
    constexpr double e = 1.602e-19f;
    // reduced Planck constant [eV*s]
    constexpr double hbar = 6.58211899e-16f;
    // electron rest mass [kg]
    constexpr double m0 = 9.10938188e-31f;
    // vacuum permittivity [F/m == C/V/m]
    constexpr double epsilon0 = 8.854e-12f;
    // magnetic flux quantum (h/e)
    constexpr double phi0 = 2 * pi*hbar;
    // Boltzmann constant
    constexpr double kb = 8.6173303e-5f;
}} // namespace cpb::constant
