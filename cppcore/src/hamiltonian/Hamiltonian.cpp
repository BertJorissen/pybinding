#include "hamiltonian/Hamiltonian.hpp"

namespace cpb {
namespace {

struct IsValid {
    template<class scalar_t>
    bool operator()(SparseMatrixRC<scalar_t> const& p) const { return p != nullptr; }
};

struct Reset {
    template<class scalar_t>
    void operator()(SparseMatrixRC<scalar_t>& p) const { p.reset(); }
};

struct GetSparseRef {
    template<class scalar_t>
    ComplexCsrConstRef operator()(SparseMatrixRC<scalar_t> const& m) const { return csrref(*m); }
};

struct NonZeros {
    template<class scalar_t>
    idx_t operator()(SparseMatrixRC<scalar_t> const& m) const { return m->nonZeros(); }
};

struct Rows {
    template<class scalar_t>
    idx_t operator()(SparseMatrixRC<scalar_t> const& m) const { return m->rows(); }
};

struct Cols {
    template<class scalar_t>
    idx_t operator()(SparseMatrixRC<scalar_t> const& m) const { return m->cols(); }
};

} // namespace

Hamiltonian::operator bool() const {
    return var::visit(IsValid(), variant_matrix);
}

void Hamiltonian::reset() {
    return var::visit(Reset(), variant_matrix);
}

ComplexCsrConstRef Hamiltonian::csrref() const {
    return var::visit(GetSparseRef(), variant_matrix);
}

idx_t Hamiltonian::non_zeros() const {
    return var::visit(NonZeros(), variant_matrix);
}

idx_t Hamiltonian::rows() const {
    return var::visit(Rows(), variant_matrix);
}

idx_t Hamiltonian::cols() const {
    return var::visit(Cols(), variant_matrix);
}

} // namespace cpb
