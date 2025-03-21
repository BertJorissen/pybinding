#include "kpm/OptimizedHamiltonian.hpp"
#include <iostream>
namespace cpb { namespace kpm {

SliceMap::SliceMap(std::vector<storage_idx_t> indices, Indices const& optimized_idx)
    : data(std::move(indices)) {
    auto find_offset = [&](ArrayXi const& idx) {
        assert(idx.size() != 0);
        auto const max_index = *std::max_element(begin(idx), end(idx));
        auto const it = std::find_if(data.begin(), data.end(),
                                     [&](storage_idx_t index) { return index > max_index; });
        assert(it != data.end());
        return static_cast<idx_t>(it - data.begin());
    };

    src_offset = find_offset(optimized_idx.src);
    dest_offset = find_offset(optimized_idx.dest);
}

struct Optimize {
    OptimizedHamiltonian& oh;
    Indices const& idx;
    Scale<> scale;

    template<class scalar_t>
    void operator()(SparseMatrixRC<scalar_t> const&) {
        if (oh.is_reordered) {
            oh.create_reordered<scalar_t>(idx, scale);
        } else {
            oh.create_scaled<scalar_t>(idx, scale);
        }

        if (oh.matrix_format == MatrixFormat::ELL) {
            auto const& csr = var::get<SparseMatrixX<scalar_t>>(oh.optimized_matrix);
            oh.optimized_matrix = num::csr_to_ell(csr);
        }

        oh.tag = var::tag<scalar_t>{};
    }
};

void OptimizedHamiltonian::optimize_for(Indices const& idx, Scale<> scale) {
    if (original_idx == idx) {
        return; // already optimized for this idx
    }

    timer.tic();
    var::visit(Optimize{*this, idx, scale}, original_h.get_variant());
    timer.toc();

    original_idx = idx;
}

template<class scalar_t>
void OptimizedHamiltonian::create_scaled(Indices const& idx, Scale<> s) {
    using real_t = num::get_real_t<scalar_t>;
    auto const scale = Scale<real_t>(s);

    auto const& h = ham::get_reference<scalar_t>(original_h);
    auto h2 = SparseMatrixX<scalar_t>();
    if (scale.b == 0) { // just scale, no b offset
        h2 = h * (2 / scale.a);
    } else { // scale and offset
        auto I = SparseMatrixX<scalar_t>{h.rows(), h.cols()};
        I.setIdentity();
        h2 = (h - I * scale.b) * (2 / scale.a);
    }
    h2.makeCompressed();

    optimized_matrix = h2.markAsRValue();
    optimized_idx = idx;
}

template<class scalar_t>
void OptimizedHamiltonian::create_reordered(Indices const& idx, Scale<> s) {
    using real_t = num::get_real_t<scalar_t>;
    auto scale = Scale<real_t>(s);

    auto const& h = ham::get_reference<scalar_t>(original_h);
    auto const system_size = h.rows();
    auto const inverted_a = real_t{2 / scale.a};

    auto h2 = SparseMatrixX<scalar_t>(system_size, system_size);
    // Reserve the same nnz per row as the original + 1 in case the scaling adds diagonal elements
    h2.reserve(VectorX<idx_t>::Constant(system_size, sparse::max_nnz_per_row(h) + 1));

    // Note: The following "queue" and "map" use vectors instead of other container types because
    //       they serve a very simple purpose. Using preallocated vectors results in better
    //       performance (this is not an assumption, it has been tested).

    // The index queue will contain the indices that need to be checked next
    bool verbose = false;
    if (verbose) {
        std::cout << "h1: nnz " << h.nonZeros() << " out of " << h.size() << std::endl;
        std::cout << h.toDense() << std::endl;
    }

    auto index_queue = std::vector<storage_idx_t>();
    index_queue.reserve(system_size);
    index_queue.push_back(idx.src[0]); // starting from the given index

    // Map from original matrix indices to reordered matrix indices
    reorder_map = std::vector<storage_idx_t>(system_size, -1); // reset all to invalid state
    // The point of the reordering is to have the target become index number 0
    reorder_map[idx.src[0]] = 0;

    // As the reordered matrix is filled, the slice border indices are recorded
    auto slice_border_indices = std::vector<storage_idx_t>();
    slice_border_indices.push_back(1);

    block_diagonal_idx = std::vector<storage_idx_t>(1, 0);
    zero_row_idx = std::vector<storage_idx_t>();

    // Fill the reordered matrix row by row
    auto const h_view = sparse::make_loop(h);
    for (auto h2_row = 0; h2_row < system_size; ++h2_row) {
        auto diagonal_inserted = false;


        // Loop over elements in the row of the original matrix
        // corresponding to the h2_row of the reordered matrix

        // if the matrix is block diagonal, the energy h2_row might not be there
        // resolve this by taking the first hamiltonian element that is not yet in index_queue
        storage_idx_t row;

        bool block_diagonal = h2_row == static_cast<int>(index_queue.size());
        if (block_diagonal) {
            // sorting is OK, these elements won't be needed anymore
            std::sort(index_queue.begin(), index_queue.end());
            storage_idx_t new_index = 1;

            // find the new index, the first positive integer not in index_queue
            for (auto looping_index : index_queue)
                if (looping_index == new_index) new_index++;
            row = new_index;

            slice_border_indices.pop_back(); // the last element is a duplicate of the second to last
            slice_border_indices.shrink_to_fit();
            slice_border_indices.push_back(static_cast<storage_idx_t>(index_queue.size()) + 1);
            block_diagonal_idx.push_back(static_cast<storage_idx_t>(index_queue.size()));

            // block-diagonal, but the first element is zero --> loop won't find element --> insert by hand
            bool zero_element = true;
            h_view.for_each_in_row(row, [&](storage_idx_t col, scalar_t) {
                if (col == row) zero_element = false;
            });
            if (zero_element) {
                // the element is zero; the loop won't add the element as it should
                if (reorder_map[row] < 0) {
                    auto const h2_col =  static_cast<storage_idx_t>(index_queue.size());
                    reorder_map[row] = h2_col;
                    index_queue.push_back(row);
                    h2.insert(h2_col, h2_col) = -scale.b * inverted_a;
                    if (verbose) {
                        std::cout << "mmm   --  " << row << " -- " << h2_col << " diag  --  " << h2_row << " -- "
                                  << h2_col << std::endl;
                    }
                    diagonal_inserted = true;
                } else {
                    throw std::runtime_error("This shouldn't happen (first el block diag is zero).");
                }
            }
        } else {
            row = index_queue[h2_row];
        }

        if (verbose) {
            std::cout << "h2_row: " << h2_row << ", row: " << row << ", reorder_map: ";
            for (auto i: reorder_map) std::cout << i << " ";
            std::cout << ", index_queue: ";
            for (auto i: index_queue) std::cout << i << " ";
            std::cout << ", block_diagonal: ";
            for (auto i: block_diagonal_idx) std::cout << i << " ";
            std::cout << ", zero_row: ";
            for (auto i: zero_row_idx) std::cout << i << " ";
            std::cout << std::endl;
        }

        h_view.for_each_in_row(row, [&](storage_idx_t col, scalar_t value) {
            if (verbose) {
                std::cout << "   --  " << row << " -- " << col << std::endl;
            }
            // This may be a new index, map it
            if (reorder_map[col] < 0) {
                reorder_map[col] = static_cast<storage_idx_t>(index_queue.size());
                index_queue.push_back(col);
            }

            // Get the reordered column index
            auto const h2_col = reorder_map[col];

            // Calculate the new value that will be inserted into the scaled/reordered matrix
            auto h2_value = value * inverted_a;
            if (row == col) { // diagonal elements
                h2_value -= scale.b * inverted_a;
                diagonal_inserted = true;
                if (verbose) {
                    std::cout <<  "   --  " << row << " -- " << col << " diag  --  " << h2_row << " -- " << h2_col << std::endl;
                }
            }

            h2.insert(h2_row, h2_col) = h2_value;
        });

        if (block_diagonal) {
            // check is element is inserted, then index_queue will change
            if (verbose) {
                std::cout << "bd   --  " << row << " -- " << row ;
                std::cout << ", reorder_map: ";
                for (auto i: reorder_map) std::cout << i << " ";
                std::cout << ", index_queue: ";
                for (auto i: index_queue) std::cout << i << " ";
                std::cout << ", block_diagonal: ";
                for (auto i: block_diagonal_idx) std::cout << i << " ";
                std::cout << ", zero_row: ";
                for (auto i: zero_row_idx) std::cout << i << " ";
                std::cout << " -- " << h2_row << std::endl;
            }

            // hard insert by hand
            if (reorder_map[row] < 0) {
                if (h2_row == static_cast<int>(index_queue.size())) {
                    zero_row_idx.push_back(row);
                }
                reorder_map[row] = static_cast<storage_idx_t>(index_queue.size());
                index_queue.push_back(row);
                if (verbose) {
                    std::cout << "h2: nnz " << h2.nonZeros() << " out of " << h2.size() << std::endl;
                    std::cout << h2.toDense() << std::endl;
                    std::cout << "bd   --  " << h2_row << " -- " << h2_row << std::endl;
                }
                h2.insert(h2_row, h2_row) = scale.b * inverted_a;
                diagonal_inserted = true;
            }
        }

        // A diagonal element may need to be inserted into the reordered matrix
        // even if the original matrix doesn't have an element on the main diagonal
        if (scale.b != 0 && !diagonal_inserted) {
            if (verbose) {
                std::cout <<  "   --  " << row << " -- " << row << " diag-scale  --  " << h2_row << " -- " << h2_row << " - " << block_diagonal << std::endl;
            }
            h2.insert(h2_row, h2_row) = -scale.b * inverted_a;
        }

        // Reached the end of a slice
        if (h2_row == slice_border_indices.back() - 1) {
            slice_border_indices.push_back(static_cast<storage_idx_t>(index_queue.size()));
        }

        // Block-diagonal, but all zeros so that no element triggers reorder


    }

    if (std::find(reorder_map.begin(), reorder_map.end(), -1) != reorder_map.end()) {
        throw std::runtime_error("OptimizedHamiltonian: this should never happen. (-1 in reorder_map)");
    }

    if (verbose) {
        std::cout << "h2_row: (end), row: (end), reorder_map: ";
        for (auto i: reorder_map) std::cout << i;
        std::cout << ", index_queue: ";
        for (auto i: index_queue) std::cout << i << " ";
        std::cout << ", block_diagonal: ";
        for (auto i: block_diagonal_idx) std::cout << i;
        std::cout << ", zero_row: ";
        for (auto i: zero_row_idx) std::cout << i << " ";
        std::cout << std::endl;
    }

    if (verbose) {
        std::cout << "h2: nnz " << h2.nonZeros() << " out of " << h2.size() << std::endl;
        std::cout << h2.toDense() << std::endl;
    }
    h2.makeCompressed();
    optimized_matrix = h2.markAsRValue();

    slice_border_indices.pop_back(); // the last element is a duplicate of the second to last
    slice_border_indices.shrink_to_fit();


    optimized_idx = reorder_indices(idx, reorder_map);
    slice_map = {std::move(slice_border_indices), optimized_idx};
}

Indices OptimizedHamiltonian::reorder_indices(Indices const& original_idx,
                                              std::vector<storage_idx_t> const& map) {
    return {transform<ArrayX>(original_idx.src,  [&](storage_idx_t i) { return map[i]; }),
            transform<ArrayX>(original_idx.dest, [&](storage_idx_t i) { return map[i]; })};
}

namespace {
    /// Return the number of non-zeros present up to `rows`
    struct NonZeros {
        idx_t rows;

        template<class scalar_t>
        size_t operator()(SparseMatrixX<scalar_t> const& csr) {
            return static_cast<size_t>(csr.outerIndexPtr()[rows]);
        }

        template<class scalar_t>
        size_t operator()(num::EllMatrix<scalar_t> const& ell) {
            return static_cast<size_t>(rows * ell.nnz_per_row);
        }
    };
}

size_t OptimizedHamiltonian::num_nonzeros(idx_t num_moments, bool optimal_size) const {
    auto result = size_t{0};
    if (!optimal_size) {
        result = num_moments * var::visit(NonZeros{size()}, optimized_matrix);
    } else {
        for (auto n = 0; n < num_moments; ++n) {
            auto const opt_size = slice_map.optimal_size(n, num_moments);
            auto const num_nonzeros = var::visit(NonZeros{opt_size}, optimized_matrix);
            result += num_nonzeros;
        }
    }
    if (optimized_idx.is_diagonal()) {
        result /= 2;
    }
    return result;
}

size_t OptimizedHamiltonian::num_vec_elements(idx_t num_moments, bool optimal_size) const {
    auto result = size_t{0};
    if (!optimal_size) {
        result = num_moments * size();
    } else {
        for (auto n = 0; n < num_moments; ++n) {
            result += static_cast<size_t>(slice_map.optimal_size(n, num_moments));
        }
    }
    if (optimized_idx.is_diagonal()) {
        result /= 2;
    }
    return result;
}

namespace {
    /// Return the data size in bytes
    struct MatrixMemory {
        template<class scalar_t>
        size_t operator()(SparseMatrixX<scalar_t> const& csr) const {
            using index_t = typename SparseMatrixX<scalar_t>::StorageIndex;
            auto const nnz = static_cast<size_t>(csr.nonZeros());
            auto const row_starts = static_cast<size_t>(csr.rows() + 1);
            return nnz * sizeof(scalar_t) + nnz * sizeof(index_t) + row_starts * sizeof(index_t);
        }

        template<class scalar_t>
        size_t operator()(num::EllMatrix<scalar_t> const& ell) const {
            using index_t = typename num::EllMatrix<scalar_t>::StorageIndex;
            auto const nnz = static_cast<size_t>(ell.nonZeros());
            return nnz * sizeof(scalar_t) + nnz * sizeof(index_t);
        }
    };

    struct VectorMemory {
        template<class scalar_t>
        size_t operator()(SparseMatrixRC<scalar_t> const&) const { return sizeof(scalar_t); }
    };
}

size_t OptimizedHamiltonian::matrix_memory() const {
    return var::visit(MatrixMemory{}, optimized_matrix);
}

size_t OptimizedHamiltonian::vector_memory() const {
    return size() * var::visit(VectorMemory{}, original_h.get_variant());
}

}} // namespace cpb::kpm
