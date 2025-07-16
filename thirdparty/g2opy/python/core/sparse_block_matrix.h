#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <g2o/core/sparse_block_matrix.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace g2o {

namespace {

template<class MatrixType = MatrixXD>
void templatedSparseBlockMatrix(py::module& m, const std::string& suffix) {
    using CLS = SparseBlockMatrix<MatrixType>;

    py::class_<CLS>(m, ("SparseBlockMatrix" + suffix).c_str())
        .def(py::init<>())
        .def(py::init([](const std::vector<int>& rbi, const std::vector<int>& cbi, int rb, int cb, bool has_storage) {
            return new CLS(rbi.data(), cbi.data(), rb, cb, has_storage);
        }), "row_block_indices"_a, "col_block_indices"_a,
            "num_row_blocks"_a, "num_col_blocks"_a, "has_storage"_a = true)     
        .def("clear", &CLS::clear, "dealloc"_a=false)
        .def("cols", &CLS::cols)
        .def("rows", &CLS::rows)
        .def("nonZeros", &CLS::nonZeros)
        .def("nonZeroBlocks", &CLS::nonZeroBlocks)

        .def("rows_of_block", &CLS::rowsOfBlock, "block_row"_a)
        .def("cols_of_block", &CLS::colsOfBlock, "block_col"_a)
        .def("row_base_of_block", &CLS::rowBaseOfBlock, "block_row"_a)
        .def("col_base_of_block", &CLS::colBaseOfBlock, "block_col"_a)

        // .def("block", [](const CLS& self, int r, int c) {
        //     const auto* blk = self.block(r, c);
        //     if (!blk)
        //         throw std::runtime_error("Block not allocated at position (" + std::to_string(r) + ", " + std::to_string(c) + ")");
        //     return *blk;
        // }, "r"_a, "c"_a,
        // py::return_value_policy::copy,  // copies Eigen::Matrix
        // "Get a const block (throws if not allocated)")
        .def("block", [](const CLS& self, int r, int c) {
            if (r < 0 || c < 0 || r >= static_cast<int>(self.blockCols().size()))
                throw std::out_of_range("Invalid block indices.");
            const auto* blk = self.block(r, c);
            if (!blk)
                return MatrixType();  // empty matrix if block not allocated
            return *blk;
        })        

        // .def("block_ptr", [](CLS& self, int r, int c, bool alloc) -> MatrixType& {
        //     auto* blk = self.block(r, c, alloc);
        //     if (!blk)
        //         throw std::runtime_error("Block not allocated and alloc=false at (" + std::to_string(r) + ", " + std::to_string(c) + ")");
        //     return *blk;
        // }, "r"_a, "c"_a, "alloc"_a=false,
        // py::return_value_policy::reference_internal,
        // "Access or allocate a mutable block at (r, c)")
        .def("block_ptr", [](CLS& self, int r, int c, bool alloc) -> MatrixType& {
            auto* blk = self.block(r, c, alloc);
            if (!blk)
                throw std::runtime_error("Block not allocated and alloc=False");
            return *blk;
        }, "r"_a, "c"_a, "alloc"_a = false, py::return_value_policy::reference_internal, "Access or allocate a mutable block at (r, c)")        

        .def("scale", &CLS::scale, "scale_factor"_a)

        .def("clone", &CLS::clone,
             py::return_value_policy::take_ownership,
             "Deep copy of the SparseBlockMatrix")

        .def("write_octave", &CLS::writeOctave, "filename"_a, "upperTriangle"_a=true)

        .def_property_readonly("row_block_indices", [](const CLS& self) {
            return self.rowBlockIndices();
        })

        .def_property_readonly("col_block_indices", [](const CLS& self) {
            return self.colBlockIndices();
        });
}

} // anonymous namespace


void declareSparseBlockMatrix(py::module& m) {
    templatedSparseBlockMatrix<MatrixXD>(m, "X");
}

} // namespace g2o
