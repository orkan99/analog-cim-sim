/******************************************************************************
 * Copyright (C) 2025 Rebecca Pelke                                           *
 * All Rights Reserved                                                        *
 *                                                                            *
 * This is work is licensed under the terms described in the LICENSE file     *
 * found in the root directory of this source tree.                           *
 ******************************************************************************/
#include "mapping/bnn_mapper/bnn_v.h"
#include "helper/config.h"

namespace nq {

MapperBnnV::MapperBnnV() : Mapper(false) {
    if (CFG.SPLIT.size() != 1) {
        throw std::runtime_error("BNN_V needs a split size of 1.");
    }
}

MapperBnnV::~MapperBnnV() {}

void MapperBnnV::d_write(const int32_t *mat, int32_t m_matrix,
                         int32_t n_matrix) {
    d_write_diff_bnn(mat, m_matrix, n_matrix);
}

void MapperBnnV::a_write(int32_t m_matrix, int32_t n_matrix) {
    a_write_p_m_bnn(m_matrix, n_matrix);
}

void MapperBnnV::d_mvm(int32_t *res, const int32_t *vec, const int32_t *mat,
                       int32_t m_matrix, int32_t n_matrix) {
    std::vector<int32_t> vd_p(n_matrix, 0);
    std::vector<int32_t> vd_m(n_matrix, 0);

    for (size_t n = 0; n < n_matrix; ++n) {
        if (vec[n] == +1) {
            vd_p[n] = vec[n];
            vd_m[n] = 0;
        } else if (vec[n] == -1) {
            vd_p[n] = 0;
            vd_m[n] = -vec[n];
        } else {
            throw std::runtime_error("BNN input is neither +1 nor -1.");
        }
    }

    for (size_t m = 0; m < m_matrix; ++m) {
        for (size_t n = 0; n < n_matrix; ++n) {
            res[m] += (gd_p_[m][n] * vd_p[n] - gd_m_[m][n] * vd_m[n]) << 1;
        }
    }

    for (size_t m = 0; m < m_matrix; ++m) {
        res[m] -= n_matrix;
    }
}

void MapperBnnV::a_mvm(int32_t *res, const int32_t *vec, const int32_t *mat,
                       int32_t m_matrix, int32_t n_matrix) {
    std::vector<float> tmp_out(m_matrix, 0);
    std::vector<int32_t> vd_p(n_matrix, 0);
    std::vector<int32_t> vd_m(n_matrix, 0);

    for (size_t n = 0; n < n_matrix; ++n) {
        if (vec[n] == +1) {
            vd_p[n] = vec[n];
            vd_m[n] = 0;
        } else if (vec[n] == -1) {
            vd_p[n] = 0;
            vd_m[n] = -vec[n];
        } else {
            throw std::runtime_error("BNN input is neither +1 nor -1.");
        }
    }

    for (size_t m = 0; m < m_matrix; ++m) {
        for (size_t n = 0; n < n_matrix; ++n) {
            tmp_out[m] += ia_p_[m][n] * vd_p[n] + ia_m_[m][n] * vd_m[n];
        }
    }

    for (size_t m = 0; m < m_matrix; ++m) {
        res[m] +=
            round(2 / i_mm_ * adc_->analog_digital_conversion(tmp_out[m]) -
                  n_matrix - 2 * n_matrix * CFG.HRS / i_mm_);
    }
}

} // namespace nq