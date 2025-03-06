/******************************************************************************
 * Copyright (C) 2025 Rebecca Pelke                                           *
 * All Rights Reserved                                                        *
 *                                                                            *
 * This is work is licensed under the terms described in the LICENSE file     *
 * found in the root directory of this source tree.                           *
 ******************************************************************************/
#ifndef MAPPER_H
#define MAPPER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "adc/adc.h"

namespace nq {

class Mapper {
  public:
    Mapper(bool is_diff_weight_mapping);
    Mapper(const Mapper &) = delete;
    virtual ~Mapper() = default;

    virtual void d_write(const int32_t *mat, int32_t m_matrix,
                         int32_t n_matrix) = 0;
    virtual void a_write(int32_t m_matrix, int32_t n_matrix) = 0;
    virtual void d_mvm(int32_t *res, const int32_t *vec, const int32_t *mat,
                       int32_t m_matrix, int32_t n_matrix) = 0;
    virtual void a_mvm(int32_t *res, const int32_t *vec, const int32_t *mat,
                       int32_t m_matrix, int32_t n_matrix) = 0;
    static std::unique_ptr<Mapper> create_from_config();

  protected:
    void d_write_diff(const int32_t *mat, int32_t m_matrix, int32_t n_matrix);
    void d_write_diff_bnn(const int32_t *mat, int32_t m_matrix,
                          int32_t n_matrix);
    void d_write_offs(const int32_t *mat, int32_t m_matrix, int32_t n_matrix);
    void a_write_p_m(int32_t m_matrix, int32_t n_matrix);
    void a_write_p_m_bnn(int32_t m_matrix, int32_t n_matrix);
    void a_write_p(int32_t m_matrix, int32_t n_matrix);
    void a_write_p_bnn(int32_t m_matrix, int32_t n_matrix);

    bool is_diff_weight_mapping_;

    // Parameters for the digital crossbar
    std::vector<std::vector<int32_t>> gd_p_;
    std::vector<std::vector<int32_t>> gd_m_;
    std::vector<uint32_t> shift_;
    std::vector<int32_t> sum_w_;

    // Parameters for the analog crossbar
    std::vector<std::vector<float>> ia_p_;
    std::vector<std::vector<float>> ia_m_;
    std::vector<float> i_step_size_;
    int num_segments_;
    float i_mm_;
    const std::unique_ptr<ADC> adc_;
};

} // namespace nq

#endif
