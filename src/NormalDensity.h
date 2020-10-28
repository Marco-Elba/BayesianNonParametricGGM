#include <RcppEigen.h>

// Start of the header guard
#ifndef NORMALDENSITY_H
#define NORMALDENSITY_H

void my_dnorm(float x, float mu, float tau2, float &out_dnorm);

void my_ldnorm(float x, float mu, float tau2, float &out_dnorm);

void lmvdnorm(const Eigen::VectorXf &x, const Eigen::VectorXf &Mean, const Eigen::MatrixXf &Cov,
              float &out_density);

void lmvdnorm_chol(const Eigen::VectorXf &x, const Eigen::VectorXf &Mean,
                   const Eigen::LLT<Eigen::MatrixXf> &Cov_chol, const Eigen::MatrixXf &Cov_chol_matrix,
                   float &out_density);

void sum_ldbeta(const std::vector<float> &x, float a, float b, float &out_sum);

void my_dbeta(float x, float a, float b, float &out_dbeta);

void ldbeta(float x, float a, float b, float &out_ldbeta);

void ldgamma(float x, float a, float b, float &out_gamma);

void sum_ldgamma(const std::vector<float> &x, float a, float b, float &out_sum);

void lbetafg(float a, float b, float &outs);

void ldbetabinom(int n, int k, float a0, float b0, float an, float bn, float &out_d);

// End of the header guard
#endif
