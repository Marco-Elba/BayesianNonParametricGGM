#include <vector>
#include <random>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>

// Start of the header guard
#ifndef CLUSTERALLOCATION_H
#define CLUSTERALLOCATION_H

// Content of the header file, so the declaration of the function

void dgdp_prior_allocation(int K, int mm1, int Ngroups,
                           const Eigen::MatrixXf &pw, Eigen::VectorXi &cind,
                           Eigen::VectorXi &cluster_size );

void dgdp_posterior_allocation(int K, int M, int Ngroups, const Eigen::VectorXf &pi_clus, const Eigen::MatrixXi &Z,
                               const Eigen::MatrixXf &pw, std::default_random_engine& random_generator,
                               std::vector<float>& probs,
                               Eigen::VectorXi &cind, Eigen::VectorXi &cluster_size,
                               Eigen::MatrixXi &cluster_group_size,
                               Eigen::VectorXi &cluster_size_one, Eigen::VectorXi &cluster_size_zero);

void dgdp_weights_update(int K, int Ngroups, const Eigen::VectorXf &alpha_mu, const Eigen::MatrixXi &cluster_size,
                         Eigen::MatrixXf &vw, Eigen::MatrixXf &pw);

void accRatio_DGDP(int M, int Ngroups, const Eigen::MatrixXf &pw,
                   Eigen::VectorXf &alpha_mu_prop, Eigen::VectorXf &alpha_mu,
                   Eigen::VectorXf &alpha_mu_norm_prop, Eigen::VectorXf &alpha_mu_norm,
                   float prior_phi_a, float prior_phi_b,
                   int &acc_alpha);

void WeightsUpdate_Eigen(int M, float phi, const Eigen::VectorXi &njz,
                         Eigen::VectorXf &vw, Eigen::VectorXf &pw);

void accRatioDP_Eigen(int M, int Ngroups, const Eigen::MatrixXf &tw, const Eigen::VectorXf &sb_params_norm,
                      const Eigen::VectorXf &sb_params_norm_prop,
                      const Eigen::VectorXf &sb_params_real, const Eigen::VectorXf &sb_params_real_prop,
                      const Eigen::VectorXf &prior_normal_mean, const Eigen::MatrixXf &prior_normal_cov,
                      float &out_ratio);

// End of the header guard
#endif
