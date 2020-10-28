#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>

// Start of the header guard
#ifndef MAT_FUNCTIONS_H
#define MAT_FUNCTIONS_H

void matrix_std(const Eigen::MatrixXf &Input_X, int n, int p, Eigen::MatrixXf &Output_X );

void matrix_beta_std(const Eigen::MatrixXf &Input_Y, const Eigen::MatrixXf &Input_X,
                     int M, int pcovs, const Eigen::VectorXf &Beta, Eigen::MatrixXf &beta_matrix,
                     Eigen::MatrixXf &Output_Y );

void make_Omega_temp(const Eigen::MatrixXf &Input_O, Eigen::MatrixXf &Output_O, int M,
                     const Eigen::VectorXi &pos );

void make_S_temp_m(const Eigen::MatrixXf &Input_S, Eigen::VectorXf &Output_S, int M,
                   const Eigen::VectorXi &pos, int col_jj );

void probs_Omega( const Eigen::MatrixXf &Omega, float sd_v0, float sd_v1,
                  float v0_square, float v1_square, int M,
                  Eigen::VectorXf &probs_z1, Eigen::VectorXf &probs_z0 );

void accRatio_eta_logit(int M, int Ngroups, const Eigen::MatrixXf &probs_pi,
                        Eigen::VectorXf &pi_vec, Eigen::VectorXf &pi_vec_prop, const Eigen::MatrixXi &Z,
                        Eigen::VectorXf &eta_prop, Eigen::VectorXf &eta,
                        const Eigen::VectorXf &eta_mean, const Eigen::LLT<Eigen::MatrixXf> &eta_cov_chol,
                        int &acc_eta );
// End of the header guard
#endif
