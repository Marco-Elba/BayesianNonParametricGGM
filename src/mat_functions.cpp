#include <cmath>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include "NormalDensity.h"
#include "log_density_gamma.h"

void matrix_std(const Eigen::MatrixXf &Input_X, int n, int p, Eigen::MatrixXf &Output_X ){
  float mean_col;
  float sum_col;
  int j;
  int i;
  for( j=0; j < p; j++ ){
    sum_col = 0;
    for( i=0; i<n; i++ ){
      sum_col += Input_X(i, j);
    }
    mean_col = sum_col / n;
    for( i=0; i<n; i++ ){
      Output_X(i, j) = Input_X(i, j) - mean_col;
    }
  }// end j
}


void matrix_beta_std(const Eigen::MatrixXf &Input_Y, const Eigen::MatrixXf &Input_X,
                     int M, int pcovs, const Eigen::VectorXf &Beta, Eigen::MatrixXf &beta_matrix,
                     Eigen::MatrixXf &Output_Y ){
  for( int l=0; l < M; l++ ){
    beta_matrix.col( l ) = Beta.segment( l*pcovs, pcovs);
  }
  Output_Y = Input_Y - Input_X * beta_matrix;
}


void make_Omega_temp(const Eigen::MatrixXf &Input_O, Eigen::MatrixXf &Output_O, int M,
                     const Eigen::VectorXi &pos ){
  for(int jj = 0; jj < (M-1); jj++){
    for(int ii = 0; ii < (M-1); ii++){
      Output_O(jj, ii) = Input_O( pos[ jj ], pos[ ii ] );
    }
  }
}

void make_S_temp_m(const Eigen::MatrixXf &Input_S, Eigen::VectorXf &Output_S, int M,
                     const Eigen::VectorXi &pos, int col_jj ){
  for(int jj = 0; jj < (M-1); jj++){
      Output_S[ jj ] = Input_S( pos[ jj ], col_jj );
  }
}

void probs_Omega( const Eigen::MatrixXf &Omega, float sd_v0, float sd_v1,
                  float v0_square, float v1_square, int M,
                  Eigen::VectorXf &probs_z1, Eigen::VectorXf &probs_z0 ){
  int pos_ind = 0;
  for(int jj = 0; jj < M; jj++){
    if( (jj+1) < M){
      for(int kk = jj+1; kk < M; kk++){
        float x2 = Omega( jj, kk ) * Omega( jj, kk );
        float p2 = 2*M_PI;
        probs_z0[ pos_ind ] = 1 / ( sd_v0 * std::sqrt(p2) ) * std::exp( - x2/(2*v0_square) );
        probs_z1[ pos_ind ] = 1 / ( sd_v1 * std::sqrt(p2) ) * std::exp( - x2/(2*v1_square) );
        pos_ind++;
      }
    }
  }
}

// Log acceptance ratio for ETA in Logistic regression
void accRatio_eta_logit(int M, int Ngroups, const Eigen::MatrixXf &probs_pi,
                        Eigen::VectorXf &pi_vec, Eigen::VectorXf &pi_vec_prop, const Eigen::MatrixXi &Z,
                        Eigen::VectorXf &eta_prop, Eigen::VectorXf &eta,
                        const Eigen::VectorXf &eta_mean, const Eigen::LLT<Eigen::MatrixXf> &eta_cov_chol,
                        int &acc_eta ){
  float out_sumbeta_prop = 0;
  float out_sumbeta = 0;
  float out_beta_prop = 0;
  float out_beta = 0;
  float out_eta_prop = 0;
  float out_eta = 0;
  float single_prob;
  int pos_ind = 0;
  for(int jj = 0; jj < M; jj++){
    if( (jj+1) < M ){
      for(int kk = 0; kk < M; kk++){
        // Proposal
        inverse_logit( eta_prop[ 0 ] + probs_pi.row( pos_ind ) * eta_prop.segment(1, Ngroups), single_prob );
        pi_vec_prop[ pos_ind ] = single_prob;
        for(int gr = 0; gr < Ngroups; gr++){
            out_sumbeta_prop += Z( gr*M + jj, kk) * std::log( single_prob ) +
                            ( 1-Z( gr*M + jj, kk) ) * std::log( 1 - single_prob );
            // Current
            out_sumbeta += Z( gr*M + jj, kk) * std::log( pi_vec( pos_ind ) ) +
              ( 1-Z( gr*M + jj, kk) ) * std::log( 1- pi_vec( pos_ind ) );
        }// gr
        pos_ind++;
      }
    }
  }// jj
  // Prior difference
  lmvdnorm_chol( eta_prop, eta_mean, eta_cov_chol , eta_cov_chol.matrixL(), out_eta_prop);
  lmvdnorm_chol( eta, eta_mean, eta_cov_chol , eta_cov_chol.matrixL(), out_eta);
  // Total :
  float out_ratio;
  out_ratio = out_sumbeta_prop - out_sumbeta + out_eta_prop - out_eta;
  // Check with Uniform draw
  float uMH = R::runif(0,1);
  if( std::log(uMH) < out_ratio ){
    acc_eta++;
    eta = eta_prop;
    pi_vec = pi_vec_prop;
  }
}
