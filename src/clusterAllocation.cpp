#include <vector>
#include <random>
#include <Rcpp.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "rmultinom.h"
#include "log_density_gamma.h"
#include "NormalDensity.h"


// Prio cluster allocation for prior DGDP
void dgdp_prior_allocation(int K, int mm1, int Ngroups,
                           const Eigen::MatrixXf &pw, Eigen::VectorXi &cind,
                           Eigen::VectorXi &cluster_size){
  // Create a dynamic vector of integers of length K
  int *draw = new int[ K ];
  double *probs = new double[ K ];  // Create extra vector for probabilities
  int edge_index;
  edge_index = 0;
  for(int gr=0; gr < Ngroups; gr++){
    double sum_probs = 0.;
    // Put probs pw_gr in array container
    for( int jj = 0; jj < K; jj++){
      probs[ jj ] = pw(jj, gr);
      sum_probs += probs[ jj ];
    }
    for( int jj = 0; jj < K; jj++){
      probs[ jj ] = probs[ jj ] / sum_probs;
    }
    // draw from the Multinomial for each observation ii
    for(int ii=0; ii < mm1; ii++){
      rmultinom(1, probs, K, draw );
      // check which one pos == 1
      int elem = 0;
      int pos = 0;
      while( pos==0 && elem < K ){
        if( draw[ elem ] == 1 ){
          pos=elem;
        }
        ++elem;
      } // end while
      cind[ edge_index ] = pos;
      // Count the cluster size
      cluster_size[ pos ] += 1;
      edge_index++;
    }// end loop ii
  }// end gr
  // Delete the dinamically allocated array !!
  delete[] probs;
  delete[] draw;
}


// Cluster allocation for POSTERIOR DGDP
void dgdp_posterior_allocation(int K, int M, int Ngroups, const Eigen::VectorXf &pi_clus,
                               const Eigen::MatrixXi &Z, const Eigen::MatrixXf &pw,
                               std::default_random_engine& random_generator,
                               std::vector<float>& probs,
                               Eigen::VectorXi &cind, Eigen::VectorXi &cluster_size,
                               Eigen::MatrixXi &cluster_group_size, Eigen::VectorXi &cluster_size_one,
                               Eigen::VectorXi &cluster_size_zero){
  // Create a dynamic vector of integers of length K
  // int *draw = new int[ K ];
  // double *probs = new double[ K ];  // Create extra vector for probabilities
  int edge_index;
  edge_index = 0;
  for(int cc = 0; cc < K; cc++){
    cluster_size[ cc ] = 0;
    cluster_size_one[ cc ] = 0;
    cluster_size_zero[ cc ] = 0;
    for(int gr = 0; gr < Ngroups; gr++){
      cluster_group_size( cc, gr ) = 0;
    }
  }
  for(int gr=0; gr < Ngroups; gr++){
    int group_first = gr*M;
    // draw from the Multinomial for each observation ii
    for(int jj = 0; jj < M; jj++){
      if( (jj+1) < M ){
        for(int kk = jj+1; kk < M; kk++){
          // Put probs pw_gr in array container
          for(int cc = 0; cc < K; cc++){
            probs[ cc ] = pw(cc, gr) * ( pi_clus[ cc ] * Z(group_first + jj, kk) +
              ( 1 - pi_clus[ cc ] )*( 1 - Z(group_first + jj, kk) ) );
          }
          // Make the probs vec to sum to 1
          // for(int cc = 0; cc < K; cc++){
          //   probs[ cc ] /= probs_sum;
          // }
          std::discrete_distribution<> multinomial(probs.begin(), probs.end());
          int pos = multinomial(random_generator);
          // rmultinom(1, probs, K, draw );
          // check which one pos == 1
          // int elem = 0;
          // int pos = 0;
          // while( pos==0 && elem < K ){
          //   if( draw[ elem ] == 1 ){
          //     pos = elem;
          //   }
          //   ++elem;
          // } // end while
          cind[ edge_index ] = pos;
          // Count the cluster size
          cluster_size[ pos ] += 1;
          cluster_group_size( pos, gr ) += 1;
          if( Z(group_first + jj, kk) == 1 ){
            cluster_size_one[ pos ] ++;
          }
          if( Z(group_first + jj, kk) == 0 ){
            cluster_size_zero[ pos ] ++;
          }
          edge_index++;
        }
      }
    }// end loop on Zi
  }// end gr
  // Delete the dinamically allocated array
  // delete[] probs;
  // delete[] draw;
}


// Function to update the weights of the stick-breaking
void dgdp_weights_update(int K, int Ngroups, const Eigen::VectorXf &alpha_mu,
                         const Eigen::MatrixXi &cluster_size,
                         Eigen::MatrixXf &vw, Eigen::MatrixXf &pw){
  float draw_beta;
  for(int gr = 0; gr < Ngroups; gr++){
    for (int jj = 0; jj < (K-1); jj++){
      int n_jz = 0;
      for(int kk = (jj+1); kk < K; kk++){
        n_jz += cluster_size(kk, gr);
      }
      draw_beta = R::rbeta( alpha_mu[0] * alpha_mu[gr+1] + cluster_size(jj, gr),
                            alpha_mu[0] * ( 1-alpha_mu[gr+1] ) + n_jz);
      vw(jj, gr) = draw_beta;
    }// END jj
    vw(K-1, gr) = 1;
    pw(0, gr) = vw(0, gr);
    for(int jj = 1; jj < K; jj++){
      pw(jj, gr) = vw(jj, gr) * pw(jj-1, gr) * ( 1 - vw(jj-1, gr) ) / vw(jj-1, gr);
    }
  }
}


// Log acceptance ratio for DGDP given ALPHA_MU Multivariate Normal
void accRatio_DGDP(int M, int Ngroups, const Eigen::MatrixXf &vw,
                   Eigen::VectorXf &alpha_mu_prop, Eigen::VectorXf &alpha_mu,
                   Eigen::VectorXf &alpha_mu_norm_prop, Eigen::VectorXf &alpha_mu_norm,
                   float prior_phi_a, float prior_phi_b, int &acc_alpha ){
  float out_sumbeta_prop = 0;
  float out_sumbeta = 0;
  float out_beta_prop = 0;
  float out_beta = 0;
  float out_gamma_prop = 0;
  float out_gamma = 0;
  for(int gr = 0; gr < Ngroups; gr++){
    for(int jj = 0; jj < (M-1); jj++){
      ldbeta( vw(jj, gr), alpha_mu_prop[0] * alpha_mu_prop[gr+1],
                alpha_mu_prop[0] * ( 1-alpha_mu_prop[gr+1] ), out_beta_prop);
      ldbeta( vw(jj, gr), alpha_mu[0] * alpha_mu[gr+1],
              alpha_mu[0] * ( 1-alpha_mu[gr+1] ), out_beta);
      // Sum
      out_sumbeta_prop += out_beta_prop;
      out_sumbeta += out_beta;
    }// jj
  }// gr
  exp_gamma_log(alpha_mu_prop[ 0 ], prior_phi_a, prior_phi_b, out_gamma_prop );
  exp_gamma_log(alpha_mu[ 0 ], prior_phi_a, prior_phi_b, out_gamma );
  // Total :
  float out_ratio;
  out_ratio = out_sumbeta_prop - out_sumbeta + out_gamma_prop - out_gamma;
  // Check with Uniform draw
  float uMH = R::runif(0,1);
  if( std::log(uMH) < out_ratio ){
    acc_alpha++;
    alpha_mu = alpha_mu_prop;
    alpha_mu_norm = alpha_mu_norm_prop;
  }
}


// Function to update the weights of the stick-breaking
void WeightsUpdate_Eigen(int M, float phi, const Eigen::VectorXi &njz,
                         Eigen::VectorXf &vw, Eigen::VectorXf &pw){
  float draw_beta;
  for (int jj=0; jj<(M-1); jj++){
    int n_jz = 0;
    for(int kk=(jj+1); kk<M; kk++){
      n_jz += njz[kk];
    }
    draw_beta = R::rbeta(1 + njz[jj], phi + n_jz);
    vw[jj] = draw_beta;
  }// END jj
  vw[M-1] = 1;
  pw[0] = vw[0];
  for(int jj=1; jj<M; jj++){
    pw[jj] = vw[jj]*pw[jj-1]*(1-vw[jj-1])/vw[jj-1];
  }
}


// Function to compute the log acceptance ratio for the DP-SB given PHI with Multivariate Normal prior
void accRatioDP_Eigen(int M, int Ngroups, const Eigen::MatrixXf &tw, const Eigen::VectorXf &sb_params_norm,
                      const Eigen::VectorXf &sb_params_norm_prop,
                      const Eigen::VectorXf &sb_params_real, const Eigen::VectorXf &sb_params_real_prop,
                      const Eigen::VectorXf &prior_normal_mean, const Eigen::MatrixXf &prior_normal_cov,
                      float &out_ratio){
  float out_sumbeta_prop = 0;
  float out_sumbeta = 0;
  float out_beta_prop = 0;
  float out_beta = 0;
  float out_norm_prop = 0;
  float out_norm = 0;
  for(int gr = 0; gr < Ngroups; gr++){
    for(int jj = 0; jj < M; jj++){
      ldbeta(tw(jj, gr), 1, sb_params_real_prop[gr], out_beta_prop);
      ldbeta(tw(jj, gr), 1, sb_params_real[gr], out_beta);
      // Sum
      out_sumbeta_prop += out_beta_prop;
      out_sumbeta += out_beta;
    }// jj
  }// gr
  lmvdnorm(sb_params_norm_prop, prior_normal_mean, prior_normal_cov, out_norm_prop);
  lmvdnorm(sb_params_norm, prior_normal_mean, prior_normal_cov, out_norm);
  // Total :
  out_ratio = out_sumbeta_prop - out_sumbeta + out_norm_prop - out_norm;
}
