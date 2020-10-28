#include <vector>
#include <random>
#include <Rcpp.h>
#include <Rmath.h>
#include <Rconfig.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "log_density_gamma.h"
#include "my_rmvnorm.h"
#include "mat_functions.h"
#include "dpSimu.h"
#include "clusterAllocation.h"
#include "NormalDensity.h"
#include "kron_functions.h"
#include "rmultinom.h"


// [[Rcpp::export]]
void MultipleGGMs_SSSL_DGDP( int n, int M, int Ngroups, int pcovs, std::vector<int> groups_dim,
                             Rcpp::NumericMatrix &Ydata, Rcpp::NumericMatrix &Xdata,
                            float alpha_a, float alpha_b, int K,
                            float pi_a, float pi_b, float v0, float h, float lambda_Sigma,
                            float scale_global, float scale_proposal,
                            int iterations, int burn_in, int tinning,
                            Rcpp::NumericMatrix out_Omega, Rcpp::IntegerMatrix out_Z,
                            Rcpp::NumericMatrix out_pi, Rcpp::IntegerMatrix out_cind,
                            Rcpp::NumericMatrix out_beta, Rcpp::IntegerVector out_nclus,
                            Rcpp::IntegerVector out_acc_alpha, Rcpp::NumericMatrix out_alpha_mu ){
  Rcpp::Rcout << "<<<<<< Start C++ code >>>>>>> " << "\n";
  // ------------------------------------------------------------
  GetRNGstate();
  // Transform from Rcpp to Eigen type
  const Eigen::Map<Eigen::MatrixXd> Ydouble (Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Ydata) );
  const Eigen::Map<Eigen::MatrixXd> Xdouble (Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(Xdata) );
  // Then cast to float
  Eigen::MatrixXf Y = Ydouble.cast <float> ();
  Eigen::MatrixXf X = Xdouble.cast <float> ();
  // Define random number generator
  std::random_device rd;
  std::default_random_engine rand_generator(rd());
  // Create K-dimensional vector to hold the probabilities for the multinomial draws
  std::vector<float> probs_multinomial(K, 0.);
  // Group first obs
  Eigen::VectorXi groups_first = Eigen::VectorXi::Zero( Ngroups );
  groups_first[ 0 ] = 0;
  for(int gr=1; gr<Ngroups; gr++){
    groups_first[ gr ] = groups_first[ gr-1 ] + groups_dim[ gr-1 ];
  }
  // Horseshoe prior on beta ( vectorised version)
  // Auxiliary parametrisation through IG
  int pM = pcovs * M;                                   // Total number of regressors
  Eigen::VectorXf beta0 = Eigen::VectorXf::Zero( pM );  // Prior mean on Beta
  Eigen::MatrixXf Diag1_beta = Eigen::MatrixXf::Zero( pM, pM );
  for(int ii = 0; ii < pM; ii++){
    Diag1_beta(ii, ii) = 1;
  }
  // Initial values for auxiliaru ni (pM) and zeta (M)
  Eigen::VectorXf ni = Eigen::VectorXf::Zero( pM );
  Eigen::VectorXf zeta = Eigen::VectorXf::Zero( M );
  for( int ll=0; ll < M; ll++){
    zeta[ ll ] = rinvgamma_rate( 0.5, 1 / scale_global );
    for( int jj = 0; jj < pcovs; jj++){
      ni[ jj + ll*pcovs ] = rinvgamma_rate( 0.5, 1);
    }
  }
  // Initial values for lambda and tau
  Eigen::VectorXf lambda2 = Eigen::VectorXf::Zero( pM );
  Eigen::VectorXf tau2 = Eigen::VectorXf::Zero( M );
  for( int ll=0; ll < M; ll++){
    tau2[ ll ] = rinvgamma_rate( 0.5, 1 / zeta[ ll ] );
    for( int jj = 0; jj < pcovs; jj++){
      lambda2[ jj + ll*pcovs ] = rinvgamma_rate( 0.5, 1 / ni[ jj + ll*pcovs ] );
    }
  }
  // initial value beta
  Eigen::VectorXf beta = Eigen::VectorXf::Zero( pM );
  Eigen::MatrixXf beta_matrix = Eigen::MatrixXf::Zero( pcovs, M );
  for( int ll = 0; ll < M; ll++){
    for( int jj = 0; jj < pcovs; jj++){
      std::normal_distribution<> norm_distr{0., std::sqrt( lambda2[ jj + ll*pcovs ] * tau2[ll] )};
      beta[ jj + ll*pcovs ] = norm_distr(rand_generator);
      // R::rnorm( 0, std::sqrt( lambda2[ jj + ll*pcovs ] * tau2[ll] ) );
    }
  }
  // P(Omega | Z, theta) propto N(a_ij | 0, v_zij ) * exp(a_ii | lambda/2)
  // P( Z | theta ) propto pi^z_ij (1-pi)^1-z_ij
  int mm_tot = M*(M - 1 ) / 2 * Ngroups;
  int mm1 = M*(M-1) / 2;  // edges for each graph
  // GIBBS sampling for Omega and Z
  Eigen::MatrixXf Ystd = Eigen::MatrixXf::Zero(n, M);
  matrix_std(Y, n, M, Ystd); // to standardise the matrix
  // Create objects for multiple graphs - concatenate matrices by row
  Eigen::MatrixXf S = Eigen::MatrixXf::Zero(M * Ngroups, M);
  for(int gr = 0; gr < Ngroups; gr++){
    S.block(M*gr, 0, M, M) = Ystd.block( groups_first[gr], 0, groups_dim[gr], M ).transpose() *
                                Ystd.block( groups_first[gr], 0, groups_dim[gr], M );
  }
  // matrices XtX for groups
  Eigen::MatrixXf XtX = Eigen::MatrixXf::Zero(pcovs * Ngroups, pcovs);
  for(int gr = 0; gr < Ngroups; gr++){
    XtX.block(pcovs*gr, 0, pcovs, pcovs) = X.block( groups_first[gr], 0, groups_dim[gr], pcovs ).transpose() *
      X.block( groups_first[gr], 0, groups_dim[gr], pcovs );
  }
  Eigen::MatrixXf XtY = Eigen::MatrixXf::Zero(pcovs * Ngroups, M);
  for(int gr = 0; gr < Ngroups; gr++){
    XtY.block(pcovs*gr, 0, pcovs, M) = X.block( groups_first[gr], 0, groups_dim[gr], pcovs ).transpose() *
      Y.block( groups_first[gr], 0, groups_dim[gr], M );
  }
  // matrix Z in {0,1} for edge inclusion
  Eigen::MatrixXi Z = Eigen::MatrixXi::Zero(M * Ngroups, M);
  Eigen::VectorXf probs_z1 = Eigen::VectorXf::Zero( mm1 );
  Eigen::VectorXf probs_z0 = Eigen::VectorXf::Zero( mm1 );
  Eigen::VectorXf probs_zpi = Eigen::VectorXf::Zero( mm1 );
  // pi | P ~ DGDP( mu_x,alpha, P0 )
  // P0 = Beta( pi_a, pi_b ) Common Locations to all groups
  // mu_x = Uniform(0,1)
  // alpha ~ Gamma( alpha_a, alpha_b )
  Eigen::MatrixXf cov_prop_norm = Eigen::MatrixXf::Zero(Ngroups + 1, Ngroups + 1);
  Eigen::MatrixXf chol_cov_prop_norm = Eigen::MatrixXf::Zero(Ngroups + 1, Ngroups + 1);
  for(int jj=0; jj < ( Ngroups+1 ); jj++){
    cov_prop_norm(jj,jj) = 1 * scale_proposal;
  }
  chol_cov_prop_norm = cov_prop_norm.llt().matrixL(); // Lower triangular
  Eigen::VectorXf alpha_mu_norm = Eigen::VectorXf::Zero( Ngroups + 1 );
  my_rmvnorm_chol_nomean(Ngroups + 1, chol_cov_prop_norm, rand_generator, alpha_mu_norm ); // draw normal
  Eigen::VectorXf alpha_mu_prop = Eigen::VectorXf::Zero( Ngroups + 1 );
  Eigen::VectorXf alpha_mu_norm_prop = Eigen::VectorXf::Zero( Ngroups + 1 );
  Eigen::VectorXf alpha_mu = Eigen::VectorXf::Zero( Ngroups + 1 );
  // Transform
  alpha_mu[ 0 ] = std::exp( alpha_mu_norm[ 0 ] ); // For alpha
  for(int gr = 0; gr < Ngroups; gr++){
    alpha_mu[ gr + 1] = 1 / (1 + std::exp( - alpha_mu_norm[ gr+1 ] ) ); // for mu_x
  }
  // stick-breaking objects
  Eigen::MatrixXf pw = Eigen::MatrixXf::Zero(K, Ngroups);
  Eigen::MatrixXf vw = Eigen::MatrixXf::Zero(K, Ngroups);
  Eigen::VectorXf vw_col = Eigen::VectorXf::Zero(K);
  Eigen::VectorXf pw_col = Eigen::VectorXf::Zero(K);
  // Weights generations: Stick Breaking rule
  for(int gr=0; gr<Ngroups; gr++){
    sb_dgdp_Eigen(K, alpha_mu[0], alpha_mu[gr+1], vw_col, pw_col);
    pw.col( gr ) = pw_col;
    vw.col( gr ) = vw_col;
  }
  // Cluster index Vector
  Eigen::VectorXi cind = Eigen::VectorXi::Zero( mm_tot ); // for all groups
  // Clusters size
  Eigen::VectorXi cluster_size = Eigen::VectorXi::Zero(K);
  Eigen::MatrixXi cluster_group_size = Eigen::MatrixXi::Zero(K, Ngroups);
  Eigen::VectorXi cluster_size_one = Eigen::VectorXi::Zero(K);
  Eigen::VectorXi cluster_size_zero = Eigen::VectorXi::Zero(K);
  dgdp_prior_allocation( K, mm1, Ngroups, pw, cind, cluster_size );
  // DP Base Measure P0
  // pi_a / (pi_a + pi_b) mean of Beta distr
  Eigen::VectorXf pi_clus = Eigen::VectorXf::Zero(K);
  for(int jj=0; jj < K; jj++){
    // pi_clus[ jj ] = R::rbeta(pi_a, pi_b);
    pi_clus[ jj ] = 0.5;
  }
  Eigen::VectorXf new_probs = Eigen::VectorXf::Zero(K);
  // Matrix V = (v_zij)
  float v1 = v0*h;
  float v1_square = v1*v1;
  float v0_square = v0*v0;
  Eigen::MatrixXf V = Eigen::MatrixXf::Constant( M * Ngroups, M, v0_square);
  // Initial value for Omega
  Eigen::MatrixXf Omega = Eigen::MatrixXf::Zero( M * Ngroups, M );
  Eigen::MatrixXf Omega_cor = Eigen::MatrixXf::Zero( M * Ngroups, M );
  for(int gr = 0; gr < Ngroups; gr++){
    for(int jj = 0; jj < M; jj++){
      Omega(jj + gr*M, jj) = 1; // make omega unit diagonal
    }
  }
  Eigen::MatrixXf Omega_temp = Eigen::MatrixXf::Zero( M-1, M-1 );
  Eigen::MatrixXf S_temp = Eigen::MatrixXf::Zero( M-1, M-1 );
  Eigen::VectorXf S_temp_m = Eigen::VectorXf::Zero( M-1 );
  Eigen::MatrixXf Inv_Omega_temp = Eigen::MatrixXf::Zero( M-1, M-1 );
  Eigen::MatrixXf Diag1 = Eigen::MatrixXf::Zero( M-1, M-1 );
  for(int ii=0; ii < (M-1); ii++){
    Diag1(ii, ii) = 1;
  }
  Eigen::MatrixXf Omega_temp_Cov = Eigen::MatrixXf::Zero( M-1, M-1 );
  Eigen::MatrixXf Omega_temp_ICov = Eigen::MatrixXf::Zero( M-1, M-1 );
  Eigen::VectorXf Omega_temp_Mean = Eigen::VectorXf::Zero( M-1 );
  Eigen::VectorXf gamma_coeff = Eigen::VectorXf::Zero( M-1 );
  Eigen::VectorXf gamma_draw = Eigen::VectorXf::Zero( M-1 );
  float omega_diag;
  Eigen::MatrixXi pos_ind = Eigen::MatrixXi::Zero( M-1, M );
  for(int jj = 0; jj < M; jj++){
    int i_ind;
    i_ind = 0;
    if( jj == 0 ){
      for(int ii = 0; ii < (M-1); ii++){
        pos_ind(ii, jj) = ii+1;
      }
    }// j == 0
    if( jj == (M-1) ){
      for(int ii = 0; ii < (M-1); ii++){
        pos_ind(ii, jj) = ii;
      }
    }// j == M
    if( jj != 0 && jj != (M-1) ){
      for(int ii = 0; ii < M; ii++){
        if (ii == jj) continue;
        pos_ind(i_ind, jj) = ii;
        i_ind ++;
      }
    }// j in between
  }// end JJ
  Eigen::VectorXf devs_X = Eigen::VectorXf::Zero( pcovs );
  for(int jj=0; jj < pcovs; jj++){
    devs_X[ jj ] = X.col(jj).dot( X.col(jj) );
  }
  Eigen::MatrixXf Sn = Eigen::MatrixXf::Zero( pM, pM );
  Eigen::MatrixXf beta_ISn = Eigen::MatrixXf::Zero( pM, pM );
  // Eigen::LLT<Eigen::MatrixXf> chol_beta_Sn;
  Eigen::VectorXf XOY = Eigen::VectorXf::Zero( pM );
  Eigen::VectorXf beta_n = Eigen::VectorXf::Zero( pM );
  Eigen::VectorXf beta_draw = Eigen::VectorXf::Zero( pM );
  int acc_alpha;
  acc_alpha = 0;
  int tinning_counter = 0;
  
  Rcpp::Rcout << "<<<<<< Start MCMC >>>>>>> " << "\n";
  // -------------------------------------------------------------------------
  // ------------------------- START MCMC ------------------------------------
  // -------------------------------------------------------------------------
  for(int iter = 0; iter < iterations; iter++){
    if( ( iter % 100 ) == 0 ){
      Rcpp::Rcout << "<<<<<< Iteration >>>>>>> " << iter + 1 << "\n";
    }
    // Rcpp::Rcout << "<<<<<<<<<<<<<<<<<<<<< Iteration >>>>>>>>>>>>>>>>>>>>>>>>> " << iter + 1 << "\n";
    // Update Beta - mean
    // Update auxiliary parameters ni and zeta
    for( int ll=0; ll < M; ll++){
      zeta[ ll ] = rinvgamma_rate( 1, 1 / scale_global + 1 / tau2[ ll ] );
      for( int jj = 0; jj < pcovs; jj++){
        ni[ jj + ll*pcovs ] = rinvgamma_rate( 1, 1 + 1 / lambda2[ jj + ll*pcovs ] );
      }
    }
    // Update parameters lambda2 and tau2
    float beta2;
    float sum_beta2;
    for(int ll = 0; ll < M; ll++){
      sum_beta2 = 0;
      for(int jj=0; jj < pcovs; jj++){
        beta2 = beta[ jj + ll*pcovs ] * beta[ jj + ll*pcovs ];
        lambda2[ jj + ll*pcovs ] = rinvgamma_rate( 1, 1 / ni[ jj + ll*pcovs ] + beta2 / ( 2*tau2[ ll ] ) );
        sum_beta2 += beta2 / lambda2[ jj + ll*pcovs ];
      }
      tau2[ ll ] = rinvgamma_rate( (pcovs+1)/2, 1 / zeta[ ll ] + 0.5*sum_beta2 );
    }
    // Update Beta
    Kron_sur_X_Eigen( Omega, XtX, XtY, Ngroups, M, pcovs, Sn, XOY);
    // Add to the diagonal
    for(int ll = 0; ll < M; ll++){
      for(int jj=0; jj < pcovs; jj++){
        Sn( jj + ll*pcovs, jj + ll*pcovs ) += 1 / ( lambda2[ jj + ll*pcovs ] * tau2[ ll ] );
      }
    }
    // Cholesky factor of Sn
    // chol_beta_Sn = Sn.llt();
    beta_ISn = Sn.llt().solve( Diag1_beta );
    beta_n = beta_ISn * XOY;
    for( int jj=0; jj < pM; jj++){
      std::normal_distribution<> norm_distr{0., 1.};
      beta_draw[ jj ] = norm_distr(rand_generator);
      // R::rnorm(0, 1);
    }
    beta = beta_ISn.llt().matrixL() * beta_draw + beta_n;
    // Update Omega - column wise
    matrix_beta_std( Y, X, M, pcovs, beta, beta_matrix, Ystd );
    std::normal_distribution<> norm_distr{0., 1.};
    for(int gr=0; gr < Ngroups; gr++){
      int group_start = gr*M;
      S.block( group_start, 0, M, M) = Ystd.block( groups_first[gr], 0, groups_dim[gr], M ).transpose() *
        Ystd.block( groups_first[gr], 0, groups_dim[gr], M );
      for(int jj = 0; jj < M; jj++){
        make_Omega_temp( Omega.block( group_start, 0, M, M), Omega_temp, M, pos_ind.col(jj) );
        make_Omega_temp( S.block( group_start, 0, M, M), S_temp, M, pos_ind.col(jj) );
        Inv_Omega_temp = Omega_temp.llt().solve( Diag1 );
        Omega_temp_Cov = ( S( group_start + jj, jj) + lambda_Sigma ) * Inv_Omega_temp;
        for(int kk=0; kk < (M-1); kk++){
          Omega_temp_Cov(kk, kk) += 1 / V( group_start + pos_ind(kk, jj), jj);
          gamma_draw[ kk ] = norm_distr(rand_generator); 
          // R::rnorm(0,1);
        }

        Omega_temp_ICov = Omega_temp_Cov.llt().solve( Diag1 );
        make_S_temp_m( S.block( group_start, 0, M, M), S_temp_m, M, pos_ind.col(jj), jj );
        Omega_temp_Mean = - Omega_temp_ICov * S_temp_m;
        gamma_coeff = Omega_temp_ICov.llt().matrixL() * gamma_draw + Omega_temp_Mean;

        // Diagonal Precisions
        omega_diag = rgamma_rate( groups_dim[ gr ]/2 + 1, ( S( group_start + jj, jj ) + lambda_Sigma) / 2 );
        Omega( group_start + jj, jj) = omega_diag + gamma_coeff.transpose() * Inv_Omega_temp * gamma_coeff;
        for(int kk = 0; kk < (M-1); kk++){
          Omega( group_start + pos_ind(kk, jj), jj ) = gamma_coeff[ kk ];
          Omega( group_start + jj, pos_ind(kk, jj) ) = gamma_coeff[ kk ];
        }
      }// end jj
      // Update Z - Upper positions dimension (M*M-1)/2 = mm1
      // To produce a Precision matrix
      // for(int jj = 0; jj < M; jj++){
      //   for(int kk = 0; kk < M; kk++){
      //     Omega_cor(group_start + jj, kk) = Omega( group_start + jj, kk ) /
      //               std::sqrt( Omega( group_start + jj, jj ) * Omega( group_start + kk, kk ) );
      //   }
      // }
      // Calculate the probabilities
      probs_Omega( Omega.block( group_start, 0, M, M), v0, v1,
                   v0_square, v1_square, M, probs_z1, probs_z0 );
      int mm_start = mm1 * gr;
      for(int jj = 0; jj < mm1; jj++){
        probs_zpi[ jj ] = probs_z1[jj] * pi_clus[ cind[ mm_start + jj ] ] / (
          probs_z1[jj] * pi_clus[ cind[ mm_start + jj ] ] + probs_z0[jj] * (1-pi_clus[ cind[ mm_start + jj ] ]) );
      }
      int ind_pos = 0;
      int bernoulli_draw;
      for(int jj = 0; jj < M; jj++){
        if((jj+1) < M){
          for(int kk = jj+1; kk < M; kk++){
            std::bernoulli_distribution bern_distr(probs_zpi[ind_pos]);
            bernoulli_draw = bern_distr(rand_generator);
            // bernoulli_draw = R::rbinom(1, probs_zpi[ind_pos] );
            Z( group_start + jj, kk ) = bernoulli_draw;
            Z( group_start + kk, jj ) = bernoulli_draw;
            // For V
            V( group_start + jj, kk ) = bernoulli_draw * v1_square + (1-bernoulli_draw) * v0_square;
            V( group_start + kk, jj ) = bernoulli_draw * v1_square + (1-bernoulli_draw) * v0_square;
            ind_pos++;
          }
        }
      }
    }// end gr
    // DGDP Cluster allocation
    dgdp_posterior_allocation(K, M, Ngroups, pi_clus, Z, pw,
                              rand_generator, probs_multinomial,
                              cind, cluster_size, cluster_group_size,
                              cluster_size_one, cluster_size_zero);
    for(int cc = 0; cc < K; cc++){
      pi_clus[ cc ] = R::rbeta( pi_a + cluster_size_one[ cc ], pi_b + cluster_size_zero[ cc ] );
    }
    // Update weights stick-breaking
    dgdp_weights_update(K, Ngroups, alpha_mu, cluster_group_size, vw, pw);
    // Update alpha and mu
    my_rmvnorm_chol( Ngroups + 1, alpha_mu_norm, chol_cov_prop_norm,
                     rand_generator, alpha_mu_norm_prop); // Draw
    alpha_mu_prop[ 0 ] = std::exp( alpha_mu_norm_prop[ 0 ] ); // For alpha
    for(int gr = 0; gr < Ngroups; gr++){
      alpha_mu_prop[ gr + 1] = 1 / (1 + std::exp( - alpha_mu_norm_prop[ gr+1 ] ) ); // for mu_x
    }
    accRatio_DGDP( K, Ngroups, vw, alpha_mu_prop, alpha_mu, alpha_mu_norm_prop, alpha_mu_norm,
                  alpha_a, alpha_b, acc_alpha);
    // ------------------------------------------------------------
    // ----------------- SAVE results -----------------------------
    // ------------------------------------------------------------
    if( iter >= burn_in){
      // save the MCMC sample when the tinning counter
      if((iter % tinning) == 0){
        tinning_counter += 1;
        // save Omega and Z
        int fill_position = 0;
        for( int gr = 0; gr < Ngroups; gr++){
          for(int jj = 0; jj < M; jj++){
            for(int ll = 0; ll <= jj; ll++){
              // save only the upper diagonal (main diagonal included) by column
              out_Omega(tinning_counter, fill_position) = Omega(gr*M + ll, jj);
              out_Z(tinning_counter, fill_position) = Z(gr*M + ll, jj);
              fill_position += 1;
            }
          }
          int newC = 0;
          which_rmultinom_Eigen( K, pw.col(gr), newC);
          out_pi(tinning_counter, gr) = pi_clus[ newC ];
        }// end gr
        for(int ii = 0; ii < mm_tot; ii++){
          out_cind(tinning_counter, ii) = cind[ ii ];
        }
        int full_clus = 0;
        for(int kk = 0; kk < K; kk++){
          if( cluster_size[kk] != 0) full_clus++;
        }
        out_nclus[tinning_counter] = full_clus;
        // Beta
        for(int beta_count = 0; beta_count < M*pcovs; beta_count++){
          out_beta(tinning_counter, beta_count ) = beta[ beta_count ];
        }
        for(int gr=0; gr < (Ngroups+1); gr++){
          out_alpha_mu(tinning_counter, gr) = alpha_mu[ gr ];
        }
      }// tinning loop
    }// end IF save
  }// End MCMC
  out_acc_alpha[ 0 ] = acc_alpha;
  PutRNGstate();
  Rcpp::Rcout << " --------------- END C++ code ------------- " << "\n";
}
