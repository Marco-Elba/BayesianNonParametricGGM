#include <random>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>


// Function to be used only inside c++, takes Eigen types
void my_rmvnorm_cpp(int p, const Eigen::VectorXf &Mean, const Eigen::MatrixXf &Sigma,
                    std::default_random_engine& random_generator,
                    Eigen::VectorXf &out_draw){
  std::normal_distribution<> norm_distr{0., 1.};
  for(int i=0; i < p; i++){
    out_draw[i] = norm_distr(random_generator);
  }
  out_draw = ( Sigma.llt().matrixL() ) * out_draw + Mean;
}

// Function that takes directly cholesky factor of Sigma
void my_rmvnorm_chol(int p, const Eigen::VectorXf &Mean, const Eigen::MatrixXf &Sigma_chol,
                     std::default_random_engine& random_generator,
                     Eigen::VectorXf &out_draw){
  std::normal_distribution<> norm_distr{0., 1.};
  for(int i=0; i < p; i++){
    out_draw[i] = norm_distr(random_generator);
  }
  out_draw = Sigma_chol * out_draw + Mean;
}

// Function that takes directly cholesky factor of Sigma with zero mean
void my_rmvnorm_chol_nomean(int p, const Eigen::MatrixXf &Sigma_chol,
                            std::default_random_engine& random_generator,
                            Eigen::VectorXf &out_draw){
  std::normal_distribution<> norm_distr{0., 1.};
  for(int i=0; i < p; i++){
    out_draw[i] = norm_distr(random_generator);
  }
  out_draw = Sigma_chol * out_draw;
}


// C++ internal function that takes in a gamma(a, scale = b) and output a gamma(a, rate=b)
double rgamma_rate( float a , float b ){
  float draw;
  draw = R::rgamma( a, 1 / b );
  return draw;
}

// C++ internal function for Inverse-Gamma draw
double rinvgamma_rate( float a , float b ){
  float draw;
  draw = 1 / R::rgamma( a, 1 / b );
  return draw;
}
