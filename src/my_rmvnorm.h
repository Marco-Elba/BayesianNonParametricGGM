#include <random>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// Start of the header guard
#ifndef MY_RMVNORM_H
#define MY_RMVNORM_H

// Content of the header file, so the declaration of the function

void my_rmvnorm_cpp(int p, const Eigen::VectorXf &Mean, const Eigen::MatrixXf &Sigma,
                    std::default_random_engine& random_generator,
                    Eigen::VectorXf &out_draw);

void my_rmvnorm_chol(int p, const Eigen::VectorXf &Mean, const Eigen::MatrixXf &Sigma_chol,
                     std::default_random_engine& random_generator,
                     Eigen::VectorXf &out_draw);

void my_rmvnorm_chol_nomean(int p, const Eigen::MatrixXf &Sigma_chol,
                            std::default_random_engine& random_generator,
                            Eigen::VectorXf &out_draw);


double rgamma_rate( float a , float b );
double rinvgamma_rate( float a , float b );

// End of the header guard
#endif
