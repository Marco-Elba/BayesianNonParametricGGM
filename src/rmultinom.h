#include <RcppEigen.h>
#include <Eigen/Dense>

// Start of the header guard
#ifndef RMULTINOM_H
#define RMULTINOM_H

// Content of the header file, so the declaration of the function
//int rmultinom(gsl_rng * r, unsigned int N, const std::vector<double> &probsAll);

void which_rmultinom_Eigen(int M, const Eigen::VectorXf &probsAll, int &out_pos);

// End of the header guard
#endif
