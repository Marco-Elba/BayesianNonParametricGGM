#include <RcppEigen.h>
#include <Eigen/Dense>

// Start of the header guard
#ifndef DPSIMU_H
#define DPSIMU_H

// Content of the header file, so the declaration of the function
void sb_dgdp_Eigen(const int M, const float phi, const float mu, Eigen::VectorXf &vw, Eigen::VectorXf &pw);

// End of the header guard
#endif
