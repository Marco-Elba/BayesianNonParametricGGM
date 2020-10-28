// Start of the header guard
#ifndef KRON_FUNCTIONS_H
#define KRON_FUNCTIONS_H

#include <Eigen/Dense>
#include <Rcpp.h>

// Content of the header file, so the declaration of the function
void Kron_sur_X(const Eigen::VectorXf &omega, const Eigen::MatrixXf &XtX, int dim, int n_reg,
                Eigen::MatrixXf &X_out);

void Kron_sur_Y(const Eigen::VectorXf &omega, const Eigen::MatrixXf &XtY, int dim, int n_reg,
                Eigen::VectorXf &Y_out);

void Kron_sur_X_Eigen(const Eigen::MatrixXf &omega, const Eigen::MatrixXf &XtX, const Eigen::MatrixXf &XtY,
                      int Ngroups, int dim, int n_reg,
                      Eigen::MatrixXf &X_out, Eigen::VectorXf &Y_out);
// End of the header guard
#endif
