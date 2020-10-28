#include <vector>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>

void Kron_sur_X(const Eigen::VectorXf &omega, const Eigen::MatrixXf &XtX, int dim, int n_reg,
              Eigen::MatrixXf &X_out){
  int ptot = dim * n_reg;
  int sj = 0;
  int sk = 0;
  for(int k = 0; k < dim; k++){
    for(int j = k; j < dim; j++){
      // Fill the lower tri + diagonal
      X_out.block(sj, sk, n_reg, n_reg) = XtX * omega[(k*dim + j)];
      // Fill upper tri
      if(sj > sk){
        X_out.block(sk, sj, n_reg, n_reg) = X_out.block(sj, sk, n_reg, n_reg).transpose() ;
      }
      sj += n_reg;
    }
    sk += n_reg;
    sj = sk;
  }// k
}

void Kron_sur_X_Eigen(const Eigen::MatrixXf &omega, const Eigen::MatrixXf &XtX, const Eigen::MatrixXf &XtY,
                      int Ngroups, int dim, int n_reg,
                      Eigen::MatrixXf &X_out, Eigen::VectorXf &Y_out){
  // Reset to zero
  for( int ii=0; ii < (dim*n_reg); ii++){
    Y_out[ ii ] = 0;
    for( int jj=0; jj < (dim*n_reg); jj++){
      X_out( ii, jj) = 0;
    }
  }
  // XtX is Ngroups*pcovs x pcovs
  // XtY is Ngroups*pcovs x M
  for(int gr=0; gr < Ngroups; gr++){
    int sj = 0;
    int sk = 0;
    for(int k = 0; k < dim; k++){
      for(int j = k; j < dim; j++){
        // Fill the lower tri + diagonal
        X_out.block(sj, sk, n_reg, n_reg) += XtX.block(gr*n_reg, 0, n_reg, n_reg) * omega(gr*dim + k, j);
        // Fill upper tri
        if(sj > sk){
          X_out.block(sk, sj, n_reg, n_reg) = X_out.block(sj, sk, n_reg, n_reg).transpose() ;
        }
        sj += n_reg;
      }
      Y_out.segment(sk, n_reg) += XtY.block(gr*n_reg, 0, n_reg, dim) * omega.row( gr*dim + k ).transpose();
      sk += n_reg;
      sj = sk;
    }// k
  }// gr
}

void Kron_sur_Y(const Eigen::VectorXf &omega, const Eigen::MatrixXf &XtY, int dim, int n_reg,
              Eigen::VectorXf &Y_out){
  // XtY is of dimension n_reg x dim
  // int ptot = dim * n_reg;
  int sj = 0;
  for(int k = 0; k < dim; k++){
      Y_out.segment(sj, n_reg) = XtY * omega.segment(dim*k, dim);
      sj += n_reg;
  }// k
}
