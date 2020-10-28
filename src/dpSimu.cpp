#include <vector>
#include <Rcpp.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <Eigen/Dense>


// Simulate a DGDP stick breaking construction with Eigen types
void sb_dgdp_Eigen(const int M, const float phi, const float mu, Eigen::VectorXf &vw, Eigen::VectorXf &pw){
  float draw_beta;
  for (int jj=0; jj<M; jj++){
    draw_beta = R::rbeta(phi*mu , phi*(1-mu) );
    vw[jj] = draw_beta;
  }
  pw[0] = vw[0];
  float probsSum = vw[0];
  for(int jj=1; jj<M; jj++){
    pw[jj] = vw[jj] * pw[jj-1] * (1-vw[jj-1]) / vw[jj-1];
    probsSum += pw[jj];
  }
  for (int jj=0; jj < M; ++jj){
    // Normalize the vector of probabilities
    pw[jj] /= probsSum;
  }
}
