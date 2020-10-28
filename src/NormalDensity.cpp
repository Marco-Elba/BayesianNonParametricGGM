#include <cmath>
#include <vector>
#include "util.h"
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// Function to calculate the density of the Normal distribution
void my_dnorm(float x, float mu, float tau2, float &out_dnorm){
	float x2 = (x-mu)*(x-mu);
	float p2 = 1/(2*M_PI);
	out_dnorm = pow(p2*tau2, 0.5) * exp(-0.5*tau2*x2);
}

// Function to calculate the Log-density of the Normal distribution
void my_ldnorm(float x, float mu, float tau2, float &out_dnorm){
  float x2 = (x-mu)*(x-mu);
  out_dnorm = 0.5 * (log(tau2) -std::log(2*M_PI) -tau2*x2);
}

// Log Density of Multivariate Normal distribution, takes Eigen in input
void lmvdnorm(const Eigen::VectorXf &x, const Eigen::VectorXf &Mean, const Eigen::MatrixXf &Cov,
              float &out_density){
  int size_x = x.size();
  Eigen::VectorXf x_centre = x - Mean;
  Eigen::LLT<Eigen::MatrixXf> Cov_chol = Cov.llt(); // to hold the cholesky factor
  Eigen::VectorXf res_solve = Cov_chol.solve(x_centre);
  // The normalization constant with log(determinant)
  Eigen::MatrixXf U = Cov_chol.matrixL(); // Lower triangular with the cholesky factor
  float kernelNorm = 0.0;
  float logDiag = 0.0;
  for(int ii=0; ii<size_x; ii++){
    kernelNorm += x_centre[ii] *  res_solve[ii];
    logDiag += std::log(U(ii, ii) );
  }
  out_density = -0.5 * (size_x*std::log(2*M_PI) + 2*logDiag + kernelNorm);
}

// Log Density of Multivariate Normal distribution, takes Eigen in input and Cholesky directly
void lmvdnorm_chol(const Eigen::VectorXf &x, const Eigen::VectorXf &Mean,
                   const Eigen::LLT<Eigen::MatrixXf> &Cov_chol, const Eigen::MatrixXf &Cov_chol_matrix,
                   float &out_density){
  int size_x = x.size();
  Eigen::VectorXf x_centre = x - Mean;
  Eigen::VectorXf res_solve = Cov_chol.solve(x_centre);
  // The normalization constant with log(determinant)
  // Eigen::MatrixXf U = Cov_chol.matrixL(); // Lower triangular with the cholesky factor
  float kernelNorm = 0.0;
  float logDiag = 0.0;
  for(int ii=0; ii<size_x; ii++){
    kernelNorm += x_centre[ii] *  res_solve[ii];
    logDiag += std::log(Cov_chol_matrix(ii, ii) );
  }
  out_density = -0.5 * (size_x*std::log(2*M_PI) + 2*logDiag + kernelNorm);
}


// Function to calculate the sum of log densities of the Beta distribution - Takes in input a vector of x
void sum_ldbeta(const std::vector<float> &x, float a, float b, float &out_sum){
	int n = x.size(); // For the number of observations
	float normConst = lgamma(a+b) - lgamma(a) - lgamma(b);
	float res = 0.0;
	for(int i=0; i<n; i++){
		res += (a-1)*log(x[i]) + (b-1)*log(1-x[i]);
	}
	out_sum = res + n*normConst;
}

// Density of Beta distribution
void my_dbeta(float x, float a, float b, float &out_dbeta){
  float normConst = gamma(a+b) / (gamma(a) * gamma(b));
  out_dbeta = normConst * std::pow(x, a-1) * std::pow(1-x, b-1);
}

// Log Density of Beta distribution
void ldbeta(float x, float a, float b, float &out_ldbeta){
  float normConst = lgamma(a+b) - lgamma(a) - lgamma(b);
  out_ldbeta = normConst + (a-1)*log(x) + (b-1)*log(1-x);
}

// Function to calculate the single density of the log Gamma distribution
void ldgamma(float x, float a, float b, float &out_gamma){
	float normConst = a*log(b) - lgamma(a);
	out_gamma = normConst + (a-1)*log(x) -x*b;
	//
}

// Function to calculate the sum of log densities of the Gamma distribution
void sum_ldgamma(const std::vector<float> &x, float a, float b, float &out_sum){
	int n = x.size(); // For the number of observations
	float normConst = a*log(b) - lgamma(a);
	float res = 0.0;
	for(int i=0; i<n; i++){
		res += (a-1)*log(x[i]) -x[i]*b;
	}
	out_sum = res + n*normConst;
}

// Log Beta function
void lbetafg(float a, float b, float &outs){
  outs = lgamma(a) + lgamma(b) - lgamma(a+b);
}

// Beta binomial density without binomial coefficient
void ldbetabinom(int n, int k, float a0, float b0, float an, float bn, float &out_d){
  float dup;
  float dlow;
  lbetafg(an, bn, dup);
  lbetafg(a0, b0, dlow);
  float cons = R::lchoose(n, k);
  out_d = cons + dup - dlow;
}
