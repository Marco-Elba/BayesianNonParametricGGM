#include <vector>
#include <Rcpp.h>
#include <Rmath.h>
#include <RcppEigen.h>
#include <Eigen/Dense>


void which_rmultinom_Eigen(int M, const Eigen::VectorXf &probsAll, int &out_pos){
  // Random sample from the Multinomial distribution
  // Create a dynamic vector of integers of length K
  int *draw = new int[M];
  // Create extra vector (maybe there is a better solution)
  double *probs = new double[M];
  double sum_probs = 0;
  for(int jj=0; jj < M; jj++){
    probs[jj] = probsAll[jj];
    sum_probs += probs[jj];
  }
  for(int jj=0; jj < M; jj++){
    probs[jj] /= sum_probs;
  }
  // Put the result in the vector draw
  rmultinom(1, probs, M, draw );
  //
  int elem = 0;
  int pos = 0;
  while(pos==0 && elem < M){
    if(draw[elem]==1){
      pos=elem;
    }
    ++elem;
  } // end while
  out_pos = pos;
  // Delete the dinamically allocated array !!
  delete[] probs;
  delete[] draw;
}
