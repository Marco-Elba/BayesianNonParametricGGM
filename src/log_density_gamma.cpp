#include <Rcpp.h>
#include <Rmath.h>

void exp_gamma_log(const float eta, const float a, const float b, float &log_density ){
  log_density = ( a * std::log(b) - lgamma(a) + eta*(a-1) -b*std::exp(eta) ) + eta;
}

float inverse_logit( float logit, float &probs ){
  probs = 1 / ( 1 + std::exp( - logit ) );
  return( probs );
}
