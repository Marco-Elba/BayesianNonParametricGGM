#include <Rcpp.h>
#include <Rmath.h>

// Start of the header guard
#ifndef LOG_DENSITY_GAMMA_H
#define LOG_DENSITY_GAMMA_H

void exp_gamma_log(const float eta, const float a, const float b, float &log_density );

double inverse_logit( float logit, float &probs );

// End of the header guard
#endif
