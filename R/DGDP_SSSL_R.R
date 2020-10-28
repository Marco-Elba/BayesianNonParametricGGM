#' @export
R_DGDP_SSSL <- function(Ydata, Xdata, M, pcovs, Ngroups, groups_dim, iterations, burn_in, tinning,
                        alpha_a, alpha_b, K, pi_a, pi_b, v0, h, lambda_Sigma,
                        scale_global, scale_proposal){
  ##
  if( iterations < burn_in )   stop( "Number of iteration must be more than number of burn-in" )
  if( any( is.na( Ydata ) ) ) stop( "Method does not deal with missing values" )
  ##
  n <- dim( Ydata )[1] # Number of obs
  if( sum(groups_dim) != n) stop("Number of observations does not match groups size")
  ##
  size_save = floor((iterations - burn_in) / tinning)
  
  out_beta = matrix(0, size_save, M * pcovs )
  out_alpha_mu = matrix(0, size_save, Ngroups+1 )
  out_Omega = matrix(0.0, size_save, (M*(M-1)/2 + M) * Ngroups)
  out_Z = matrix( as.integer(0), size_save, (M*(M-1)/2 + M) * Ngroups)
  out_acc_alpha = integer(1)
  out_pi = matrix(0.0, size_save, Ngroups)
  out_cind = matrix( as.integer(0), size_save, (M*(M-1)/2) * Ngroups)
  out_nclus = integer( size_save )
  ## ---------------------------------------------------------------------------- ##
  invisible(.Call(`_NPMGGM_MultipleGGMs_SSSL_DGDP`,
                  as.integer(n),
                  as.integer(M),
                  as.integer(Ngroups),
                  as.integer(pcovs),
                  as.integer(groups_dim),
                  Ydata, Xdata,
                  alpha_a, alpha_b,
                  as.integer(K),
                  pi_a, pi_b,
                  v0, h,
                  lambda_Sigma,
                  scale_global,
                  scale_proposal,
                  iterations,
                  burn_in,
                  tinning,
                  out_Omega,
                  out_Z,
                  out_pi,
                  out_cind,
                  out_beta,
                  out_nclus,
                  out_acc_alpha,
                  out_alpha_mu)
  )
  ## ---------------------------------------------------------------------------- ##
  # make a list with one element for each group
  Omega = vector(mode="list", length = Ngroups)
  Z = vector(mode="list", length = Ngroups)
  cluster_allocation = vector(mode="list", length = Ngroups)
  # each element is an array with the MCMC draws
  for(gr in 1:Ngroups){
    Omega[[gr]] = array(0, dim = c(size_save, M, M))
    Z[[gr]] = array(0, dim = c(size_save, M, M))
    cluster_allocation[[gr]] = array(0, dim = c(size_save, M, M))
  }
  Beta = array(0, dim = c(size_save, M, pcovs))
  
  # fill the objects to output
  n_upper_plus_diag = M*(M-1)/2 + M
  n_upper = M*(M-1)/2
  mat_temp = matrix(0, M, M)
  mat_temp_cind = matrix(0, M, M)
  upper_tri = upper.tri(mat_temp, diag=TRUE)
  upper_tri_no_diag = upper.tri(mat_temp, diag=FALSE)
  for(gr in 1:Ngroups){
    for(mcmc_sample in 1:size_save){
      # Omega
      mat_temp[upper_tri] = out_Omega[mcmc_sample, ((gr-1)*n_upper_plus_diag + 1):(n_upper_plus_diag*gr)]
      Omega[[gr]][mcmc_sample, , ] = mat_temp
      # Z
      mat_temp[upper_tri] = out_Z[mcmc_sample, ((gr-1)*n_upper_plus_diag + 1):(n_upper_plus_diag*gr)]
      Z[[gr]][mcmc_sample, , ] = mat_temp
      # Cluster allocation
      mat_temp_cind[upper_tri_no_diag] = out_cind[mcmc_sample, ((gr-1)*n_upper + 1):(n_upper*gr)]
      cluster_allocation[[gr]][mcmc_sample, , ] = mat_temp_cind
    }
  }
  for(jj in 1:M){
    Beta[ , jj, ] = out_beta[, ((jj-1)*pcovs + 1):(pcovs*jj) ]
  }
  output <- list(out_Omega=Omega,
                 out_Z=Z,
                 out_pi=out_pi,
                 out_cind=cluster_allocation,
                 out_beta=Beta,
                 out_nclus=out_nclus,
                 out_acc_alpha=out_acc_alpha,
                 out_alpha_mu=out_alpha_mu
  )
  return(output)
}
