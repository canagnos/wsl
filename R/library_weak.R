#' @details Weakly supervised learning is a framework for handling multiple noisy incomplete labels, typically generated from a number of labelling heuristics produced by an expert whose intention is to produce noisy labels on a potentially small subset of the available examples. Ground truth can also be accommodated where available. The main function is \code{basic_optimiser} which estimates the accuracy of the labelling heuristics in an unsupervised fashion, under the assumption that any errors in the heuristics are uncorrelated. The proposed framework allows for the use of weighted learning as an approximation to noisy label learning. This is achieved by the function \code{transform_weak_data}.
#'
#' @examples
#' # We first simulate 1000 samples from a logistic regression
#' n = 1000
#' p = 10
#' seed = 11
#' data_obj = simulate_logreg(ntrain=n/2, p=p, betav = 1)
#'
#' # We now simulate the production of weak labels using simulate_lambda
#' # by adding noise and missingness to the ground truth labels
#'
#' lf_obj = simulate_lambda(
#'   accuracy_true = c(0.6, 0.9, 0.65, 1), # last column is ground truth
#'   coverage_true = c(0.1,0.1,0.1,0.05),
#'   labels_true = data_obj$data$Y*2-1 # we map [0,1] to [-1,0,1]
#' )
#'
#' # Our main object of interest is the matrix of weak labels, Lambda:
#'
#' head(lf_obj$Lambda)
#'
#' # Our first task is to estimate the accuracy and coverage parameters from the data
#' # We can do this without using ground truth at all:
#'
#' mle_accuracies_unsupervised = basic_optimiser(
#'   lf_obj$Lambda[, 1:3], ground_truth = FALSE,
#'   maxiters = 10 # increase to 100, this was set small for package testing reasons
#' )
#'
#' # or we can use some ground truth in a semi-supervised manner:
#'
#' mle_accuracies_semisupervised = basic_optimiser(
#'   lf_obj$Lambda,
#'   maxiters = 10, # increase to 100, this was set small for package testing reasons
#'   ground_truth = TRUE
#' )
#'
#' # Armed with maximum likelihood estimates of coverage and accuracy, we can now convert
#' # the weak labels to a larger dataset by converting accuracy to example weights.
#' # We use first the estimate produced by the semi-supervised maximum likelihood estimation,
#' # and the covariates from the training dataset produced by our data simulator at the start:
#'
#' weak_data_semisupervised = transform_weak_data(
#'   X = data_obj$data[, -(p+1)],
#'   Lambda = lf_obj$Lambda,
#'   accuracies = mle_accuracies_semisupervised$a_hat
#' )
#'
#' # And now repeat for the MLEs produced without ground truth:
#'
#' weak_data_unsupervised = transform_weak_data(
#'   X = data_obj$data[, -(p+1)],
#'   Lambda = lf_obj$Lambda,
#'   accuracies = mle_accuracies_unsupervised$a_hat
#' )
#'
#' # We may now use any classifier that supports example weights.
#' # With very little work the standard R logistic regression package
#' # may be used to handle example weights. We use a wrapper provided by the wsl package:
#'
#' weak_fit_semisupervised = fit_logistic(
#'   data = weak_data_semisupervised$dataweak,
#'   weights = weak_data_semisupervised$weights
#' )
#' weak_fit_unsupervised = fit_logistic(
#'   data = weak_data_unsupervised$dataweak,
#'   weights = weak_data_unsupervised$weights
#' )
#'
#' # We now deploy both models on the test dataset:
#'
#' output_weak_fit_semisupervised = predict_logistic(
#'   weak_fit_semisupervised,
#'   data_obj$newdata
#' )
#' output_weak_fit_unsupervised = predict_logistic(
#'   weak_fit_unsupervised,
#'   data_obj$newdata
#' )
#'
#' # For comparison purposes, we also fit a logistic regression on the ground truth data, too:
#'
#' ground_truth_fit = fit_logistic(data_obj$data)
#' output_ground_truth_fit = predict_logistic(ground_truth_fit, data_obj$newdata)
#'
#' # We can now compare performance, e.g., by using the hmeasure package:
#' library(hmeasure)
#' scores = data.frame(
#'   TL = output_ground_truth_fit$scores,
#'   WTL_MLE = output_weak_fit_semisupervised$scores, # weak and true labels
#'   WL_MLE = output_weak_fit_unsupervised$scores # only weak labels
#' )
#' hmeasure_obj = HMeasure(scores = scores, true.class = data_obj$newdata$Y)
#' plotROC(hmeasure_obj)
"_PACKAGE"



#' Probability density of true label and observed weak labels given accuracy and coverage
#'
#' This function represents the basic generative model of weak supervision, where the observed values of a set of labelling functions are related via a simple binomial model to estimates of accuracy and coverage for each labelling function.
#'
#' @param lambda An mx1 vector of weak labels in \code{{-1,0,1}}.
#' @param y A scalar ground truth label in \code{{-1,1}} Cannot be missing.
#' @param a An mx1 vector of accuracy estimates in \code{[0,1]} for each labelling function.
#' @param b An mx1 vector of coverage estimates in \code{[0,1]} for each labelling function.
#' @param grad A boolean flag. If TRUE, the gradient of the density is computed instead.
#'
#' @return The probability density of an observed set of weak labels, given parameter values for accuracy and coverage.
accuracy_coverage_density = function(
  lambda, # m x 1
  y, # scalar, in {-1,0,1}
  a=rep(0.8, length(lambda)), # m x 1
  b=rep(0.8, length(lambda)), # m x 1
  grad=FALSE # compute the function, or its gradient?
){
  m = length(lambda)
  stopifnot(length(a) == m)
  stopifnot(length(b) == m)
  stopifnot(is_valid_label(y, c(-1,1)))
  stopifnot(length(y) == 1)

  out = 1

  s1 = b*a*as.numeric(lambda == y)
  s2 = b*(1-a)*as.numeric(lambda == -y)
  s3 = (1-b)*as.numeric(lambda == 0)

  s = prod(s1+s2+s3)
  if (grad){
    out = rep(NA, m)
    for (j in 1:m){
      out[j] = (s/(s1[j] + s2[j] + s3[j]))*(b[j]*as.numeric(lambda[j] == y) - b[j]*as.numeric(lambda[j] == -y))
    }
    return(0.5*out)
  } else {
    return(0.5*s)
  }
}

#' Marginal log likelihood gradient
#'
#' This function computes the gradient of the marginal log likelihood of a set of weak labels with respect to the accuracy vector (the coverage vector can be estimated in closed-form).
#'
#' @param Lambda An nxm matrix of weak labels, where each row respresents the weak labels produced by m labelling functions on a single data example.
#' @param a An mx1 estimate of accuracy probabilities.
#' @param b An mx1 estimate of coverage probabilities
#' @return An mx1 vector where the i'th element represents the partial derivative of the marginal log likelihood with respect to the accuracy of the i'th labelling function, evaluated at the currently provided values a.
marginal_log_likelihood_grad = function(
  Lambda, # n x m matrix
  a = rep(0.8, dim(Lambda)[2]), # m x 1 vector
  b = rep(0.8, dim(Lambda)[2]) # m x 1 vector
){
  n = dim(Lambda)[1]
  m = dim(Lambda)[2]
  stopifnot(length(a) == m)
  stopifnot(length(b) == m)
  stopifnot(is_valid_label(as.vector(Lambda), c(-1,0,1)))

  sgrad = rep(0, m)
  for (i in 1:n){
    x = Lambda[i,]
    nom = accuracy_coverage_density(x, 1, a, b, grad=TRUE) + accuracy_coverage_density(x, -1, a, b, grad=TRUE)
    denom = accuracy_coverage_density(x, 1, a, b) + accuracy_coverage_density(x, -1, a, b)
    sgrad = sgrad + nom/denom
  }
  sgrad = sgrad/n

  return(sgrad)
}

#' Marginal log likelihood gradient
#'
#' This function computes the marginal log likelihood of a set of weak labels given an estimate of accuracy and coverage for each labelling function.
#'
#' @param Lambda An nxm matrix of weak labels, where each row respresents the weak labels produced by m labelling functions on a single data example.
#' @param a An mx1 estimate of accuracy probabilities.
#' @param b An mx1 estimate of coverage probabilities
#' @return A scalar value equal to the marginal log likelihood.
marginal_log_likelihood = function(
  Lambda,  # n x m matrix
  a = rep(0.8, dim(Lambda)[2]), # m x 1 vector
  b = rep(0.8, dim(Lambda)[2]) # m x 1 vector
){
  n = dim(Lambda)[1]
  m = dim(Lambda)[2]
  stopifnot(length(a) == m)
  stopifnot(length(b) == m)
  stopifnot(is_valid_label(as.vector(Lambda), c(-1,0,1)))

  s = mean(apply(Lambda, 1, function(x){
    log(accuracy_coverage_density(x, 1, a, b) + accuracy_coverage_density(x, -1, a, b))
  }))
  return(s)
}

#' Basic optimiser
#'
#' This function is a very basic gradient descent optimiser of the marginal log likelihood that computes the accuracy and coverage of a set of labelling heuristics on the basis of a matrix of noisy incomplete labels.
#'
#' @param Lambda An nxm matrix of weak labels, where each row respresents the weak labels produced by m labelling functions on a single data example.
#' @param a_init An optional initialisation for the accuracies
#' @param stepsize The step size of the gradient descent
#' @param assume_better Do we assume that all weak rules are better than chance? Then we can set 0.5 as a lower bound for all accuracies.
#' @param maxiters The maximum number of iterations for the gradient descent.
#' @param verbose Do we want the optimiser to print out its progress?
#' @param upper_bound What is the maximum accuracy allowed?
#' @param ground_truth Optionally, we can flag the fact that the last column of Lambda is in fact ground truth, so it can be assumed to have perfect accuracy, when present.
#'
#' @export
basic_optimiser = function(
  Lambda,
  a_init = NA,
  stepsize = 0.1,
  assume_better = TRUE,
  maxiters = 100,
  verbose = TRUE,
  upper_bound = 0.999,
  ground_truth = FALSE
){
  # only works for two labelling functions
  if (assume_better){
    start_at = 0.51
  } else {
    start_at = 0.01
  }

  n = dim(Lambda)[1]
  m = dim(Lambda)[2]
  stopifnot(m <= 4)
  stopifnot(is_valid_label((Lambda), c(-1,0,1)))

  if (any(is.na(a_init))){
    a_tmp = rep(0.75, m)
  } else {
    a_tmp = a_init
  }

  b_hat = apply(Lambda, 2, function(x){mean(x!=0)})
  loss = rep(NA, maxiters)
  a_trajectory = matrix(NA, maxiters, m)
  iter = 1
  best_loss = -Inf
  for (iter in 1:maxiters){

    a_tmp = a_tmp + stepsize * marginal_log_likelihood_grad(
      Lambda, a = a_tmp, b=b_hat)
    if (ground_truth){a_tmp[m] = upper_bound} # fix ground truth if known
    a_tmp = pmax(pmin(a_tmp, upper_bound), start_at)
    loss[iter] = marginal_log_likelihood(Lambda, a = a_tmp, b = b_hat)
    if (loss[iter] > best_loss){
      a_hat = a_tmp
      best_loss = loss[iter]
    }
    a_trajectory[iter, ] = a_tmp
    if (verbose){cat(iter/maxiters, ': ', a_tmp, ' (', loss[iter], ')\n')}
  }

  # an alternative closed-form estimate of the accuracy
  if (ground_truth){
    y_true = Lambda[,m]
    a_tilde = rep(NA, m)
    for (j in 1:(m-1)){
      ind_simultaneously_observed = which((Lambda[,m] != 0) & (Lambda[,j] != 0))
      a_tilde[j] = mean(
        y_true[ind_simultaneously_observed] == Lambda[ind_simultaneously_observed,j]
      )
    }
    a_tilde[m] = upper_bound
  } else {
    a_tilde = rep(NA, m)
  }
  out = list(a_hat = a_hat, b_hat = b_hat, a_tilde = a_tilde, traj = a_trajectory)
  return(out)
}

#' Plotting routine for accuracy estimates
#'
#' A convenience function for plotting accuracy trajectories
#'
#' @param traj A matrix containing the trajectories of the estimates of accuracy during gradient descent
#' @param a_true The true accuracy values
#' @param main A title for the plot
#'
#' @importFrom graphics abline lines plot
#'
#'@export
plot_accuracy_estimates = function(traj, a_true, main='Accuracy Estimates'){
  m = dim(traj)[2]

  #par(mfrow=c(1,2))
  #plot(out$loss, type='l', main = 'Marginal Log Likelihod', ylab='Loss', xlab='Iterations')
  cols = c('blue','green','black')
  for (j in 1:m){
    if (j==1){
      plot(traj[,j], col=cols[j], ylim=c(0.5,1), type='l', main = main, ylab='a*', xlab = 'Iterations')
    } else {
      lines(traj[,j], col=cols[j], type='l')
    }
    abline(h=a_true[j], col=cols[j], lty=2)

  }
#  if (!is.null(out$a_tilde)){
#    abline(h=a_tilde[1:3], col=cols, lty=2)
#  }

}

#' A transformation from weakly labelled examples to weighted examples
#'
#' It is possible to map weak labels to weighted learning, and in this fashion perform noise-aware empirical loss minimisation while using existing classifier interfaces
#'
#' @param X is an nxp matrix of data examples - note this should not include the ground truth label
#' @param Lambda is the matrix of weak labels
#' @param accuracies is a vector of estimated (or known) accuracies for each labelling rule. This can include perfect accuracy in the presence of ground truth.
#'
#'
#'@export
#'
transform_weak_data = function(X, Lambda, accuracies){
  stopifnot(is_valid_label(Lambda, label_space =c(-1,0,1)))

  # concatenate all non-missing entries
  m = dim(Lambda)[2]
  j = 1
  k = dim(X)[2]
  dataweak = matrix(NA, 0, k+1)
  weights = rep(NA, 0)
  for (j in 1:m){

    # remove any missing labels
    ind = which(Lambda[,j] != 0)

    # append the covariates together with the jth weak label
    new_chunk = cbind(X[ind,], (Lambda[ind, j]+1)/2)
    dataweak = rbind(dataweak, new_chunk)
    weights = c(weights, rep(accuracies[j], length(ind)))

  }
  colnames(dataweak) = c(colnames(X), 'Y')
  dataweak = as.data.frame(dataweak)
  return(list(dataweak = dataweak, weights = weights))
}

#' Run an entire marginal log likelihood optimisation experiment
#'
#' A convenience function to run experiments with different accuracy and coverage parameters and assess the performance of the optimisation routine
#'
#' @param accuracy the desired accuracy for each labelling rule
#' @param coverage the desired coverage for each labelling rule
#' @param ground_truth whether to allow the optimiser to see ground truth or not
#' @param n The number of examples to generate
#' @param p The number of covariates to generate
#' @param maxiters The maximum number of iterations for the gradient descent loop
#' @param stepsize The step size for the gradient descent loop
#'
#' @export
run_experiment = function(
  accuracy,
  coverage,
  ground_truth = FALSE,
  n=1000,
  p=10,
  maxiters=200,
  stepsize=0.5){
  # generate data
  data_obj = simulate_logreg(ntrain=n/2, p=p, betav = 1)

  # simulate labelling functions
  lf_obj = simulate_lambda(
    accuracy_true = accuracy, # last column is ground truth (accuracy 100%)
    coverage_true = coverage,
    seed = 2,
    labels_true = data_obj$data$Y*2-1
  )

  if (ground_truth){
    Lambda = lf_obj$Lambda
  } else {
    cat('No ground truth.\n')
    Lambda = lf_obj$Lambda[, 1:3]
  }
  # first we use no ground truth at all (remove last column)
  mle_accuracies_unsupervised = basic_optimiser(
    Lambda,
    stepsize = stepsize,
    maxiters = maxiters,
    ground_truth = ground_truth
  )

  return(mle_accuracies_unsupervised)
}
