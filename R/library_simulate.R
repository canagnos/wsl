#' Labelling function simulator
#'
#' Simulate p weak, independent labelling functions (LF) with known accuracy and coverage
#'
#' @param accuracy_true A px1 vector representing the accuracy of the LFs
#' @param coverage_true A px1 vector representing the coverage of the LFs
#' @param labels_true An nx1 vector of ground truth labels in \code{{-1,1}}. Note that when \code{labels_true} is not \code{NULL}, it overwrites any value of \code{sample_size} that might be provided. If \code{NULL}, \code{labels_true} is generated from a binomial distribution with probability 1/2.
#' @param sample_size The number of labels to generate if \code{labels_true} is \code{NULL}. Overwritten by \code{length(labels_true)} otherwise.
#' @param seed Always set your seed :)
#' @param verbose If \code{TRUE}, some statistics are printed to stdout.
#' @return A list containing \code{labels_true} (the ground truth labels which were either provided or generated), and \code{nxp} matrix taking values in \code{{-1,0,1}}, where \code{-1} represents the negative class, \code{1} represents the positive class, and \code{0} represents a missing value (following standard conventions in WSL).
#' @examples
#' simulate_lambda(
#'     accuracy_true = c(0.7,0.9), coverage_true = c(0.2,0.6))
#' simulate_lambda(
#'     accuracy_true = c(0.7,0.9), coverage_true = c(0.2,0.6), labels_true = c(rep(-1, 5), rep(1,5)))
#' @export
#' @importFrom stats glm predict rbinom rnorm runif
simulate_lambda = function(
  accuracy_true,
  coverage_true,
  labels_true = NULL,
  sample_size = 1000,
  seed = 1,
  verbose=FALSE
){
  set.seed(seed)

  m = length(accuracy_true) # this is the number of labelling functions
  stopifnot(m == length(coverage_true)) # this ensures that accuracy and coverage have same length
  if (verbose){
    cat('Accuracy: ', round(accuracy_true,2), '\n')
    cat('Coverage: ', round(coverage_true,2), '\n')
  }



  if (is.null(labels_true)){
    if(verbose){cat('Simulating labels.\n')}
    labels_true = sample(c(-1,1), sample_size, replace = TRUE)
  } else {
    if(verbose){cat('Received labels.\n')}
    stopifnot(is_valid_label(labels_true, c(-1,1))) # must have no missing values
    sample_size = length(labels_true)
  }

  # initialise output structure
  Lambda = matrix(NA, sample_size, m)


  j = 1
  for (j in 1:m){
    Lambda[,j] = labels_true
    ind_wrong = which(runif(sample_size) > accuracy_true[j])
    ind_missing = which(runif(sample_size) > coverage_true[j])
    Lambda[ind_wrong, j] = - Lambda[ind_wrong, j]
    Lambda[ind_missing, j] = 0
  }


  return(list(Lambda = Lambda, labels_true = labels_true))
}


#' Helper function to check label space
#'
#' This function checks if all the values in a data frame, matrix or vector belong in the given space of labels. It is used for testing.
#'
#' @param y A vector, matrix, or data frame of labels
#' @param label_space A vector of allowed labels
#' @return TRUE/FALSE
#' @examples
#' stopifnot(!is_valid_label(c(0,1,-1), c(-1,1)))
#' stopifnot(is_valid_label(c(0,1,-1), c(-1,0,1)))
#' @export
is_valid_label = function(y, label_space = c(-1,0,1)){
  if (is.data.frame(y)){
    y = as.matrix(y)
  }
  if (is.matrix(y)){
    y = as.vector(y)
  }
  out = all(is.element(y, label_space))
  return(out)
}


sigmoidal = function(x){
  return(1/(1+exp(-x)))
}

logistic = function(x){
  return(log(x/(1-x)))
}

#' Simulate data from a logistic regression
#'
#' This function simulates two datasets from a logistic regression of dimension p: one training dataset, and a test dataset. The coefficients are normally distributed as \code{N(0,betav)} where \code{betav} can be specified by the user. An intercept is added.
#'
#' @param ntrain The number of datapoints in the train dataset
#' @param ntest The number of datapoints in the test dataset
#' @param p The number of covariates (excludin the intercept)
#' @param betav The variance of the prior for the regression coefficients
#' @param seed Always set your seed :)
#' @return A list of two data frames. The first data frame is called \code{data} and has dimension \code{ntrain x (p+1)}. The second data frame is called \code{newdata} and has dimension \code{ntest x (p+1)}. In both cases, the label space is \code{{0,1}} (as per standard convention for logistic regression), and the labels are in the last column of the respective data frames, called \code{Y}.
#' @examples
#' simulate_logreg(ntrain=100)
#' @export
#' @importFrom stats glm predict rbinom rnorm runif
simulate_logreg = function(
  ntrain = 1000,
  ntest = ntrain,
  p = 2,
  betav = 5,
  seed = 2
){
  set.seed(seed)
  n = ntrain + ntest

  # classes are -1 or 1
  beta_true = matrix(rnorm(p+1,0,betav), p+1, 1)
  X = matrix(rnorm(n*p), n, p)
  y = rbinom(
    n = n, size = 1,
    prob = sigmoidal(X %*% beta_true[1:p] + beta_true[p+1])
  )


  data = as.data.frame(cbind(X[1:ntrain, ], y[1:ntrain]))
  colnames(data)[p+1] = 'Y'
  newdata = as.data.frame(cbind(X[(ntrain+1):n, ], y[(ntrain+1):n]))
  colnames(newdata)[p+1] = 'Y'

  out_sim = list(
    data = data,
    newdata = newdata
  )
  return(out_sim)
}


