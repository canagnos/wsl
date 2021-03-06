% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/library_weak.R
\name{basic_optimiser}
\alias{basic_optimiser}
\title{Basic optimiser}
\usage{
basic_optimiser(Lambda, a_init = NA, stepsize = 0.1, assume_better = TRUE,
  maxiters = 100, verbose = TRUE, upper_bound = 0.999,
  ground_truth = FALSE)
}
\arguments{
\item{Lambda}{An nxm matrix of weak labels, where each row respresents the weak labels produced by m labelling functions on a single data example.}

\item{a_init}{An optional initialisation for the accuracies}

\item{stepsize}{The step size of the gradient descent}

\item{assume_better}{Do we assume that all weak rules are better than chance? Then we can set 0.5 as a lower bound for all accuracies.}

\item{maxiters}{The maximum number of iterations for the gradient descent.}

\item{verbose}{Do we want the optimiser to print out its progress?}

\item{upper_bound}{What is the maximum accuracy allowed?}

\item{ground_truth}{Optionally, we can flag the fact that the last column of Lambda is in fact ground truth, so it can be assumed to have perfect accuracy, when present.}
}
\description{
This function is a very basic gradient descent optimiser of the marginal log likelihood that computes the accuracy and coverage of a set of labelling heuristics on the basis of a matrix of noisy incomplete labels.
}

