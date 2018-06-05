
#' A simple wrapper for the glm function
#'
#' This wrapper simplifies the interface of the glm function to that of a logistic classifier. It also abuses the inteface of the glm a little bit, using the optional parameter 'weights' which formally corresponds to the number of trials in a binomial experiment as a proxy for an estimated accuracy of that particular label. This is a theoretically sensible move, which allows us to use existing interfaces for logistic regression in many languages, but can be improved with a re-implemented gradient descent.
#'
#' @param data An nx(p+1) matrix, with one column called "Y" containing the labels, and p columns containing the covariates (categorical or numeric).
#' @param weights (optional) An nx1 vector of weights in \code{[0,1]} representing the accuracy of each label (formally the number of trials in that binomial experiment).
#' @return A fitted glm object.
#' @export
#' @importFrom stats glm predict rbinom rnorm runif
fit_logistic = function(data, weights=NULL){
  stopifnot(is_valid_label(data$Y, label_space = c(0,1)))
  if (!is.null(weights)){
    weights = round(weights*100)
  }
  m_obj = glm(
    Y~.,
    data = data,
    family='binomial',
    weights = weights
  )

  return(m_obj)
}


#' A simple wrapper for the predict method of a glm object
#'
#' This wrapper simplifies the interface of the predict method of a glm object to that of a logistic classifier.
#'
#' @param m_obj A fitted glm object
#' @param newdata A nxp matrix with columns identically named to the covariates that were used to train the model. The matrix can have additional columns without having an effect on the output.
#' @return A list with two elements: \code{scores} is an nx1 vector of probabilities of the respective label being equal to 1, and \code{y_hat} are the estimated labels with 0.5 as the decision boundary.
#' @export
#' @importFrom stats glm predict rbinom rnorm runif
predict_logistic = function(m_obj, newdata){
  stopifnot(is_valid_label(newdata$Y, label_space = c(0,1)))
  scores = predict(
    m_obj, newdata = newdata, type = 'response')
  return(list(scores=scores, y_hat = as.numeric(scores > 0.5)))
}

