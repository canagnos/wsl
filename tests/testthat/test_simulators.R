context("Data simulators")

test_that("Labelling function simulator yields right output", {
  out = simulate_lambda(
    accuracy_true = c(0.7,0.9),
    coverage_true = c(0.2,0.6),
    sample_size = 100
  )
  expect_that(dim(out$Lambda), equals(c(100,2)))
  expect_that(is_valid_label(out$Lambda, c(-1,0,1)), is_true())
  expect_that(is_valid_label(out$labels_true, c(-1,1)), is_true())
})

test_that("Coverage of labelling function simulator is correct", {
  coverage_true = c(0.2,0.6)
  out = simulate_lambda(
    accuracy_true = c(0.7,0.9),
    coverage_true = coverage_true,
    sample_size = 10000
  )
  expect_equal(colMeans(out$Lambda == 0), 1-coverage_true, tolerance = 0.01)
  out = simulate_lambda(
    accuracy_true = c(0.7,0.9),
    coverage_true = coverage_true,
    labels_true = c(rep(-1,500), rep(1,500))
  )
  expect_equal(
    colMeans(out$Lambda == 0), 1-coverage_true, tolerance = 0.1)
})


test_that("Accuracy of labelling function simulator is correct", {
  accuracy_true = c(0.7,0.9)
  labels_true = c(rep(-1,500), rep(1,500))
  out = simulate_lambda(
    accuracy_true = accuracy_true,
    coverage_true = c(0.2,0.6),
    labels_true = labels_true
  )
  # check non-missing values against the ground truth
  expect_equal(
    apply(out$Lambda, 2, function(x){
      mean(x[x!=0] == labels_true[x!=0])}),
    accuracy_true, tolerance = 0.05)
})


test_that("Logistic is the inverse of the sigmoidal", {
  expect_equal(sigmoidal(logistic(0.5)), 0.5)
})


test_that("Logistic regression simulator produces output that can be reasonably fitted by logistic regression from the glm package", {
  out_logreg = simulate_logreg(ntrain=100)
  m1 = glm(Y~., out_logreg$data, family='binomial')
  estimated_labels = as.numeric(predict(
    m1, newdata = out_logreg$newdata, type='response') > 0.5)
  # on this simple problem we expect more than 90% accuracy
  expect_that(
    mean(estimated_labels == out_logreg$newdata$Y) > 0.9,
    is_true()
  )
  # check that label space is {0,1}
  expect_that(is_valid_label(out_logreg$Y, c(0,1)), is_true())
})
