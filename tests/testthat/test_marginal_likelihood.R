context("Marginal likelihood optimisation")

a = c(0.6, 0.8, 0.95)
b = c(0.8, 0.8, 0.8) # just test high coverage
lf_obj = simulate_lambda(
  accuracy_true = a, # last column is ground truth (accuracy 100%)
  coverage_true = b,
  seed = 2,
  sample_size = 100
)

test_that("MLL scores truth higher than random numbers",{
  expect_gte(
    marginal_log_likelihood(lf_obj$Lambda, a=a, b=b), marginal_log_likelihood(lf_obj$Lambda, a=rev(a), b=b)
  )
  expect_gte(
    marginal_log_likelihood(lf_obj$Lambda, a=a, b=b), marginal_log_likelihood(lf_obj$Lambda, a=rep(1, 3), b=b)
  )
  expect_gte(
    marginal_log_likelihood(lf_obj$Lambda, a=a, b=b), marginal_log_likelihood(lf_obj$Lambda, a=rep(0.5, 3), b=b)
  )
})

test_that("MLL scores truth higher than random numbers",{
  expect_gte(
    marginal_log_likelihood(lf_obj$Lambda, a=a, b=b), marginal_log_likelihood(lf_obj$Lambda, a=rev(a), b=b)
  )
})

test_that("MLL gradient points in the right direction", {
  expect_identical(sign(marginal_log_likelihood_grad(lf_obj$Lambda, a=c(0.99,0.99,0.99), b=b)), c(-1,-1,-1))
  expect_identical(sign(marginal_log_likelihood_grad(lf_obj$Lambda, a=c(0.85,0.85,0.85), b=b)), c(-1, -1, 1))
  expect_identical(sign(marginal_log_likelihood_grad(lf_obj$Lambda, a=c(0.7,0.7,0.7), b=b)), c(-1, 1, 1))
  expect_identical(sign(marginal_log_likelihood_grad(lf_obj$Lambda, a=c(0.55,0.55,0.55), b=b)), c(1, 1, 1))
})

test_that("MLL gradient gets smaller near the truth", {
  expect_gte(
    sum(abs(marginal_log_likelihood_grad(lf_obj$Lambda, a=c(0.65, 0.85, 0.9), b=b))),
    sum(abs(marginal_log_likelihood_grad(lf_obj$Lambda, a=c(0.6, 0.8, 0.95), b=b)))
  )
})

test_that("MLL gradient descent moves in the right direction", {
  a_init = rep(0.75, 3)
  a_hat = basic_optimiser(
    lf_obj$Lambda, a_init = a_init, maxiters=10, verbose=FALSE)$a_hat
  expect_gte(sum(abs(a_init-a)), sum(abs(a_hat-a)))
})



# now test low coverage
b = c(0.2, 0.2, 0.2) # just test high coverage
lf_obj = simulate_lambda(
  accuracy_true = a, # last column is ground truth (accuracy 100%)
  coverage_true = b,
  seed = 2,
  sample_size = 100
)

test_that("With low coverage, the MLL of truth is better than that of the result of gradient descent", {
  a_init = rep(0.75, 3)
  a_hat = basic_optimiser(
    lf_obj$Lambda, a_init = a_init, maxiters=20, verbose=FALSE)$a_hat
  expect_gte(
    marginal_log_likelihood(lf_obj$Lambda, a = a, b = b),
    marginal_log_likelihood(lf_obj$Lambda, a = a_hat, b = b)
  )
})
