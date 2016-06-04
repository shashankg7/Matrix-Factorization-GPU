# Matrix-factorization using Alternating least square method on GPU

Large scale matrix factorization on GPU. Instructions to run:

> import MF

> mf = MF(size_of_square_matrix, dim_of_factors, regularization_coef, learning_rate)

> mf.mat_fact(num_iters)

> W, H = mf.get_factors()

Requirements:

1. Theano
2. Numpy

