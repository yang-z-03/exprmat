
from scipy.cluster.hierarchy import cophenet, single
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import non_negative_factorization


def nmf(
    matrix, n_comps, W = None, H = None,
    init = None, update_h = True, solver = 'cd', 
    beta_loss = 'frobenius', tolerance = 0.0001, max_iter = 200, 
    alpha_w = 0.0, alpha_h = 'same', l1_ratio = 0.0, 
    random_state = 42, verbose = 0, shuffle = False
):
    
    w, h, n = non_negative_factorization(
        matrix, W, H, n_components = n_comps,
        init = init, 
        update_H = update_h,
        solver = solver,
        beta_loss = beta_loss,
        tol = tolerance,
        max_iter = max_iter,
        alpha_W = alpha_w,
        alpha_H = alpha_h,
        l1_ratio = l1_ratio,
        random_state = random_state,
        verbose = verbose,
        shuffle = shuffle
    )

    return w, h, n
