
# special objects for the trials dictionary 
from collections import namedtuple
rk = namedtuple('trial_params', 'd, i')
import pandas
from sklearn.utils import check_random_state

from exprmat import info, warning, error


# value
def rv(**kwargs):
    """
    Basically a nice wrapper around kwargs to avoid having to type out dictionary syntax.
    Returns dictionary object.
    """
    return kwargs


def check_integer_list(
    num_list, min_val = None, max_val = None, 
    var_name = "variable", return_as_list = True):
    
    # formats into an iterable
    _check_integer_boolean = isinstance(num_list, int)

    if _check_integer_boolean:
        num_list = [num_list]
    elif isinstance(num_list, range):
        if len(num_list) == 0: error(f"{var_name} specified range has length 0.")
        else: pass
    elif isinstance(num_list, list):
        if len(num_list) == 0: error(f"{var_name} specified list has length 0.")
        else:
            for num in num_list:
                if not isinstance(num, int):
                    error(f"{var_name} values must be integers.")
                else: pass
    else: error(f"{var_name} must be integer, or a non-empty list/range.")

    # check that each value in the iterable satisfies the limits
    for num in num_list:
        if min_val is not None:
            if num < min_val: error(f"{var_name} must be greater than {min_val}.")
        if max_val is not None:
            if num > max_val: error(f"{var_name} must be less than {max_val}.")

    # the formated integer can be returned in the original format if specified
    if _check_integer_boolean and return_as_list == False:
        num_list = num_list[0]
    
    return num_list


def check_integer(num, min_val = None, max_val = None, var_name = "variable"):
    
    # check that it is an integer
    if not isinstance(num, int):
        error(f"{var_name}: must be an integer.")

    # check that the integer value satisfies the limits
    if min_val is not None:
        if num < min_val: error(f"{var_name} must be greater than {min_val}.")
    if max_val is not None:
        if num > max_val: error(f"{var_name} must be less than {max_val}.")

    return num


def check_dict(dict_object, true_n_clusters, true_n_cells):
    
    # check n_clusters and n_cells are integers
    true_n_clusters = check_integer(true_n_clusters)
    true_n_cells    = check_integer(true_n_cells)

    # check the number of cells
    n_cells_list = [len(x['labels']) for x in dict_object.values()]
    assert n_cells_list.count(true_n_cells) == len(n_cells_list), "number of cells is not consistent between trials"
    n_cells = true_n_cells

    # check the number of clusters
    n_clusters_list = [x['facility'].shape[0] for x in dict_object.values()]
    assert n_clusters_list.count(true_n_clusters) == len(n_clusters_list), "number of clusters not consistent between trials"
    n_clusters =  true_n_clusters

    return dict_object

    # check the number of cells
    # n_cells_list = [len(x['labels']) for x in dict_object.values()]
    # assert n_cells_list.count(true_n_cells) == len(n_cells_list), "number of cells not consistent"
    # n_cells = true_n_cells

    # # check the number of clusters
    # # there is an edge case for this, which is if the kth cluster is not assigned any cells
    # # will fix if error occurs, otherwise can just count that at least 75% of runs is correct?
    # n_clusters_list = [np.max(x['labels']) + 1 for x in dict_object.values()]
    # assert n_clusters_list.count(true_n_clusters) == len(n_clusters_list), "number of cells not consistent"
    # n_clusters =  true_n_clusters


def run_trials_minibatch_kmeans(data, n_clusters, d_range, n_runs, batch_size, random_state):

    import itertools
    from sklearn.cluster import MiniBatchKMeans

    # check that consensus matrix has no NaN, infinity values
    from sklearn.utils import check_array
    data = check_array(data)

    # check n_clusters
    n_clusters = check_integer(
        n_clusters,
        min_val = 2,
        var_name = 'n_clusters'
    )

    # check d_range
    d_range = check_integer_list(
        d_range,
        min_val = 2,
        max_val = data.shape[1],
        var_name = 'd_range'
    )

    # check n_runs
    n_runs = check_integer(n_runs, min_val=1, var_name="n_runs")

    # check batch_size
    batch_size = check_integer(
        batch_size,
        min_val = 10,
        max_val = data.shape[0],
        var_name = 'batch_size'
    )

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

    trials_dict = {}
    for d, i in itertools.product(d_range, range(1, n_runs + 1)):
        kmeans = MiniBatchKMeans(
            n_clusters = n_clusters,
            batch_size = batch_size,
            random_state = random_state
        )

        # cluster the points
        kmeans.fit(data[:, :d])

        # add results into dictionary
        trials_dict[(d, i)] = rv(
            facility = kmeans.cluster_centers_,
            labels = kmeans.labels_,
            inertia = kmeans.inertia_
        )
    
    return trials_dict


def combine_facilities(dict_object, K, n_facility, batch_size, random_state):

    from sklearn.cluster import MiniBatchKMeans
    import numpy as np

    # check that the dictionary object is formatted correctly
    assert isinstance(dict_object, dict)
    for key, value in dict_object.items():
        assert isinstance(value, dict)
        assert value['facility'] is not None
        assert value['labels'] is not None

    # check arguments are formated as integers (limits not trivial to do)
    K = check_integer(K, var_name = 'K')
    n_facility = check_integer(n_facility, var_name = 'n_facility')
    batch_size = check_integer(batch_size, var_name = 'batch_size')

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

    dict_object_combined = {}

    # combine the facilities from different trials
    for key, value in dict_object.items():
        facilities = value['facility']
        labels = value['labels']

        # count the number of cells assigned to each facility
        weights = list(map(
            lambda cluster_no: np.count_nonzero(labels == cluster_no), 
            range(0, n_facility)
        ))

        # cluster the facility using weighted k means
        kmeans = MiniBatchKMeans(
            n_clusters = K,
            batch_size = batch_size,
            random_state = random_state
        )
        kmeans.fit(facilities, sample_weight = weights)

        # write result
        dict_object_combined[key] = rv(
            facility = kmeans.cluster_centers_,
            labels = kmeans.labels_[labels], 
            inertia = kmeans.inertia_
        )

    return dict_object_combined



def convert_dict_into_binary_matrix(dict_object, true_n_clusters, true_n_cells):

    import numpy as np

    n_clusters, n_cells = true_n_clusters, true_n_cells

    # check dict_object has correct number of clusters and cells for every run
    dict_object = check_dict(dict_object, true_n_clusters, true_n_cells)

    # initialise empty B array with the correct shape
    B = np.zeros((n_cells, 0), dtype=int)

    # for each run, we create the block to be appended to B
    for i, value in enumerate(dict_object.values()):
        cell_labels = value['labels']

        # create a total of 'n_clusters' columns for this iteration
        b = np.zeros((n_cells, n_clusters), dtype=int)

        # annotate the correct columns for each row/cell
        b[range(0, n_cells), cell_labels] = 1

        # this checks that every cell only has one cluster assignment
        assert np.all(np.sum(b, axis=1) == 1), "some cells have multiple cluster assignments"

        # append to the right of the overall binary consensus matrix
        B = np.append(B, b, axis=1)

    # remove columns in B with no cells assigned 
    # this happens if some initialised centroids do not get assigned anything
    cols_with_no_cells = np.sum(B, axis=0) != 0
    if not np.all(cols_with_no_cells):
        B = B[:, np.sum(B, axis=0) != 0]

    return B


def cluster_consensus_matrix(consensus_matrix, n_clusters, batch_size, random_state):

    # check that consensus matrix has no NaN, infinity values
    from sklearn.utils import check_array
    consensus_matrix = check_array(consensus_matrix)

    # check n_clusters
    n_clusters = check_integer(
        n_clusters,
        min_val = 2,
        max_val = consensus_matrix.shape[0],
        var_name = 'n_clusters'
    )

    # check batch_size
    batch_size = check_integer(
        batch_size,
        min_val = 10,
        max_val = consensus_matrix.shape[0],
        var_name = 'batch_size'
    )

    # check random state
    from sklearn.utils import check_random_state
    random_state = check_random_state(random_state)

    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters = n_clusters,
        batch_size = batch_size,
        random_state = random_state
    )
    kmeans.fit(consensus_matrix)

    return kmeans.labels_


def consensus(
    adata,
    basis = 'pca',
    key_added = 'sc3',
    n_clusters = [3, 5, 10],
    d_range = None,
    n_runs = 5,
    n_facility = None,
    multiplier_facility = None,
    batch_size = None,
    random_state = None
):
    """\
    Run consensus clustering to cluster cells in an AnnData object.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_clusters
        Number of clusters. Default: [3,5,10]
    d_range
        Number of PCs. Default is 25, or the number of PCs in the 
        AnnData object, whichever is lower. Can accept a list
        (e.g. `[15, 20, 25`]).
    n_runs
        Number of realisations to perform for the consensus.
        Default is 5, recommended > 1.
    n_facility
        Number of microclusters. Overridden if `multiplier_facility`
        is provided with a value.
    multiplier_facility
        Multiplier for microclusters. Number of microclusters
        is calculated as this parameter multiplied by the max of
        `n_clusters`. Default is 3.
    batch_size
        Batch size for k-means. Default is 100.
    random_state
        Random state of the algorithm.

    Returns
    -------
    AnnData object with cluster labels written in the `.obs` dataframe.

    adata.obs['sc3.{k}']
        Labels for `n_clusters = k`.
    """

    # check that AnnData object already has dim reduc
    if basis in adata.obsm.keys(): X_pca = adata.obsm[basis]
    else: error(f'{basis} does not present in your obsm slot')

    # get the number of cells from row dimension
    n_cells = X_pca.shape[0]

    # check and formats n_clusters into in a list format, using default values if unprovided
    n_clusters = check_integer_list(
        n_clusters, 
        min_val = 2, 
        max_val = n_cells,
        var_name = 'n_clusters'
    )

    # check d_range
    if d_range is None:
        d_range = [min(25, X_pca.shape[1])]
    d_range = check_integer_list(
        d_range,
        min_val = 2,
        max_val = X_pca.shape[1],
        var_name = 'd_range'
    )

    # check n_runs
    n_runs = check_integer(n_runs, min_val = 1, var_name = "n_runs")

    # check batch_size
    if batch_size is None:
        batch_size=min(100, adata.shape[0])
    if isinstance(batch_size, int):
        if batch_size > n_cells:
            error("batch size for k-means must be smaller than number of cells.")
    else: error("batch size must be positive integer value (default: 100)")

    # check random state
    random_state = check_random_state(random_state)
 
    # determine number of facilities to use
    if n_facility is not None:
        warning(f"n_facility set to {n_facility}, ignoring multiplier_facility parameter")
    else:
        if multiplier_facility is None:
            info(f"multiplier_facility not set, using default value of 3")
            multiplier_facility = 3
        
        # use multiplier value to calculate
        n_facility = max(n_clusters) * multiplier_facility

        if n_facility >= n_cells:
            n_facility = n_cells - 1

        info(f"number of facilities calculated as {n_facility}")    
    
    n_facility = check_integer(
        n_facility, 
        min_val = 2, 
        max_val = n_cells,
        var_name = 'n_facility'
    )

    # run over many different trials
    trials_dict = run_trials_minibatch_kmeans(
        data = X_pca,
        n_clusters = n_facility,
        d_range = d_range,
        n_runs = n_runs,
        batch_size = batch_size,
        random_state = random_state
    )

    # write the individual trials into AnnData object
    adata.uns[f'{key_added}.trials'] = trials_dict

    # run different number of clusters
    for K in n_clusters:
        consensus_dict = combine_facilities(
            dict_object = trials_dict, 
            K = K, 
            n_facility = n_facility,
            batch_size = batch_size, 
            random_state = random_state
        )

        B = convert_dict_into_binary_matrix(
            dict_object = consensus_dict,
            true_n_clusters = K,
            true_n_cells = n_cells
        )

        cell_labels = cluster_consensus_matrix(
            consensus_matrix = B,
            n_clusters = K,
            batch_size = batch_size,
            random_state = random_state
        )

        # write results into AnnData object
        adata.obs[f'{key_added}.{K}'] = cell_labels
        adata.obs[f'{key_added}.{K}'] = adata.obs[f'{key_added}.{K}'].astype('str').astype('category')

    return adata