
import os
from multiprocessing import Manager
from threading import Thread
from joblib import delayed, Parallel
from rich.progress import TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

import numpy as np
from scipy.sparse import issparse, spmatrix
from exprmat.ansi import error, warning, info


def get_n_jobs(n_jobs):

    if n_jobs is None or (n_jobs < 0 and os.cpu_count() + 1 + n_jobs <= 0): return 1
    elif n_jobs > os.cpu_count(): return os.cpu_count()
    elif n_jobs < 0: return os.cpu_count() + 1 + n_jobs
    else: return n_jobs


def parallelize(
    callback,
    collection,
    n_jobs = None,
    n_split = None,
    unit: str = "",
    as_array: bool = True,
    use_ixs: bool = False,
    backend: str = "loky",
    extractor = None,
    show_progress_bar: bool = True,
):
    """
    Parallelize function call over a collection of elements.

    Parameters
    ----------

    callback
        Function to parallelize.

    collection
        Sequence of items which to chunkify.

    n_jobs
        Number of parallel jobs.

    n_split
        Split `collection` into `n_split` chunks. If `None`, split into `n_jobs` chunks.

    unit
        Unit of the progress bar.

    as_array
        Whether to convert the results not :class:`numpy.ndarray`.

    use_ixs
        Whether to pass indices to the callback.

    backend
        Which backend to use for multiprocessing. See :class:`joblib.Parallel` for valid options.

    extractor
        Function to apply to the result after all jobs have finished.

    show_progress_bar
        Whether to show a progress bar.
    """

    if show_progress_bar: from rich.progress import Progress as progress
    else: progress = None

    def update(pbar, task, queue, n_total):
        
        n_finished = 0
        while n_finished < n_total:
            try: res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    error(f"finished only `{n_finished} out of `{n_total}` tasks.`", e)
                break

            assert res in (None, (1, None), 1)  # (None, 1) means only 1 job
            if res == (1, None):
                n_finished += 1
                if pbar is not None: pbar.advance(task, advance = 1)
            elif res is None: n_finished += 1
            elif pbar is not None: pbar.advance(task, advance = 1)

        if pbar is not None:
            pbar.stop()


    def wrapper(*args, **kwargs):

        if pass_queue and show_progress_bar:
            pbar = None if progress is None else progress(
                TextColumn("[progress.description]"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>4.1f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn()
            )

            pbartask = None
            if pbar is not None: 
                pbartask = pbar.add_task(description = "", total = col_len)
                pbar.start()

            queue = Manager().Queue()
            thread = Thread(target = update, args = (pbar, pbartask, queue, len(collections)))
            thread.start()

        else: pbar, queue, thread = None, None, None

        res = Parallel(n_jobs = n_jobs, backend = backend)(
            delayed(callback)(
                *((i, cs) if use_ixs else (cs,)),
                *args, **kwargs, queue = queue,
            ) for i, cs in enumerate(collections)
        )

        res = np.array(res) if as_array else res
        if thread is not None: thread.join()
        return res if extractor is None else extractor(res)


    col_len = collection.shape[0] if issparse(collection) else len(collection)
    if n_split is None: n_split = get_n_jobs(n_jobs = n_jobs)

    if issparse(collection):
        if n_split == collection.shape[0]:
            collections = [collection[[ix], :] for ix in range(collection.shape[0])]
        else:
            step = collection.shape[0] // n_split
            ixs = [
                np.arange(i * step, min((i + 1) * step, collection.shape[0]))
                for i in range(n_split)
            ]

            ixs[-1] = np.append(
                ixs[-1], np.arange(ixs[-1][-1] + 1, collection.shape[0])
            )

            collections = [collection[ix, :] for ix in filter(len, ixs)]
    else: collections = list(filter(len, np.array_split(collection, n_split)))
    pass_queue = not hasattr(callback, "py_func")  # we'd be inside a numba function
    return wrapper