import itertools
import pickle
import queue
import signal

import multiprocessing as mp


def denumerate(quit_event, work_queue, tmp_queue):
    """
    Strips sequence numbers from work_queue items and yields the work.
    If work_queue is empty and quit_event is set, exit.
    """
    while True:
        try:
            n, item = work_queue.get(timeout=0.1)
        except queue.Empty:
            if quit_event.is_set():
                return
        else:
            tmp_queue.put(n)
            yield item


def renumerate(iterator, done_queue, tmp_queue):
    """
    Recombines results with the sequence numbers stored in tmp_queue.
    """
    for item in iterator:
        n = tmp_queue.get()
        done_queue.put((n, item))


def worker(function, started_event, stopped_event, quit_event, work_queue, done_queue, args, kwargs):
    """
    The main function for subprocesses.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        tmp_queue = queue.Queue() # holds work item numbers to be recombined with the result
        started_event.set()
        renumerate(function(denumerate(quit_event, work_queue, tmp_queue), *args, **kwargs), done_queue, tmp_queue)
    finally:
        stopped_event.set()


class _PureGeneratorPoolMP(object):

    def __init__(self, function, processes=1, *args, **kwargs):
        self._processes = processes
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._pool = []

        ctx = mp.get_context('spawn')

        # Work items are placed on this queue by the main process.
        self._work_queue = ctx.Queue()
        # Sub-processes place results on this queue.
        self._done_queue = ctx.Queue()
        # Tells sub-processes that we are done and they should exit.
        self._quit_event = ctx.Event()

        for id in range(processes):
            started_event = ctx.Event()
            stopped_event = ctx.Event()
            p = ctx.Process(target=worker, daemon=True, args=(
                function, started_event, stopped_event, self._quit_event,
                self._work_queue, self._done_queue, self._args, self._kwargs
            ))
            self._pool.append((p, started_event, stopped_event))

    def __enter__(self):
        for p in self._pool:
            p[0].start()
        for p in self._pool:
            if not p[1].wait(timeout=1):
                raise TimeoutError('Timed out waiting for worker process to start.')
        return self

    def _put_work(self, item):
        pickle.dumps(item)
        self._work_queue.put(item)

    @property
    def _workers_exited(self):
        # p[2].is_set means the process wants to exit but is waiting for
        # use to read from the _done_queue, or it has exited.
        # p[0].is_alive() is false when the process has exited.
        # The process may exit without setting p[2] due to race conditions
        # in multiprocessing - see __exit__.
        return (p[2].is_set() or not p[0].is_alive() for p in self._pool)

    def apply(self, iterable):
        iterable = enumerate(iterable)

        sent_count = 0
        received_count = 0

        # Prime the queue with some items.
        for item in itertools.islice(iterable, 32):
            self._put_work(item)
            sent_count += 1

        # Dict to use for sorting received items back into
        # their original order.
        received = {}

        while received_count < sent_count:
            try:
                n, item = self._done_queue.get(timeout=0.1)
            except queue.Empty:
                if any(self._workers_exited):
                    raise ChildProcessError('A worker process stopped unexpectedly.')
            else:
                received[n] = item
                while received_count in received:
                    yield received[received_count]
                    del received[received_count]
                    received_count += 1
                try:
                    self._put_work(next(iterable))
                    sent_count += 1
                except StopIteration:
                    pass

    def __exit__(self, *args):
        # In the event of a KeyboardInterrupt we might end up running this code
        # after multiprocessing killed all remaining worker processes. If that is
        # true then we can't rely on them having set _quit_event before being
        # terminated. Whether this function or multiprocessing's atexit handler
        # runs first is the subject of a race condition due to with/yield semantics.
        self._quit_event.set()
        # _workers_exited checks both _quit_event and P.is_alive() in order to
        # detect when this has happened.
        while not all(self._workers_exited):
            try:
                self._done_queue.get(timeout=0.1)
            except queue.Empty:
                pass
        # We also can't guarantee that the workers emptied the _work_queue if
        # multiprocessing terminated them before we got here.
        self._work_queue.cancel_join_thread()
        for p in self._pool:
            p[0].join()


class _PureGeneratorPoolSingle(object):

    """
    An implementation of PureGeneratorPool that doesn't use multiple processes.
    """

    def __init__(self, function, *args, **kwargs):
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._work_queue = queue.Queue()
        self._proc = self._function(self._work, *args, **kwargs)

    @property
    def _work(self):
        while True:
            try:
                yield self._work_queue.get(block=False)
            except queue.Empty:
                return

    def __enter__(self):
        return self

    def apply(self, iterable):
        for item in iterable:
            self._work_queue.put(item)
            yield next(self._proc)

    def __exit__(self, *args):
        try:
            next(self._proc)
        except StopIteration:
            pass


def PureGeneratorPool(function, processes, *args, **kwargs):

    """
    Implements a parallel processing pool similar to multiprocessing.Pool. However,
    Pool.map(f, i) calls f on every item in i individually. f is expected to return
    the result. PureGeneratorPool.apply(f, i) calls f exactly once for each process
    it starts, and then delivers an iterator containing work items. f is expected
    to yield results. In practice, this means you can pass large objects to f and
    they will only be pickled once rather than for every item in i. It also allows
    you to do one-time setup at the beginning of f.

    f must be a "pure generator". This means it must yield exactly one result for
    each item in the iterator, and that result must only depend on the current
    item being processed. It must not have any mutable state which affects the
    output. For example, any function of the form:

        itertools.partial(map, f)

    is a pure generator if f is pure.

    And further:

        def gen(g, f, it):
            g()
            yield from f(it)

    is a pure generator if f is a pure generator, regardless of whether or not g
    is pure.

    apply() preserves the ordering of items in the input iterator.
    """

    if processes > 1:
        return _PureGeneratorPoolMP(function, processes, *args, **kwargs)
    else:
        return _PureGeneratorPoolSingle(function, *args, **kwargs)


def itermap(function, iterable, processes=1, *args, **kwargs):

    """One-shot function to make a PureGeneratorPool and apply it."""

    with PureGeneratorPool(function, processes, *args, **kwargs) as pool:
        yield from pool.apply(iterable)


if __name__ in ['__main__', '__mp_main__']:

    def f(iterator, *args, **kwargs):
        # f first creates an unpickable, unsharable object. It must be done
        # exactly once per process.
        print('This line MUST be printed exactly once by each process.', args, kwargs)
        for item in iterator:
            #time.sleep(1)
            yield item


if __name__ == '__main__':

    import click
    from tqdm import tqdm

    @click.command()
    @click.option('-j', '--jobs', type=int, default=1000000)
    @click.option('-t', '--threads', type=int, default=2)
    @click.option('-v', '--verbose', is_flag=True)
    def main(jobs, threads, verbose):
        for result in itermap(f, iter(tqdm(range(jobs))), processes=threads, a=2, b=3):
            if(verbose):
                print(result, end=' ')
        print('')

    main()
