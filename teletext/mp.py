import itertools
import pickle
import queue

import multiprocessing as mp
import time


def denumerate(quit_event, pipe, tmp_queue):
    """
    Strips sequence numbers from work_queue items and yields the work.
    If quit_event is set, exit.
    """
    while True:
        if quit_event.is_set():
            return
        else:
            if pipe.poll(timeout=0.1):
                n, item = pipe.recv()
                tmp_queue.put(n)
                yield item


def renumerate(iterator, pipe, tmp_queue):
    """
    Recombines results with the sequence numbers stored in tmp_queue.
    """
    for item in iterator:
        n = tmp_queue.get()
        pipe.send((n, item))


def worker(function, quit_event, pipe, args, kwargs):
    """
    The main function for subprocesses.
    """
    try:
        tmp_queue = queue.Queue() # holds work item numbers to be recombined with the result
        renumerate(function(denumerate(quit_event, pipe, tmp_queue), *args, **kwargs), pipe, tmp_queue)
    finally:
        pass


class _PureGeneratorPoolMP(object):

    def __init__(self, function, processes=1, *args, **kwargs):
        self._processes = processes
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._procs = []
        self._pipes = []

        # Similar to how, on Linux, putting an unpickleable object on a Queue
        # causes an uncatchable exception, passing unpickleable objects to
        # ctx.Process does the same thing on Windows. So we must check that
        # everything can be pickled before attempting to use it. Luckily this
        # is only done once.
        pickle.dumps(self._function)
        pickle.dumps(self._args)
        pickle.dumps(self._kwargs)

        ctx = mp.get_context('spawn')

        # Tells sub-processes that we are done and they should exit.
        self._quit_event = ctx.Event()

        for id in range(processes):
            local, remote = ctx.Pipe(duplex=True)
            p = ctx.Process(target=worker, daemon=True, args=(
                function, self._quit_event, remote, self._args, self._kwargs
            ))
            self._procs.append(p)
            self._pipes.append(local)

    def __enter__(self):
        for p in self._procs:
            p.start()
        return self

    def apply(self, iterable):
        iterable = enumerate(iterable)
        received = {}
        sent_count = 0
        received_count = 0
        done = False

        try:
            # Send 32 items to each pipe to prime it.
            for i in range(8):
                for p, item in zip(self._pipes, itertools.islice(iterable, len(self._pipes))):
                    p.send(item)
                    sent_count += 1

            while True:
                # Wait for any pipe to become ready. No timeout.
                for p in mp.connection.wait(self._pipes):
                    n, item = p.recv()
                    received[n] = item
                    try:
                        p.send(next(iterable))
                        sent_count += 1
                    except StopIteration:
                        done = True

                # Yield what items we can.
                while received_count in received:
                    yield received[received_count]
                    del received[received_count]
                    received_count += 1

                # Check if we've done all the work.
                if done and sent_count == received_count:
                    return

        except (BrokenPipeError, ConnectionResetError, EOFError):
            raise ChildProcessError('A worker process stopped unexpectedly.')

    def __exit__(self, *args):
        self._quit_event.set()
        for p in self._procs:
            p.join()


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
