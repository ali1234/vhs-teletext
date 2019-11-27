import atexit
import itertools
import pickle
import queue

import multiprocessing as mp

import zmq


def denumerate(work, control, tmp_queue):

    """Strips sequence numbers from work_queue items and yields the work."""

    poller = zmq.Poller()
    poller.register(work, zmq.POLLIN)
    poller.register(control, zmq.POLLIN)

    while True:
        socks = dict(poller.poll())
        if socks.get(work) == zmq.POLLIN:
            n, item = work.recv_pyobj()
            tmp_queue.put((n, len(item)))
            yield from item
        if socks.get(control) == zmq.POLLIN:
            return

def renumerate(iterator, result, tmp_queue):

    """Recombines results with the sequence numbers stored in tmp_queue."""

    try:
        while True:
            r = [next(iterator)]
            n, l = tmp_queue.get()
            while len(r) < l:
                r.append(next(iterator))
            result.send_pyobj((n, r))
    except StopIteration:
        pass


def worker(work_port, result_port, control_port, status_port, function, args, kwargs):

    """Subprocess main. Runs a generator function on items from a pipe."""

    tmp_queue = queue.Queue()

    ctx = zmq.Context()
    work = ctx.socket(zmq.PULL)
    work.set_hwm(10)
    result = ctx.socket(zmq.PUSH)
    status = ctx.socket(zmq.PUSH)
    control = ctx.socket(zmq.SUB)

    try:
        work.connect(f'tcp://localhost:{work_port}')
        result.connect(f'tcp://localhost:{result_port}')
        status.connect(f"tcp://localhost:{status_port}")
        control.connect(f"tcp://localhost:{control_port}")
        control.setsockopt(zmq.SUBSCRIBE, b"")
        status.send_string('CON')

        renumerate(function(denumerate(work, control, tmp_queue), *args, **kwargs), result, tmp_queue)
    except KeyboardInterrupt:
        pass
    finally:
        status.send_string('DED')


class _PureGeneratorPoolMP(object):

    def __init__(self, function, processes=1, *args, **kwargs):
        self._processes = processes
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._procs = []

        # Similar to how, on Linux, putting an unpickleable object on a Queue
        # causes an uncatchable exception, passing unpickleable objects to
        # ctx.Process does the same thing on Windows. So we must check that
        # everything can be pickled before attempting to use it. Luckily this
        # is only done once.
        pickle.dumps(self._function)
        pickle.dumps(self._args)
        pickle.dumps(self._kwargs)

    def __enter__(self):
        mp_ctx = mp.get_context('spawn')

        self._ctx = zmq.Context()

        self._work = self._ctx.socket(zmq.PUSH)
        work_port = self._work.bind_to_random_port('tcp://*')

        self._result = self._ctx.socket(zmq.PULL)
        result_port = self._result.bind_to_random_port('tcp://*')

        self._status = self._ctx.socket(zmq.PULL)
        status_port = self._status.bind_to_random_port('tcp://*')

        self._control = self._ctx.socket(zmq.PUB)
        control_port = self._control.bind_to_random_port('tcp://*')

        try:

            for id in range(self._processes):
                p = mp_ctx.Process(target=worker, args=(
                    work_port, result_port, control_port, status_port,
                    self._function, self._args, self._kwargs
                ))
                self._procs.append(p)

            for p in self._procs:
                p.start()

            atexit.register(self.__exit__)

            for p in self._procs:
                s = self._status.recv_string()
                if s == 'DED':
                    raise ChildProcessError("Worker failed to start.")

        except (KeyboardInterrupt, ChildProcessError):
            self._control.send_string("DIE")
            raise

        return self

    def apply(self, iterable):
        try:
            chunksize = min(64, 1+(len(iterable)//len(self._procs)))
        except TypeError:
            chunksize = 64

        it = iter(iterable)
        iterable = enumerate(iter(lambda: list(itertools.islice(it, chunksize)), []))
        received = {}
        sent_count = 0
        received_count = 0
        done = False

        poller = zmq.Poller()
        poller.register(self._work, zmq.POLLOUT)
        poller.register(self._status, zmq.POLLIN)
        poller.register(self._result, zmq.POLLIN)

        while True:
            socks = dict(poller.poll())

            if socks.get(self._status) == zmq.POLLIN:
                raise ChildProcessError('Worker exited unexpectedly.')

            if socks.get(self._result) == zmq.POLLIN:
                n, item = self._result.recv_pyobj()
                received[n] = item

                while received_count in received:
                    yield from received[received_count]
                    del received[received_count]
                    received_count += 1
                    if sent_count - received_count < self._processes * 3:
                        poller.register(self._work, zmq.POLLOUT)

                if done and sent_count == received_count:
                    return

            if socks.get(self._work) == zmq.POLLOUT:
                try:
                    self._work.send_pyobj(next(iterable))
                    sent_count += 1
                    if sent_count - received_count > self._processes * 4:
                        poller.unregister(self._work)
                except StopIteration:
                    done = True

    def __exit__(self, *args):
        self._control.send_string("DIE")
        for proc in self._procs:
            proc.join()
        atexit.unregister(self.__exit__)


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
