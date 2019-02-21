import itertools
import queue
import time

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


def slave(function, quit_event, work_queue, done_queue, args, kwargs):
    """The main function for subprocesses. """
    tmp_queue = mp.Queue() # holds work item numbers to be recombined with the result
    try:
        renumerate(function(denumerate(quit_event, work_queue, tmp_queue), *args, **kwargs), done_queue, tmp_queue)
    except KeyboardInterrupt:
        pass


def itermap(function, iterator, processes=1, *args, **kwargs):
    """
    Implements a multiprocessing pool similar in function to multiprocessing.Pool.
    However, Pool.map(f, i) calls f on every item in i individually. f is expected
    to return the result. itermap(f, i) calls f exactly once for each process it
    starts, and then delivers an iterator containing work items. f is expected to
    yield results. In practice, this means you can pass large objects to f and they
    will only be pickled once rather than for every item in i. It also allows you
    to do one-time setup at the beginning of f.

    itermap() preserves the ordering of items in the input iterator.
    """
    if processes == 1:
        yield from function(iterator, *args, **kwargs)
    else:
        iterator = enumerate(iterator)

        ctx = mp.get_context('spawn')

        # Work items are placed on this queue by the main process.
        work_queue = ctx.Queue()
        # Sub-processes place results on this queue.
        done_queue = ctx.Queue()
        # Tells sub-processes that we are done and they should exit.
        quit_event = ctx.Event()

        pool = [ctx.Process(
            target=slave, args=(function, quit_event, work_queue, done_queue, args, kwargs)
        ) for id in range(processes)]

        try:
            for p in pool:
                p.start()

            sent_count = 0
            received_count = 0

            # Prime the queue with some items.
            for item in itertools.islice(iterator, 100):
                work_queue.put(item)
                sent_count += 1

            # Dict to use for sorting received items back into
            # their original order.
            received = {}

            while received_count < sent_count:
                n, item = done_queue.get()
                received[n] = item
                while received_count in received:
                    yield received[received_count]
                    del received[received_count]
                    received_count += 1
                try:
                    work_queue.put(next(iterator))
                    sent_count += 1
                except StopIteration:
                    quit_event.set()

        finally:
            for p in pool:
                p.join()


if __name__ in ['__main__', '__mp_main__']:

    def f(iterator, *args, **kwargs):
        # f first creates an unpickable, unsharable object. It must be done
        # exactly once per process.
        print('This line MUST be printed exactly once by each process.', args, kwargs)
        for item in iterator:
            time.sleep(0.001)
            yield item


if __name__ == '__main__':

    import click

    @click.command()
    @click.option('-j', '--jobs', type=int, default=1)
    @click.option('-t', '--threads', type=int, default=1)
    def main(jobs, threads):
        for result in itermap(f, iter(range(jobs)), processes=threads, a=2, b=3):
            print(result, end=' ')
        print('')

    main()
