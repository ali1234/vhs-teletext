import pty, os, sys, errno, fcntl, termios, subprocess, signal, atexit, time


def urxvt(name='urxvt', opts=[]):
    (master, slave) = os.openpty()

    # create a urxvt as a child so we can kill it when we're done
    pid = os.fork()
    if pid == 0:
        # restore SIGPIPE signal for the urxvt
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        os.execv('/usr/bin/urxvt', [name, '-pty-fd', str(master)] + opts)

    return pid,slave

def change_terminal((termpid,fd), take_stderr=False):
    # fork so we can change our controlling terminal
    child = os.fork()
    if child == 0:
        # become session leader
        os.setsid()
        # change controlling terminal
        fcntl.ioctl(fd, termios.TIOCSCTTY, '')
        # replace stdio
        if sys.stdin.isatty():
            os.dup2(fd, 0)
        if sys.stdout.isatty():
            os.dup2(fd, 1)
        if take_stderr:
            if sys.stderr.isatty():
                os.dup2(fd, 2)

    else:
        pid, status = os.wait()
        if pid == child:
            os.kill(termpid, 15)
            os.waitpid(termpid, 0)
        elif pid == termpid:
            os.kill(child, 15)
            os.waitpid(child, 0)
        exit(0)


def less(opts = []):

    # restore SIGPIPE for the less
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # ignore ctrl-c
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    p = subprocess.Popen(['less', '-SrX'] + opts, stdin=subprocess.PIPE)
    os.dup2(p.stdin.fileno(), 1)
    p.stdin.close()

    def wait_for_pager():
        # yes, we need both these, or it doesn't work:
        sys.stdout.flush()
        os.close(1)
        p.wait()

    atexit.register(wait_for_pager)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='pty test.')
    parser.add_argument('count', type=int, help='Print COUNT lines then exit.')
    parser.add_argument('-W', '--windowed', help='Output in a separate window.', action='store_true')
    parser.add_argument('-L', '--less', help='Page output with less.', action='store_true')

    args = parser.parse_args()

    if args.windowed:
        change_terminal(urxvt())
        if args.less:
            less()
    else:
        if args.less:
            less(['-F'])

    n = 0
    for n in xrange(args.count):
        print 'Hello. This is a test.', n
        n += 1
        time.sleep(0.01)
