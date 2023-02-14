import pathlib

from watchdog.events import FileModifiedEvent, FileDeletedEvent
from watchdog.observers import Observer

from .service import Service
from .subpage import Subpage


class ServiceDir(Service):
    """
    Implements a service backed by a directory of t42 files.

    The files should be organized by page number with one subpage
    per file, like this: 100/0000.t42

    Whenever a file is modified it will be reloaded into the
    service for broadcast in the next loop of the magazine.
    """
    def __init__(self, directory):
        super().__init__()
        self._dir = directory

    def file_changed(self, f, deleted=False):
        try:
            m = int(f.parent.name[0])
            p = int(f.parent.name[1:], 16)
            s = int(f.stem, 16)
        except ValueError:
            pass
        else:
            if deleted:
                del self.magazines[m].pages[p].subpages[s]
            else:
                self.magazines[m].pages[p].subpages[s] = Subpage.from_file(f.open('rb'))

    def __enter__(self):
        self.observer = Observer()
        self.observer.schedule(self, self._dir, recursive=True)
        self.observer.start()

        # perform initial scan of the pages
        path = pathlib.Path(self._dir)
        for f in path.rglob("*"):
            if f.is_file():
                self.file_changed(f)

        return self

    def __exit__(self, *args, **kwargs):
        self.observer.stop()
        self.observer.join()

    def dispatch(self, evt):
        f = pathlib.Path(evt.src_path)
        if isinstance(evt, FileModifiedEvent):
            self.file_changed(f)
        elif isinstance(evt, FileDeletedEvent):
            self.file_changed(f, deleted=True)

