import time
import datetime
import os


class Logger(object):
    """Class to log outputs"""

    def __init__(self, filename, orig_stdout):
        self.filename = filename
        self.terminal = orig_stdout
        self.logfile = open(self.filename, "w")
        self.logfile.close()
        self.log = False
        self.verbose = True
        self.vnext = False
        self.linebuf = ''
        self.buff = []

        self.isatty = self.terminal.isatty  # Simulate sys.stdout terminal (needed for Keras dynamic print)

    def openlog(self):
        if not self.log:
            self.log = True

    def write(self, message):
        if self.verbose:
            self.terminal.write(message)
        if (self.vnext & (not self.verbose)):
            self.terminal.write(message)
            self.vnext = False
        if self.log:
            self.linebuf = ''

            for c in message:
                if c == '\r':
                    self.linebuf = ''
                    readFile = open(self.filename)
                    lines = readFile.readlines()
                    readFile.close()
                    w = open(self.filename, 'w')
                    nbdel = 1
                    w.writelines([item for item in lines[:-nbdel]])
                    w.close()
                else:
                    self.linebuf += c

            self.logfile = open(self.filename, "a")
            self.logfile.write(self.linebuf)
            self.logfile.close()

    def closelog(self):
        self.log = False

    def flush(self):
        pass


def init_logger(init_sys, filename, log=True, verbose=True, timestamp=True):
    """Initialize logger

    Parameters
    ----------
    :param init_sys: object, *sys* module
    :param filename: str, complete filename and path to save the log
    :param log: Boolean, default *True*. Whether log the outputs or not.
    :param verbose: Boolean, default *True*. Whether or not send outputs to the terminal.
    :param timestamp:Boolean, default *True*. Whether or not add timestamp to filename.
    """
    if timestamp:
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
        splitext = os.path.splitext(filename)
        filename = splitext[0] + '_' + st + splitext[1]
    directory = os.path.split(filename)[0]
    if not os.path.isdir(directory) and directory != "":
        os.mkdir(directory)
    orig_stdout = init_sys.stdout
    orig_stderr = init_sys.stderr
    init_sys.stdout = Logger(filename, orig_stdout)
    init_sys.stdout.log = log
    init_sys.stdout.verbose = verbose
    init_sys.stderr = init_sys.stdout

    return orig_stdout, orig_stderr, init_sys.stdout, init_sys.stderr
