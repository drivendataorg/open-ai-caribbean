from multiprocessing import Pool, cpu_count


def multithreaded_map(function, arg_list, n_threads=None):
    n_threads = n_threads if n_threads is not None else cpu_count()
    pool = Pool(processes=n_threads)
    result = pool.map(function, arg_list)
    pool.close()
    pool.join()

    return result


def multithreaded_starmap(function, arg_list, n_threads=None):
    n_threads = n_threads if n_threads is not None else cpu_count()
    pool = Pool(processes=n_threads)
    result = pool.starmap(function, arg_list)
    pool.close()
    pool.join()

    return result


# https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def htime(c):
    # https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/general.py
    c = round(c)

    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = c % 60

    if days > 0:
        return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
    if hours > 0:
        return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
    if minutes > 0:
        return '{:d}m {:d}s'.format(minutes, seconds)
    return '{:d}s'.format(seconds)


class RunningAverage(object):
    def __init__(self):
        self.samples = 0
        self.running_sum = 0

    def update(self, samples, average):
        self.samples += samples
        self.running_sum += samples * average

    def average(self):
        return self.running_sum / self.samples
