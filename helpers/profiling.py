import time
import sys


class Profiler:
    def __init__(self, enabled=True):
        self.profile = {}
        self.enabled = enabled
        self.out = sys.stdout

    def start(self, fname):
        if fname not in self.profile:
            self.profile[fname] = {
                'current': None,
                'total': 0,
                'calls': 0
            }
        self.profile[fname]['current'] = time.time()
        self.profile[fname]['calls'] += 1

    def finish(self, fname):
        self.profile[fname]['total'] += time.time() - self.profile[fname]['current']

    def print_profile(self):
        self.out.write("=" * 85 + "\n")
        self.out.write("=" + " " * 35 + "Time Profiles" + " " * 35 + "=\n")
        self.out.write("=" * 85 + "\n")
        self.out.write(f"{'Function name':40s} {'# of calls':15s} {'time per call':15s} {'total time':15s}\n")
        self.out.write("-" * 85 + "\n")
        for k in self.profile:
            self.out.write(f"{k:40s} {self.profile[k]['calls']:<15d} ")
            self.out.write(f"{self.profile[k]['total']/self.profile[k]['calls']:<15.2f} ")
            self.out.write(f"{self.profile[k]['total']:<15.2f}\n")
        self.out.write("=" * 85 + "\n")


global_profiler = Profiler(True)


def profile(func, profiler=global_profiler):
    """Wraps specified functions of an object with start and finish"""
    if not profiler.enabled:
        return func

    def wrap(*args, **kwargs):
        profiler.start(func.__name__)
        result = func(*args, **kwargs)
        profiler.finish(func.__name__)
        return result

    wrap.__name__ = func.__name__
    return wrap
