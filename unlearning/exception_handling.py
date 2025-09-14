import functools
import traceback


def print_exceptions(f):
    return functools.wraps(f)(ExceptionPrinter(f))


class ExceptionPrinter:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            traceback.print_exception(e)
            raise

    def __getattr__(self, attr):
        if "f" not in self.__dict__:
            raise AttributeError()

        return getattr(self.f, attr)
