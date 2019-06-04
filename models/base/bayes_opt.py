from bayes_opt import BayesianOptimization

from bayes_opt import JSONLogger
from bayes_opt import Events
from bayes_opt.util import load_logs
from bayes_opt import ScreenLogger
from bayes_opt.util import Colours

def _step(self, instance, colour=Colours.black):
        res = instance.res[-1]
        cells = []

        cells.append(self._format_number(self._iterations + 1, 'iter'))
        cells.append(self._format_key(str(timedelta(seconds=int(self._time_metrics()[2]))), "time delta"))
        cells.append(self._format_number(abs(res["target"]), 'target'))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key], key))

        return "| " + " | ".join(map(colour, cells)) + " |"
    
def _header(self, instance):
        cells = []
        cells.append(self._format_key("iter"))
        cells.append(self._format_key("time delta"))
        cells.append(self._format_key("target"))
        for key in instance.space.keys:
            cells.append(self._format_key(key))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)
    
def _format_number(self, x, key):
        if key.startswith('n_hidden'):
          x = int(x)

        if isinstance(x, int):
                s = "{x:< {s}}".format(
                    x=x,
                    s=len(key),
                )
        else:
            s = "{x:< {s}.{p}}".format(
                x=x,
                s=len(key),
                p=self._default_precision,
            )

        if len(s) > len(key):
            if "." in s:
                return s[:len(key)]
            else:
                return s[:len(key) - 3] + "..."
        return s

def _format_key(self, key, key2=None):
        l = len(key2) if key2 else len(key)
        s = "{key:^{s}}".format(
            key=key,
            s=l
        )
        return s
    
ScreenLogger._step = _step
ScreenLogger._header = _header
ScreenLogger._format_number = _format_number
ScreenLogger._format_key = _format_key