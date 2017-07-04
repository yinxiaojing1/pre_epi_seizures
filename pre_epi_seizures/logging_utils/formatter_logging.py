import logging
import datetime as dt


class MyFormatter(logging.Formatter):
    converter=dt.datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
logger.addHandler(console)

formatter = MyFormatter(fmt='[%(asctime)s]:%(funcName)s:%(levelno)s:%(message)s',datefmt='%Y-%m-%d %H:%M:%S.%f')
console.setFormatter(formatter)
