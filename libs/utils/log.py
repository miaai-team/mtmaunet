def make_print_to_file(path='./',fileName=None):
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            # self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    if not fileName:
        fileName = datetime.datetime.now().strftime('log_' + '%Y_%m_%d_%Hh_%Mmin')
    sys.stdout = Logger(fileName + '.txt', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, ' '))