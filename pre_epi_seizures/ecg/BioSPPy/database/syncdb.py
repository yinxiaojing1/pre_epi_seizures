"""
.. module:: syncdb
   :platform: Unix, Windows
   :synopsis: Unison synchronization for BioMESH DB

.. moduleauthor:: Carlos Carreiras
"""


import os
import subprocess
import threading
import Queue
import socket



def altSyncNow(queue, user, host, port, dbName):
    # new experimental sync
    
    # connect to the remote server (now it'll log to a file)
    logPath = os.path.abspath(os.path.expanduser('~/altSync.log'))
    fid = open(logPath, 'a')
    fid.write("\n--SYNC START--\n")
    fid.write("User: %s\n" % str(user))
    fid.write("Host: %s\n" % str(host))
    fid.write("Port: %s\n" % str(port))
    fid.write("Database: %s\n\n" % str(dbName))
    
    # wait for changes
    while True:
        try:
            cmd = queue.get()
        except Queue.Empty:
            pass
        else:
            if cmd['function'] == 'SIGTERM':
                fid.write("--SIGTERM--\n")
                fid.close()
                break
            else:
                fid.write(cmd['function'] + "\n")


def altSyncServer():
    
    HOST = '127.0.0.1'
    PORT = 50007
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print 'Connected by', addr
    while 1:
        data = conn.recv(1024)
        if not data: break
        conn.sendall(data)
    conn.close()


def sync(source=None, destination=None, queue=None, logPath='~/sync.log'):
    """
    
    Generic synchronization method between two folders.
    
    Kwargs:
        source (str): Path at the source.
        
        destination (str): Path at the destination.
        
        queue (Queue): Queue to communicate with the thread.
        
        logPath (str): Path for the log file. Default: '~/sync.log'
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        At the source folder, no deletions are allowed, and backup files (.old) are ignored.
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # check inputs
    if source is None:
        raise TypeError, "A source (server) path must be provided."
    if destination is None:
        raise TypeError, "A destination (client) path must be provided."
    
    # write start to log
    logPath = os.path.abspath(os.path.expanduser(logPath))
    with open(logPath, 'a') as f:
        f.write("\n--SYNC START--\n")
    
    # run unison
    thread = run('unison -log=true -logfile=' + logPath + ' -batch -confirmbigdel=false -ignorelocks -ignore="Name {,.}*{.old}" -nodeletion ' + source + ' ' + source + ' ' + destination, queue)
    
    return thread
    
    
def syncDB(dbName=None, host=None, dstPath=None, srvPath=None, userName='biomesh'):
    """
    
    Method to synchronize a database with the remote server.
    
    Kwargs:
        dbName (str): Name of the database.
        
        host (str): Address of the remoter server, e.g. '193.136.222.234'.
        
        dstPath (str): Local path to synchronize.
        
        srvPath (str): Remote path to synchronize.
        
        userName (str): User name at the remote server.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # check inputs
    if dbName is None:
        raise TypeError, "A DB name must be provided."
    if host is None:
        raise TypeError, "A host address must be provided."
    if dstPath is None:
        raise TypeError, "A local path for the file must be provided."
    if srvPath is None:
        raise TypeError, "A remote path for the file must be provided."
    
    # server
    source = "ssh://" + userName + "@" + host + srvPath + '/' + dbName
    
    # client
    destination = os.path.abspath(dstPath)
    
    # sync
    queue = Queue.Queue()
    thread = sync(source, destination, queue)
    
    return thread, queue



def run(command, queue):
    """
    
    Run a shell command in a separate thread.
    
    Kwargs:
        command (str): The command to execute.
        
        queue (Queue): Queue to communicate with the thread.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    thread = threading.Thread(target = __run, args=(command, queue))
    thread.start()
    
    return thread


def __run(command, queue):
    """
    
    Runs the shell command with subprocess.
    
    Kwargs:
        command (str): The command to execute.
        
        queue (Queue): Queue to communicate with the thread.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    p = subprocess.Popen(unicode(command).encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
    # outputs, errors
    stdout, stderr = p.communicate()
    
    # return code
    ret = p.returncode
    
    # send success to queue
    if ret == 0:
        queue.put({'sync': True})
    else:
        queue.put({'sync': False, 'stdout': stdout, 'stderr': stderr, 'ret': ret})
    
    return None


if __name__ == '__main__':
    
    # create directory for tests
    path = os.path.abspath(os.path.expanduser('~/tmp/syncdb'))
    if not os.path.exists(path):
        os.makedirs(path)
    
    # synchronize
    thread, queue = syncDB('storageTest', '193.136.222.234', path, '/BioMESH')
    
    print "Synchronizing with remote server."
    thread.join(2)
    if thread.is_alive():
        print "This may take some time.\nPlease wait..."
        thread.join()
        out = queue.get()
        if not out['sync']:
            print "Synchronization failed: check ssh configuration."
