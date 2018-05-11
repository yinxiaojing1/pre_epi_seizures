"""
.. module:: WebServer
   :platform: Windows, Linux
   :synopsis: WebSockets server.

.. moduleauthor:: Ana Priscila Alves, Carlos Carreiras


"""

# Imports
# built-in
# import json
# import logging
import multiprocessing
from multiprocessing import queues
import Queue
# import sys
import time
import threading
import traceback
import uuid
from Queue import Empty, Full

# 3rd party
from twisted.internet import error, protocol, reactor
from twisted.internet.serialport import SerialPort
# from twisted.protocols import basic
# import ujson as json


# default timeout
default_timeout = None


class SP(protocol.Protocol):
    
    def __init__(self, app, durable, *args, **kwargs):
        self.appClass = app
        self.durable = durable
        self.args = args
        self.kwargs = kwargs
        self.uuid = uuid.uuid4()
    
    def connectionMade(self):
        # start ProcessECG process
        print "New connection accepted!\n"
        
        # start app
        try:
            self.app = self.appClass.instantiate(*self.args, **self.kwargs)
        except Exception:
            print "App did not start."
            print traceback.format_exc()
            self.transport.loseConnection()
            return
        
        # update protocol to SPApp
        time.sleep(0.1)
        self.app.SP = self
        self.app.uuid = self.uuid
        
        # notify app
        self.app.CMDqueue.put({'SU': 'NEW_CONN', 'uuid': self.uuid})
    
    def send(self, data):
        # send data to client
        
        self.transport.write(data)
    
    def dataReceived(self, data):
        # data received from client
        
        self.app.CMDqueue.put(bytearray(data))
    
    def connectionLost(self, reason):
        # stop ProcessECG process, disconnect
        print "Connection LOST!\n"
        
        # notify app
        self.app.CMDqueue.put({'SU': 'LOST_CONN', 'uuid': self.uuid})
        self.app.go.clear()
        
        if not self.durable:
            # stop server
            reactor.stop()
        
        return


class SPCom(threading.Thread):
    def __init__(self, go, queue):
        # run parent __init__
        super(SPCom, self).__init__()
        
        self.go = go
        self.queue = queue
        self.SP = None
    
    def run(self):
        # main loop
        
        while self.go.is_set():
            try:
                # get new message
                msg = self.queue.get(timeout=1)
            except Empty:
                continue
            else:
                # send message
                reactor.callFromThread(self.SP.send, msg)


class LockQueue(queues.Queue):
    
    def __init__(self, cleanup=True, *args, **kwargs):
        # run parent init
        super(LockQueue, self).__init__(*args, **kwargs)
        
        # lock
        self._gplock = multiprocessing.Event()
        self._gplock.set()
        
        # cleanup flag
        self.cleanup = cleanup
    
    def __getstate__(self):
        # to pickle
        
        return self._gplock, self.cleanup, super(LockQueue, self).__getstate__()
    
    def __setstate__(self, state):
        # to unpickle
        
        self._gplock, self.cleanup, state = state
        super(LockQueue, self).__setstate__(state)
    
    def lock(self):
        # lock the queue: put raises Full, get raises Empty
        
        self._gplock.set()
        
        if self.cleanup:
            # remove items from queue
            while True:
                try:
                    _ = super(LockQueue, self).get(timeout=0.5)
                except Empty:
                    break
    
    def unlock(self):
        # unlock the queue: normal queue behavior
        
        self._gplock.clear()
    
    def get(self, *args, **kwargs):
        # get from queue, but check lock
        
        # check lock
        if self._gplock.is_set():
            raise Empty
        
        # original behavior
        return super(LockQueue, self).get(*args, **kwargs)
    
    def put(self, *args, **kwargs):
        # put to queue, but check lock
        
        # check lock
        if self._gplock.is_set():
            raise Full
        
        # original behavior
        return super(LockQueue, self).put(*args, **kwargs)

class DummyComQueue(LockQueue):
    
    def put(self, item):
        
        print item
    
    def get(self, *args, **kwargs):
        
        raise Empty

class SPApp(threading.Thread):
    
    def __init__(self, queue=None):
        # run parent __init__

        super(SPApp, self).__init__()
        
        # command queue
        self.CMDqueue = Queue.Queue()
        self.timeout = default_timeout
        
        # run
        self.go = multiprocessing.Event()
        self.go.set()
        
        # SPCom
        self.SP = None
        self.com = None
        self.comQ = queue
        self.uuid = None
    
    @classmethod
    def instantiate(cls, *args, **kwargs):
        
        app = cls(*args, **kwargs)
        app.start()
        
        return app
    
    def startCom(self):
        # SPCom
        
        # stop previous com (if exists)
        self.stopCom()
        
        # com Q
        if self.comQ is None:
            self.comQ = LockQueue()
        
        # start com
        self.goCom = multiprocessing.Event()
        self.goCom.set()
        
        self.com = SPCom(self.goCom, self.comQ)
        self.com.start()
    
    def stopCom(self):
        # SPCom
        
        if self.com:
            self.goCom.clear()
            self.com.join()
            self.comQ.lock()
            self.com = None
    
    def send(self, message):
        # send message to com queue
        self.comQ.put(message)
    
    def _registerSP(self):
        # register protocol to com
        
        self.com.SP = self.SP
        self.comQ.unlock()
    
    def _callback(self, callback, *args):
        if callback:
            try:
                callback(*args)
            except Exception:
                print traceback.format_exc()
                self.stopReactor()
    
    def stopReactor(self):
        # stop reactor
        reactor.callFromThread(reactor.stop)
    
    def restartReactor(self):
        # restart
        
        reactor.callFromThread(reactor.fireSystemEvent, 'user_restart')
        self.stopReactor()
    
    def run(self):
        # main loop
        
        # start SPCom thread
        self.startCom()
        
        # aux
        qFails = 0
        
        while self.go.is_set():
            try:
                # get new message
                msg = self.CMDqueue.get(timeout=self.timeout)
            except Queue.Empty:
                qFails += 1
                if qFails >= 5:
                    print "Empty CMDQueue."
                    break
            else:
                # acknowledge message
                self.CMDqueue.task_done()
                qFails = 0
                
                if not msg:
                    continue
                # SU message?
                try:
                    cmd = msg['SU']
                    print "Connection Made"
                except:
                    # print traceback.format_exc() 
                    self._callback(self.on_message, msg)
                    # continue
                # except KeyError:
                    # normal message
                    # self._callback(self.on_message, msg)
                else:
                    # uuid check
                    try:
                        uid = msg['uuid']
                    except KeyError:
                        # no uuid, ignore
                        continue
                    else:
                        if uid == self.uuid:
                            if cmd == 'NEW_CONN':
                                # new connection
                                self._registerSP()
                                self._callback(self.on_open)
                            elif cmd == 'LOST_CONN':
                                # connection lost
                                self._callback(self.on_close)
                        else:
                            # invalid uuid, ignore
                            continue
        
        # stop SPCom thread
        self.stopCom()
    
    def on_open(self):
        # new connection
        
        pass
    
    def on_close(self):
        # connection lost
        
        pass
    
    def on_message(self, message):
        # deal with messages
        
        pass

class Echo(SPApp):
    
    def on_open(self):
        # new connection
        
        #self.send()
        print "on_open"
    
    def on_close(self):
        # connection lost
        
        print "Echo: bye.\n"
    
    def on_message(self, message):
        # deal with messages
        
        try:
            #cmd = message['Command']
            print "ON_MESSAGE 0:",message[0]
            print "ON_MESSAGE 1:", message[1]
        except KeyError:
            #self.send({'Reply': "No 'Command' keyword."})
            return

def serverPort(cls, port="COM30", baud=9600, setupFcn=None, cleanupFcn=None, durable=True, *args, **kwargs):
    # websocket server
    
    print "Listening at port %s.\n" % str(port)
    
    # build protocol
    try:
        listener = SerialPort(SP(cls, durable, *args, **kwargs), port, reactor, baudrate=baud)
    except Exception:
        print "Protocol not built."
        print traceback.format_exc()
        return
    
    # register system event callbacks
    if setupFcn is not None:
        reactor.addSystemEventTrigger('after', 'startup', setupFcn)
    if cleanupFcn is not None:
        reactor.addSystemEventTrigger('before', 'shutdown', cleanupFcn)
    
    def handle_error(d):
        d['status'] = 'error'
    
    def handle_restart(d):
        d['status'] = 'restart'
    
    appMem = {'status': 'stop'}
    reactor.addSystemEventTrigger('after', 'user_restart', handle_restart, appMem)
    
    # run reactor
    try:
        reactor.run()
    except Exception:
        print traceback.format_exc()
        handle_error(appMem)
        try:
            reactor.stop()
        except error.ReactorNotRunning:
            pass
    
    print "Stopped listening."
    
    return appMem['status']



if __name__=='__main__':
    Q = LockQueue()
    serverPort(Echo(), port="COM30", queue=Q)
    
