"""
.. module:: WebServer
   :platform: Windows, Linux
   :synopsis: WebSockets server.

.. moduleauthor:: Ana Priscila Alves, Carlos Carreiras


"""

# Imports
# built-in
# import json
import logging
import multiprocessing
from multiprocessing import queues
import Queue
import sys
import time
import threading
import traceback
import uuid
from Queue import Empty, Full

# 3rd party
from txws import WebSocketFactory
from twisted.internet import protocol, reactor
try:
    from twisted.internet import ssl
except ImportError:
    ssl = None
    SSL_OK = False
else:
    SSL_OK = True
import ujson as json
import websocket


# logger
logger = logging.getLogger('webServer')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# default timeout
default_timeout = None



class WS(protocol.Protocol):
    
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
        
        # update protocol to WSApp
        time.sleep(0.1)
        self.app.WS = self
        self.app.uuid = self.uuid
        
        # notify app
        self.app.CMDqueue.put({'SU': 'NEW_CONN', 'uuid': self.uuid})
    
    def send(self, data):
        # send data to client
        
        self.transport.write(json.dumps(data))
        
    def dataReceived(self, data):
        # data received from client
        
        # decode JSON
        try:
            data = json.loads(data)
        except Exception, e:
            print "Exception decoding JSON:", e
            data = {}
        
        # put data in queue
        self.app.CMDqueue.put(data)
    
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


class WSFactory(protocol.Factory):
    
    def __init__(self, app, durable, *args, **kwargs):
        self.app = app
        self.durable = durable
        self.args = args
        self.kwargs = kwargs
    
    def buildProtocol(self, addr):
        return WS(self.app, self.durable, *self.args, **self.kwargs)


class WSCom(threading.Thread):
    def __init__(self, go, queue):
        # run parent __init__
        super(WSCom, self).__init__()
        
        self.go = go
        self.queue = queue
        self.WS = None
    
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
                reactor.callFromThread(self.WS.send, msg)


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


class WSApp(threading.Thread):
    
    def __init__(self, queue=None):
        # run parent __init__
        super(WSApp, self).__init__()
        
        # command queue
        self.CMDqueue = Queue.Queue()
        self.timeout = default_timeout
        
        # run
        self.go = multiprocessing.Event()
        self.go.set()
        
        # WSCom
        self.WS = None
        self.com = None
        self.comQ = queue
        self.uuid = None
    
    @classmethod
    def instantiate(cls, *args, **kwargs):
        
        app = cls(*args, **kwargs)
        app.start()
        
        return app
    
    def startCom(self):
        # WSCom
        
        # stop previous com (if exists)
        self.stopCom()
        
        # com Q
        if self.comQ is None:
            self.comQ = LockQueue()
        
        # start com
        self.goCom = multiprocessing.Event()
        self.goCom.set()
        
        self.com = WSCom(self.goCom, self.comQ)
        self.com.start()
    
    def stopCom(self):
        # WSCom
        
        if self.com:
            self.goCom.clear()
            self.com.join()
            self.comQ.lock()
            self.com = None
    
    def send(self, message):
        # send message to com queue
        self.comQ.put(message)
    
    def _registerWS(self):
        # register protocol to com
        
        self.com.WS = self.WS
        self.comQ.unlock()
    
    def _callback(self, callback, *args):
        if callback:
            try:
                callback(*args)
            except Exception, e:
                logger.error(e)
                if logger.isEnabledFor(logging.DEBUG):
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)
    
    def run(self):
        # main loop
        
        # start WSCom thread
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
                except KeyError:
                    # normal message
                    self._callback(self.on_message, msg)
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
                                self._registerWS()
                                self._callback(self.on_open)
                            elif cmd == 'LOST_CONN':
                                # connection lost
                                self._callback(self.on_close)
                        else:
                            # invalid uuid, ignore
                            continue
        
        # stop WSCom thread
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


def serveWS(cls, port=9999, setupFcn=None, cleanupFcn=None, durable=True, SSL=False, *args, **kwargs):
    # websocket server
    
    print "Listening at port %s.\n" % str(port)
    
    # build protocol
    try:
        if SSL and SSL_OK:
            raise NotImplementedError, "Websocket server over SSL not yet implemented."
            # sslFactory = ssl.DefaultOpenSSLContextFactory('D:\\cert\\server.key', 'D:\\cert\\server.crt')
            # listener = reactor.listenSSL(port, WebSocketFactory(WSFactory(cls, *args, **kwargs)), sslFactory)
        else:
            listener = reactor.listenTCP(port, WebSocketFactory(WSFactory(cls, durable, *args, **kwargs)))
    except Exception:
        print "Protocol not built."
        print traceback.format_exc()
        return
    
    # run setup code before starting reactor (e.g. start browser)
    if setupFcn is not None:
        setupFcn()
    
    # run reactor
    try:
        reactor.run()
    except Exception:
        print "Reactor stopped."
        print traceback.format_exc()
        time.sleep(2)
        reactor.stop()
        time.sleep(2)
    
    # run cleanup after reactor is stopped (e.g. close browser)
    if cleanupFcn is not None:
        cleanupFcn()
    
    print "Stopped listening."


class WSClient(object):
    """
    Cloned from websocket.WebSocketApp
    The interface is like JavaScript WebSocket object.
    """
    def __init__(self, url, header=[], keep_running=True, get_mask_key=None):
        """
        url: websocket url.
        header: custom header for websocket handshake.
       keep_running: a boolean flag indicating whether the app's main loop should
         keep running, defaults to True
       get_mask_key: a callable to produce new mask keys, see the WebSocket.set_mask_key's
         docstring for more information
        """
        self.url = url
        self.header = header
        self.keep_running = keep_running
        self.get_mask_key = get_mask_key
        self.sock = None

    def send(self, data, opcode=None):
        """
        send message.
        data: message to send. If you set opcode to OPCODE_TEXT, data must be utf-8 string or unicode.
        opcode: operation code of data. default is OPCODE_TEXT.
        """
        
        if opcode is None:
            opcode = websocket.ABNF.OPCODE_TEXT
        
        # encode JSON
        data = json.dumps(data)
        
        if self.sock.send(data, opcode) == 0:
            raise websocket.WebSocketConnectionClosedException()

    def close(self):
        """
        close websocket connection.
        """
        self.keep_running = False
        self.sock.close()

    def _send_ping(self, interval):
        while True:
            for _ in range(interval):
                time.sleep(1)
                if not self.keep_running:
                    return
            self.sock.ping()

    def run_forever(self, sockopt=None, sslopt=None, ping_interval=0):
        """
        run event loop for WebSocket framework.
        This loop is infinite loop and is alive during websocket is available.
        sockopt: values for socket.setsockopt.
            sockopt must be tuple and each element is argument of sock.setscokopt.
        sslopt: ssl socket optional dict.
        ping_interval: automatically send "ping" command every specified period(second)
            if set to 0, not send automatically.
        """
        if sockopt is None:
            sockopt = []
        if sslopt is None:
            sslopt = {}
        if self.sock:
            raise websocket.WebSocketException("socket is already opened")
        thread = None

        try:
            self.sock = websocket.WebSocket(self.get_mask_key, sockopt=sockopt, sslopt=sslopt)
            self.sock.settimeout(default_timeout)
            self.sock.connect(self.url, header=self.header)
            self._callback(self.on_open)

            if ping_interval:
                thread = threading.Thread(target=self._send_ping, args=(ping_interval,))
                thread.setDaemon(True)
                thread.start()

            while self.keep_running:
                data = self.sock.recv()
                if data is None:
                    break
                # decode JSON
                data = json.loads(data)
                self._callback(self.on_message, data)
        except Exception, e:
            self._callback(self.on_error, e)
        finally:
            if thread:
                self.keep_running = False
            self.sock.close()
            self._callback(self.on_close)
            self.sock = None

    def _callback(self, callback, *args):
        if callback:
            try:
                callback(*args)
            except Exception, e:
                logger.error(e)
                if logger.isEnabledFor(logging.DEBUG):
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)
    
    @classmethod
    def instantiate(cls, url, *args, **kwargs):
        
        app = cls(url, *args, **kwargs)
        app.run_forever()
    
    def on_open(self):
        # new connection
        
        pass
    
    def on_close(self):
        # connection lost
        
        pass
    
    def on_message(self, message):
        # deal with messages
        
        pass
    
    def on_error(self, reason):
        # error in execution
        
        pass


def connectWS(cls, host=None, port=9999, SSL=False, *args, **kwargs):
    # websocket client
    
    # default host
    if host is None:
        host = "127.0.0.1"
    
    # format url
    if SSL:
        url = "wss://%s:%d"
    else:
        url = "ws://%s:%d"
    
    url = url % (host, port)
    
    print "Connecting to %s:%s.\n" % (str(host), str(port))
    
    # instantiate
    try:
        cls.instantiate(url, *args, **kwargs)
    except Exception:
        print "Client failed."
        print traceback.format_exc()
    
    print "Client disconnected."


class Echo(WSApp):
    
    def on_open(self):
        # new connection
        
        self.send({'Reply': "Hello, I'm an echo server."})
    
    def on_close(self):
        # connection lost
        
        print "Echo: bye.\n"
    
    def on_message(self, message):
        # deal with messages
        
        try:
            cmd = message['Command']
        except KeyError:
            self.send({'Reply': "No 'Command' keyword."})
            return
        
        if cmd == 'ECHO':
            # echo message
            try:
                txt = message['Text']
            except KeyError:
                txt = "Say something using the 'Text' keyword."
            # send
            self.send({'Reply': txt})


class TestClient(WSClient):
    
    def __init__(self, *args, **kwargs):
        # run parent __init__
        super(TestClient, self).__init__(*args, **kwargs)
        
        self.nb = 0
    
    def on_open(self):
        
        print "Connection established!"
    
    def on_message(self, message):
        
        try:
            reply = message['Reply']
        except KeyError:
            print "Nothing useful received..."
        else:
            print "Server sent:", reply
        
        if self.nb < 5:
            self.nb += 1
            time.sleep(3)
            self.send({'Command': 'ECHO', 'Text': 'Hello World!'})
        else:
            self.keep_running = False
        
    def on_close(self):
        
        print "Client: bye.\n"



if __name__=='__main__':
    Q = LockQueue()
    serveWS(Echo, port=9999, queue=Q)
    
