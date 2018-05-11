"""
.. module:: server
   :platform: Unix, Windows
   :synopsis: This module implements the BITCloud server.

.. moduleauthor:: Carlos Carreiras


"""

# Notes


# Imports
# built-in
import collections
import os
import sys
import signal
import json
import smtplib
import logging
import time
from email import Encoders
from email.MIMEBase import MIMEBase
from email.MIMEMultipart import MIMEMultipart
from email.Utils import formatdate
from email.MIMEText import MIMEText
from multiprocessing import Process, Pipe
from Queue import Empty

# 3rd party
from kombu import Connection
import pymongo as pmg
import matplotlib
# change to a back-end that does not use DISPLAY (for Linux), not interactive
matplotlib.use('Agg')



class MQWorker(object):
    """
    Consumer class
    """
    
    TRANSPORT = 'amqp'
    VHOST = '/'
    PASSWD = 'worker'
    RABBIT_PORT = 5672
    MONGO_PORT = 27017
    TIMEOUT = 30 # seconds
    # "%s %d %s" % (username, taskId, machine)
    SUCCESS_MSG = """
    Dear %s,
     
    Task %d has been successfully completed.
    The results are available on the %s machine.
    
    Kind regards.
    """
    FAILURE_MSG = """
    Dear %s,
    
    Unfortunalety, Task %d has exited with errors.
    The logs are available on the %s machine.
    
    Kind regards.
    """
    REPEAT_MSG = """
    Dear %s,
    
    Task %d has already been processed elsewhere.
    Repeat caught by the %s machine.
    
    Kind regards.
    """
    # "%d" % (taskId)
    EMAIL_SUBJECT = "[BiometricsPyKit] Status of Task %d"
    EMAIL_SENDER = "carlos.carreiras@lx.it.pt"
    
    def __init__(self, rabbitHost='localhost', mongoHost='localhost', workerName=None, dbName='BiometricsExperiments',
                 queue='BiometricsQ', dstPath='~', mode='production', parameters=None):
        # create a new instance of the consumer class
        
        # get logger
        self._logger = logging.getLogger('Consumer')
        
        # database path
        self._dstPath = dstPath
        
        # processing function
        if mode == 'production':
            self._processFcn = processTask
        elif mode == 'test':
            dbName = 'biometricWorkerTest'
            self._processFcn = processTask_test
        else:
            raise ValueError, "Undefined mode %s." % str(mode)
        
        # processing process
        self._process = None
        
        # parameters for processing function
        self._processParameters = parameters
        
        # check inputs
        if workerName is None:
            self._logger.error("A worker name was not specified.")
            raise TypeError, "Please specify a worker name."
        
        # connect to RabbitMQ
        self._connection = Connection(
                                      hostname=rabbitHost,
                                      port=self.RABBIT_PORT,
                                      transport=self.TRANSPORT,
                                      userid=workerName,
                                      password=self.PASSWD,
                                      virtual_host=self.VHOST
                                      )
        self._queueName = queue
        self._name = workerName
        
        # connect to MongoDB
        self._logger.info("Connecting to MongoDB server (%s:%d).", mongoHost, self.MONGO_PORT)
        try:
            self._mongoConnection = pmg.Connection(mongoHost, self.MONGO_PORT)
        except Exception, e:
            self._logger.error("Connection to MongoDB server (%s:%d) failed.", mongoHost, self.MONGO_PORT)
            self._logger.exception(str(e))
            raise
        
        # get database
        db = self._mongoConnection[dbName]
        # get collections
        self._experiments = db['experiments']
        self._users = db['users']
    
    def declare_queue(self):
        # connect
        self._logger.info("Connecting to RabbitMQ server.")
        try:
            self._connection.connect()
        except Exception, e:
            self._logger.error("Connection to RabbitMQ server failed.")
            self._logger.exception(str(e))
            raise
        
        # declare the queue
        self._logger.info("Declaring the queue on RabbitMQ.")
        self._queue = self._connection.SimpleQueue(self._queueName)
        
        # quality of service - only get one task at a time
        self._queue.consumer.qos(prefetch_count=1)
    
    def on_message(self, message):
        # to process a task
        
        # decode the task ID
        taskID = message.payload['_id']
        self._logger.info("Received task %d." % taskID)
        
        # get task from MongoDB
        task = self._experiments.find_one({'_id': taskID})
        
        # get rid of unicode strings
        task = convertUniDict(task)
        
        # do not repeat tasks
        try:
            status = task['status']
        except KeyError:
            pass
        else:
            if status in ['finished', 'error']:
                # notify repeat, ignore task
                self._logger.info("Task %d already processed elsewhere -- ignoring task." % taskID)
                self.notify(task['user'], taskID, self.REPEAT_MSG)
                
                # acknowledge message
                self.acknowledge_message(message, taskID)
                
                return
        
        # update the database path
        try:
            task['starting_data']['srcParameters']['dstPath'] = self._dstPath
        except KeyError:
            pass
        
        # execute the task
        self._logger.info("Processing the task.")
        self._experiments.update({'_id': taskID}, {'$set': {'status': 'running'}})
        
        pc, cc = Pipe()
        self._process = Process(target=self._processFcn, args=(cc, task, taskID, self._processParameters))
        self._process.start()
        self._process.join()
        
        output = pc.recv()
        res = output['exitStatus']
        
        if res:
            # notify success
            self._logger.info("Task completed!")
            self._experiments.update({'_id': taskID}, {'$set': {'status': 'finished', 'worker': self._name, 'results': output['results']}})
            # email
            self.notify(task['user'], taskID, self.SUCCESS_MSG)
        else:
            # notify failure
            self._logger.error("Task failed, check task logs for details.")
            self._experiments.update({'_id': taskID}, {'$set': {'status': 'error', 'worker': self._name, 'results': {}}})
            # email
            self.notify(task['user'], taskID, self.FAILURE_MSG)
            
        # acknowledge message
        self.acknowledge_message(message, taskID)
    
    def notify(self, username, taskID, message):
        # email
        
        # get the email from MongoDB
        doc = self._users.find_one({'username': username}, {'email': 1})
        try:
            mail = doc['email']
        except KeyError:
            # no mail available
            self.logger.info("Notification via E-Mail unavailable (unregistered user).")
        else:
            # mail is available
            # format messages
            msg = message % (username, taskID, self._name)
            subject = self.EMAIL_SUBJECT % taskID
            # send mail
            sendEmail(mail, self.EMAIL_SENDER, Subject=subject, Text=msg)
            self._logger.info("E-Mail notification sent.")
    
    def acknowledge_message(self, message, taskID):
        # acknowledge the message
        
        self._logger.info("Acknowledging task %d.", taskID)
        message.ack()
    
    def go(self):
        # run
        
        # declare queue
        self.declare_queue()
        
        # start
        while True:
            try:
                try:
                    message = self._queue.get(timeout=self.TIMEOUT)
                except Empty:
                    continue
                else:
                    self.on_message(message)
            except KillSignal:
                break
        
        # stop
        self.stop()

    def stop(self):
        # cleanly stop
        
        self._logger.info("Terminating the worker process.")
        # kill process
        if (self._process is not None) and self._process.is_alive():
            self._process.terminate()
        
        # disconnect RabbitMQ
        self._connection.release()
        # disconnect MongoDB
        self._mongoConnection.close()


class KillSignal(Exception):
    # Exception raised when the OS sends a kill signal
    
    def __init__(self, sig):
        # set the signal code
        self.sig = sig
    
    def __str__(self):
        return str("Received the kill signal %d." % self.sig)


def setKillHandler(func):
    if os.name == 'nt':
        try:
            import win32api
        except ImportError:
            # sigterm will not be caught in windows
            print "pywin32 not installed."
        else:
            win32api.SetConsoleCtrlHandler(func, True)
    else:
        signal.signal(signal.SIGTERM, func)
    
    # in both cases, add the interrupt from keyboard
    signal.signal(signal.SIGINT, func)


def onKillSignal(sig, func=None):
    # function executed when a kill signal is caught
    
    raise KillSignal(sig)


def processTask(conn, task, taskID, parameters):
    # process a task
    
    # set up environment
    # change to a matplotlib back-end that does not use DISPLAY (for Linux), not interactive
    import matplotlib
    matplotlib.use('Agg')
    
    # biometrics modules
    sys.path.append(parameters['BioSPPy'])
    sys.path.append(parameters['BiometricsPyKit'])
    
    # biometrics configuration
    import config
    config.baseFolder = parameters.pop('experimentsFolder')
    try:
        config.numberProcesses = parameters.pop('numberProcesses')
    except KeyError:
        # the default in config
        pass
    try:
        config.queueTimeOut = parameters.pop('queueTimeOut')
    except KeyError:
        # the default in config
        pass
    
    # wizard
    import BiometricWizard as wiz
    reload(wiz)
    
    # run task
    res, output = wiz.Main(task, taskID)
    
    # send success status
    conn.send({'exitStatus': res,
               'results': output,
               })
    conn.close()
    
    return None


def processTask_test(conn, task, taskID, parameters):
    # process a task (test mode)
    
    time.sleep(5)
    
    # send success status
    conn.send({'exitStatus': True, 'results': {}})
    conn.close()
    
    return None


def sendEmail(To, From, CC='', ReplyTo='', Subject='', Text='', FilePath=None, Host='cascais.lx.it.pt', Port=25):
    # send mails
    
    msg = MIMEMultipart()
    msg["From"] = From
    msg["To"] = To
    msg["Cc"] = CC
    msg["Subject"] = Subject
    msg['Date']    = formatdate(localtime=True)
    msg.add_header('reply-to', ReplyTo)

    # attach a file
    if FilePath is not None:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(FilePath, "rb").read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(FilePath))
        msg.attach(part)
    
    # msg.attach(MIMEText(Text, 'html'))
    msg.attach(MIMEText(Text))
    
    try:
        server = smtplib.SMTP(Host, Port)
        # server.login(username, password)  # optional
    except Exception, e:
        logging.error("Unable to connect to SMTP server.")
        logging.exception(str(e))
    else:
        try:
            if CC != "":
                server.sendmail(From, [To, CC], msg.as_string())
            else:
                server.sendmail(From, To, msg.as_string())
            server.close()
            logging.info("Successfully sent mail.")
        except Exception, e:
            logging.error("Unable to send email.")
            logging.exception(str(e))


def convertUniDict(data):
    # function to convert unicode strings in a dictionary to str
    
    if isinstance(data, str):
        return data
    elif isinstance(data, unicode):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convertUniDict, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convertUniDict, data))
    else:
        return data



if __name__ == '__main__':
    # get the configuration file
    try:
        with open(sys.argv[1], 'r') as fid:
            parameters = json.load(fid)
    except IndexError:
        raise ValueError, "Please specify a configuration file as input."
    
    # set kill handler
    setKillHandler(onKillSignal)
    
    # biometrics modules
    sys.path.append(parameters['parameters']['BioSPPy'])
    sys.path.append(parameters['parameters']['BiometricsPyKit'])
    
    # configure logging
    from misc import misc
    logger = misc.getLogger('Consumer', parameters.pop('logPath'), 'info')
    
    # run the worker
    logger.info("Starting the worker process.")
    worker = MQWorker(**parameters)
    worker.go()
    

