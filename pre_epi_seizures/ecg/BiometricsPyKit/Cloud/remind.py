"""
.. module:: remind
   :platform: Unix, Windows
   :synopsis: This module implements reminder tools for Vitalidi, prompting the users to regularly check in to the system.

.. moduleauthor:: Carlos Carreiras


"""

# Imports
# built in
import datetime
import itertools
import time
import traceback
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.Utils import formatdate
from email.MIMEText import MIMEText

# 3rd party
import numpy as np
import pymongo as pmg


# E-Mail settings
SMTP_HOST = 'cascais.lx.it.pt'
SMTP_PORT = 25
REPLY = 'carlos.carreiras@lx.it.pt'
FROM = 'reminder@vitalidi.com'
SUBJECT = 'Vitalidi Reminder'

# E-Mail messages
MSG = """
Dear %s,

You haven't checked in on the Vitalidi Demonstrator for a while.
%s

Come check you heart signal at the Vitalidi Demonstrator!

Hoping to see you soon,
Your Vitalidi Demonstrator

PS: You can snooze these messages at the demonstrator.
"""



class MongoReminder(object):
    """
    Class to manage the reminders database. The databse stores the enrolled
    users, the biometric operations log, and snooze flags (TTL documents).
    """
    
    MG_HOST = '193.136.222.234'
    
    def __init__(self):
        """
        
        Set up connection to the reminders database.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # connect to mongodb
        self._connection = pmg.Connection(self.MG_HOST, 27017)
        self._db = self._connection['BiometricReminders']
        
        # ensure collections and indexes
        self._logs = self._db['Logs']
        self._logs.ensure_index([('user', 1), ('date', -1)])
        
        self._users = self._db['Users']
        self._users.ensure_index('user')
        
        self._snoozers = self._db['Snoozers']
        self._snoozers.ensure_index('ttl_date', expireAfterSeconds=0)
        self._snoozers.ensure_index('user')
        
        self._messages = self._db['messages']
    
    def __enter__(self):
        """
        
        __enter__ Method for 'with' statement.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        return self
    
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        
        __exit__ Method for 'with' statement.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self.close()
    
    def close(self):
        """
        
        Close the connection to the reminders database.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        self._connection.close()
    
    def addUser(self, name=None, mail=None):
        """
        
        Add a new user to the database.
        
        Kwargs:
            name (string): User name.
            
            mail (string): The user's email.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if name is None:
            raise TypeError, "Please specify the user name."
        if mail is None:
            raise TypeError, "Please specify the user email."
        
        # check if user already exists
        doc = self._users.find_one({'user': name}, {'_id': 1})
        try:
            self._users.update({'_id': doc['_id']}, {'$set': {'email': mail}})
        except TypeError:
            self._users.insert({'user': name, 'email': mail})
    
    def removeUser(self, name=None):
        """
        
        Remove a user from the database.
        
        Kwargs:
            name (string): User name.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if name is None:
            raise TypeError, "Please specify the user name."
        
        self._users.remove({'user': name}, multi=False)
    
    def getUser(self, name=None, stats=False):
        """
        
        Get the information regarding a user.
        
        Kwargs:
            name (string): User name.
            
            stats (bool): If True, compute compute the accuracy of the biometric system for the user.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        doc = self._users.find_one({'user': name})
        
        if stats and doc is not None:
            # authentication
            TA, FA = 0., 0.
            for log in self.iterLogs(name=name, command='auth', limit=100):
                if log['correct']:
                    TA += 1
                else:
                    FA += 1
            authErr = FA / (TA + FA) if FA > 0 else 0.
            # identification
            TI, FI = 0., 0.
            for log in self.iterLogs(name=name, command='id', limit=100):
                if log['correct']:
                    TI += 1
                else:
                    FI += 1
            idErr = FI / (TI + FI) if FI > 0 else 0.
            
            doc['stats'] = {'authentication': authErr,
                            'identification': idErr,
                            }
        
        return doc
    
    def listUsers(self):
        """
        
        Get a list of the registered users.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        return sorted([item for item in self._users.find(fields={'_id': 0, 'user': 1, 'email': 1})], key=lambda x: x['user'])
    
    def log(self, name=None, date=None, command=None, **kwargs):
        """
        
        Log a biometric operation.
        
        Kwargs:
            name (string): User name.
            
            date (datetime object): The date of the operation.
            
            command (string): The biometric operation: 'enroll', 'auth', or 'id'.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if name is None:
            raise TypeError, "Please specify the user name."
        if date is None:
            date = datetime.datetime.utcnow()
        if command is None:
            raise TypeError, "Please specify the biometric command."
        if command not in ['enroll', 'auth', 'id']:
            raise ValueError, "Invalid biometric command; must be one of 'enroll', 'auth', or 'id'."
        
        # new log
        doc = {'user': name,
               'date': date,
               'command': command,
               }
        doc.update(kwargs)
        
        # add to mongo
        self._logs.insert(doc)
    
    def getLastLog(self, name=None, command=None):
        """
        
        Get the last log.
        
        Kwargs:
            name (string): User name.
            
            command (string): The biometric operation: 'enroll', 'auth', or 'id'.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        spec = {}
        if name is not None:
            spec['user'] = name
        
        if command is not None:
            if command not in ['enroll', 'auth', 'id']:
                raise ValueError, "Invalid biometric command; must be one of 'enroll', 'auth', or 'id'."
            spec['command'] = command
        
        return self._logs.find_one(spec, sort=[('date', -1)])
    
    def iterLogs(self, name=None, command=None, limit=0):
        """
        
        Get an iterator for the logs of a user.
        
        Kwargs:
            name (string): User name.
            
            command (string): The biometric operation: 'enroll', 'auth', or 'id'.
            
            limit (int): The maximum number of logs to return; if set to 0 (the default), returns all logs.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        spec = {}
        if name is not None:
            spec['user'] = name
        
        if command is not None:
            if command not in ['enroll', 'auth', 'id']:
                raise ValueError, "Invalid biometric command; must be one of 'enroll', 'auth', or 'id'."
            spec['command'] = command
            
        return self._logs.find(spec, limit=limit, sort=[('date', -1)])
    
    def snooze(self, name=None, delay=48):
        """
        
        Snooze the reminders for a user.
        
        Kwargs:
            name (string): User name.
            
            delay (float, int): The amount of time to snooze (hours).
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if name is None:
            raise TypeError, "Please specify the user name."
        
        # convert delay to seconds
        delay = datetime.timedelta(seconds=3600*delay)
        ttl = datetime.datetime.utcnow() + delay
        
        # check if user already exists
        doc = self._snoozers.find_one({'user': name}, {'_id': 1})
        try:
            self._snoozers.update({'_id': doc['_id']}, {'$set': {'ttl_date': ttl}})
        except TypeError:
            self._snoozers.insert({'user': name, 'ttl_date': ttl})
    
    def unsnooze(self, name=None):
        """
        
        Cancel the reminder snooze for a user.
        
        Kwargs:
            name (string): User name.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if name is None:
            raise TypeError, "Please specify the user name."
        
        self._snoozers.remove({'user': name}, multi=False)
    
    def isSnoozing(self, name=None):
        """
        
        Determine if a user is snoozing.
        
        Kwargs:
            name (string): User name.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if name is None:
            raise TypeError, "Please specify the user name."
        
        return bool(self._snoozers.find_one({'user': name}, {'_id': 1}))
    
    def listReminders(self, threshold=24):
        """
        
        List the users to remind if last log ('auth' or 'id') occurred more
        than the time specified by threshold.
        
        Kwargs:
            threshold (float, int): The reminder threshold (hours).
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # convert threshold to seconds
        threshold = datetime.timedelta(seconds=3600*threshold)
        ct = datetime.datetime.utcnow()
        
        # filter with snooze
        candidates = [item for item in self.listUsers() if not self.isSnoozing(item['user'])]
        
        
        # filter with threshold
        reminders = []
        for item in candidates:
            # get last logs
            doc = self.getLastLog(name=item['user'])
            try:
                cmd = doc['command']
                date = doc['date']
            except (TypeError, KeyError):
                pass
            else:
                if cmd in ['auth', 'id'] and (ct - date) <= threshold:
                    continue
            
            reminders.append(item)
        
        return reminders
    
    def addMessage(self, message=None):
        """
        
        Add a new message.
        
        Kwargs:
            message (str): The message to add.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        # check inputs
        if message is None:
            raise TypeError, "Please specify the message to add."
        
        self._messages.insert({'message': message})
    
    def removeMessage(self, _id=None):
        """
        
        Delete a message by its id.
        
        Kwargs:
            _id (int): The id of the message to remove.
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        if _id is None:
            raise TypeError, "Please specify the _id of the message to remove."
        
        self._messages.remove({'_id': _id})
    
    def getRandomMessage(self):
        """
        
        Retrieve a random message.
        
        Kwargs:
            
        
        Kwrvals:
            
        
        See Also:
            
        
        Notes:
            
        
        Example:
            
        
        References:
            .. [1]
            
        """
        
        docs = self._messages.find({}, {'_id': 1})
        
        nb = docs.count()
        if nb <= 0:
            raise ValueError, "No messages available."
        
        for _ in xrange(np.random.randint(1, nb + 1, 1)[0]):
            doc = docs.next()
        
        message = self._messages.find_one({'_id': doc['_id']}, {'message': 1})['message']
        
        return message


def remindCycle(hours=None, ignoreWeekend=True):
    """
    
    Reminder daemon. Sends out e-mail reminders at specified times.
    
    Kwargs:
        hours (list): Reminder times.
        
        ignoreWeekend (bool): When set to true, ignore the reminders that occur on weekends. Default: True.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # check inputs
    if hours is None:
        hours = [12, 18]
    
    # local time
    ct = datetime.datetime.now()
    hours = [datetime.datetime(ct.year, ct.month, ct.day, hour=item) for item in hours]
    
    # general delays
    aux = [hours[-1] - datetime.timedelta(days=1)] + hours
    delays = itertools.cycle([item.total_seconds() for item in np.diff(aux)])
    
    # first delay
    for h in hours:
        _ = delays.next()
        if h > ct:
            break
    else:
        # for never breaks
        _ = delays.next()
        h = hours[0] + datetime.timedelta(days=1)
    
    d = h - ct
    d = abs(d.total_seconds())
    
    while True:
        try:
            time.sleep(d)
            th = d / 7200. # half the time, in hours
            
            ct = datetime.datetime.now()
            if not (ignoreWeekend and ct.isoweekday() in [6, 7]):
                with MongoReminder() as db:
                    for user in db.listReminders(threshold=th):
                        try:
                            txt = db.getRandomMessage()
                        except ValueError:
                            txt = ''
                        message = MSG % (user['user'], txt)
                        sendMail(user['email'], message)
            
            d = delays.next()
        except Exception:
            traceback.print_exc()
            break


def sendMail(mail=None, txt=None):
    """
    
    Send a reminder e-mail.
    
    Kwargs:
        mail (string): The destination e-mail address.
        
        txt (string): The mail message.
    
    Kwrvals:
        
    
    See Also:
        
    
    Notes:
        
    
    Example:
        
    
    References:
        .. [1]
        
    """
    
    # check inputs
    if mail is None:
        raise TypeError, "Please specify the e-mail address."
    if txt is None:
        raise TypeError, "Please specify the mail message."
    
    msg = MIMEMultipart()
    msg["From"] = FROM
    msg["To"] = mail
    msg["Subject"] = SUBJECT
    msg['Date'] = formatdate(localtime=True)
    msg.add_header('reply-to', REPLY)
    msg.attach(MIMEText(txt))
    
    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
    except Exception:
        traceback.print_exc()
    else:
        try:
            server.sendmail(FROM, mail, msg.as_string())
        except Exception:
            traceback.print_exc()
        server.close()



if __name__ == '__main__':
    remindCycle(hours=[11, 13, 15, 17])
    
