import time

clocks=dict()

def tic(id='.'):
    clocks[id]=0
    clocks[id]=time.time()
    return clocks[id]


def tac(id='.'):
    try:
        return (time.time()-clocks[id])
    except:
        return -1


def clc(id=''):
    if len(id)==0: clocks.clear()
    else: clocks.pop(id,0)