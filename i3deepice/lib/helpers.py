import numpy as np
import scipy.stats as st
from icecube import dataclasses

def get_t0(pulses):
    time = []
    charge = []
    for i in pulses:
        for j in i[1]:
            charge.append(j.charge)
            time.append(j.time)
    return median(time, weights=charge)


def median(arr, weights=None):
    if weights is not None:
        weights = 1. * np.array(weights)
    else:
        weights = np.ones(len(arr))
    rv = st.rv_discrete(values=(arr, weights / weights.sum()))
    return rv.median()

def charge_after_time(charges, times, t=100):
    mask = (times - np.min(times)) < t
    return np.sum(charges[mask])


def time_of_percentage(charges, times, percentage):
    charges = charges.tolist()
    cut = np.sum(charges) / (100. / percentage)
    sum = 0
    for i in charges:
        sum = sum + i
        if sum > cut:
            tim = times[charges.index(i)]
            break
    return tim

#based on the pulses
def pulses_quantiles(charges, times, quantile):
    tot_charge = np.sum(charges)
    cut = tot_charge*quantile
    progress = 0
    for i, charge in enumerate(charges):
        progress += charge
        if progress >= cut:
            return times[i]


def nmoment(x, counts, c, n):
    return np.sum(counts*(x-c)**n) / np.sum(counts)


def normalize(time,charge):

    charge=charge[time[:]<=time[0]+400]
    time=time[time[:]<=time[0]+400]

    t=max(time-min(time))
    #np.seterr(divide='ignore', invalid='ignore')
    if(t==t and t!=0):
        #print(max(time-min(time)))
        return (time-min(time))/max(time-min(time)),charge
    else:
        return time, charge


def mean(charge,time):
    if (len(time)<4):
        return 0
    else:
        time,charge=normalize(time,charge)
        return nmoment(time,charge, 0,1)

def var(charge,time):
    if (len(time)<4):
        return 0
    else:
        return nmoment(time,charge, 0,2)

def skw(charge,time):
    return nmoment(time,charge, 0,3)

def kur(charge,time):
    return nmoment(time,charge, 0,4)

def mult(charge,time):
    if (len(time)<4):
        return 0
    else:
        time , charge = normalize(time,charge)
        kur1=kur(charge,time)

        if(kur1==kur1 and kur1!=0):
            kur2=(skw(charge,time)**2+1)/kur1
            if(kur2 == kur2):
                return kur2
            else:
                return 0
        else:
            return 0

#working
def diff(charge,time):
    if (len(time)<4):
        return 0
    else:

        time , charge = normalize(time,charge)
        diff1= mean(charge,time)-time[np.argmax(charge)]

        if (diff1 != diff1):
            return 0
        else:
            return diff1
