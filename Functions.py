import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

import pandas as pd
import csv
from scipy import signal


import requests

import math
from datetime import datetime, date, timedelta
from scipy.signal import find_peaks
from matplotlib import gridspec
from matplotlib.pyplot import MultipleLocator
from scipy import ndimage

def get_data(year):
    """Retrieves data from Cassini's Saturn orbits for a given year."""
    #Filename
    filename = str(year) + "_FGM_KSM_1M.TAB"

    #Retrieve data from csv file in pandas dataframe
    df = pd.read_csv(filename, infer_datetime_format=True, sep='\s+', names=["Time","BX","BY","BZ","Btotal","X","Y",'Z','LocalHour','NPTS'], engine='python')
    return df


def get_data2(year):
    """Retrieves data from Cassini's Saturn orbits for a given year."""
    #Filename
    filename = str(year) + "_FGM_KSM_1M.TAB"

    #Retrieve data from csv file in pandas dataframe
    data = pd.read_csv(filename, infer_datetime_format=True, sep='\s+', names=["Time","BX","BY","BZ","Btotal","X","Y",'Z','LocalHour','NPTS'], engine='python')
    data["Time"] = pd.to_datetime(data["Time"], format='%Y/%m/%d %H:%M:%S')
    data = data.set_index("Time")
    y = data['Y']
    z = data['Z']
    data['Rs'] = ((z*z)+(y*y))**0.5
    data["Elevation"] = np.degrees(np.arctan(data["BZ"]/(data["BX"]**2 + data["BY"]**2)**0.5))
    data["Azimuth"] = np.degrees(np.arccos(-data["BX"]/(data["BX"]**2 + data["BY"]**2)**0.5))
    return data

#Define function that retrieves a given orbit from the calculated peaks
def getorbit(year, n):
    """Gets an orbit from the specified year."""
    data2=get_data2(year)
    Rs = data2['Rs'].to_numpy()
    peaks,_ = find_peaks(Rs, height=12, distance=10000)
    orbit = data2[peaks[n-1]:peaks[n]]

    return orbit

def plot_KSM(data, year, start_day, end_day, xFormat = "Hour"):
    """This function plots the Elevation, Azimuth and Total B_Field for a specified orbit """

    #calculate start and end dates as well as their indices
    first_day = datetime(int(year), 1, 1)
    start_date = pd.Timestamp(first_day + timedelta(start_day-1, 30))
    end_date = pd.Timestamp(first_day + timedelta(end_day-1, 30))

    #use the nearest available start and end dates if they are not in data
    try:
        start_index = list(data.index).index(start_date)
        end_index = list(data.index).index(end_date)
        print("Start Date: ",start_date)
        print("End Date: ",end_date)

    except ValueError:
        start_index = data.index.get_loc(start_date, method="nearest")
        end_index = data.index.get_loc(end_date, method="nearest")
        print("Start Date: ",data.index[start_index])
        print("End Date: ",data.index[end_index])

    #retrieve data for specified days for B_tot, Elevation, and Azimuth
    timeframe = data.index[start_index:end_index]
    B_tot = data["Btotal"][start_index:end_index]
    elevation = data["Elevation"][start_index:end_index]
    azimuth = data["Azimuth"][start_index:end_index]

    #format dates for x-axis in plot
    if xFormat == "Hour":
        xlocator = mdates.HourLocator(interval = 5)
        x_fmt = mdates.DateFormatter('%H:%M:%S')
    elif xFormat == "Day": 
        xlocator = mdates.DayLocator(interval = 1)
        x_fmt = mdates.DateFormatter("%m/%d")

    gs = gridspec.GridSpec(nrows=3, ncols=1)

    ax1 = plt.subplot(gs[0])
    ax1.set_title("Cassini FGM Data (KSM, 1-minute average) Days %s-%s, %s"%(str(start_day), str(end_day), year))
    ax1.plot(timeframe, elevation, "black", label="$\lambda$")
    ax1.set_ylabel(r'Elevation $ \lambda, \left(deg\right)$')
    ax1.axhline(0, color= "black", ls="--")

    ax2 = plt.subplot(gs[1],sharex=ax1)
    ax2.plot(timeframe, azimuth, "black", label="$\delta$")
    ax2.set_ylabel(r'Azimuth $\delta, \left(deg\right)$')

    ax3 = plt.subplot(gs[2],sharex=ax1)
    ax3.plot(timeframe, B_tot,'black',label="$Total B-Field $")
    ax3.set_ylabel(r'$Total B-Field \left(nT\right)$')
    ax3.xaxis.set_major_locator(xlocator)
    ax3.xaxis.set_major_formatter(x_fmt)
    ax3.set_xlabel('SCET')

#define a function that fills the data and makes the time column the index of the data frame
def fill_data(data): 
    """Add columns for Rs=(x**2 + y**2)**0.5 and the elevation and azimuthal angles."""
    #Set the index as the 'Time' column
    data["Time"] = pd.to_datetime(data["Time"], format='%Y/%m/%d %H:%M:%S')
    data = data.set_index("Time")

    #add columns
    y = data["Y"]
    z = data["Z"]
    data['Rs'] = ((z*z)+(y*y))**0.5
    data["Elevation"] = np.degrees(np.arctan(data["BZ"]/(data["BX"]**2 + data["BY"]**2)**0.5))
    data["Azimuth"] = np.degrees(np.arccos(-data["BX"]/(data["BX"]**2 + data["BY"]**2)**0.5))

    return data

def get_day_data(daymin, daymax, year):
    """Returns a sample of the data for a given day range and year"""
    global get_data, fill_data

    yeardata = get_data(year)
    yeardata = fill_data(yeardata)
    start_date = pd.Timestamp(datetime(int(year), 1, 1) + timedelta(daymin-1, 30))
    end_date = pd.Timestamp(datetime(int(year), 1, 1) + timedelta(daymax-1, 30))
    start_index = yeardata.index.get_loc(start_date, method="nearest")
    end_index = yeardata.index.get_loc(end_date, method="nearest")
    print("Sample start date: ",yeardata.index[start_index],"\nSample end date:",yeardata.index[end_index])
    yeardata = yeardata[start_index:end_index]

    return yeardata

def crossings_graph(daymin, daymax, year):
    """Plots the bow shock and magnetopause crossings."""
    global get_day_data

    #retrieve data for specified day range
    tempdata = get_day_data(daymin, daymax, year)
    circledata = tempdata[::1440] #data points to use in plot to give a sense of the movement of the satellite

    #days appearing in circledata to annotate in plot
    days_annotation = [str(i+1) for i in list((circledata.index - datetime(year,1,1)).days)] #create list of strings with days

    #plot crossings
    fig, ax = plt.subplots(figsize=(8,12))
    ax.plot(tempdata['X'], tempdata['Rs'], color="black")
    ax.plot(circledata['X'], circledata['Rs'],"o", color="black")
    ax.plot(0, 0,'o',markersize=15, linewidth=30, mfc='none', color="black")

    ax.invert_yaxis()
    #annotate days in plot
    for i, day in enumerate(days_annotation):
        ax.annotate(day, (circledata['X'][i], circledata['Rs'][i]))
    ax.annotate("S", (0,0),xytext=(5,5),textcoords='offset points')

    ax.set_xlabel(r"$X_{KSM}  \left(R_s\right)$")
    ax.set_ylabel(r"$\left(Y_{KSM}^2 + Z_{KSM}^2\right)^{1/2} \left(R_s\right)$")
    ax.set_title("Bow Shock and Magnetopause Crossings\n During the SOI Orbit Projected into Cylindrical KSM Coordinates, Days %s-%s, %s"%(str(daymin), str(daymax), str(year)));

def identify_crossings_1(data, windowsize, estimatedNumCrossings):
    """Retrieve times where Cassini crosses Saturn's bow shock on a prolonged basisin data that is
    restricted to a specified timeframe."""

    degree = 1 #degree of polynomial fit
    timesteps = np.linspace(1, windowsize, windowsize) #define arbitrary timesteps

    gradient_list = []
    std_list = []

    #iterate over timeframe specified within data
    for i in range(np.shape(data)[0]-windowsize):
        window = data["Btotal"][i:i+windowsize] #get window of data for B_tot
        grad = np.polyfit(timesteps, window, degree)[0] #calculate gradient
        std = np.std(window)
        gradient_list.append(abs(grad))
        std_list.append(std)

    #retrieve indices for top unconsecutive gradients
    ind = np.sort(np.argpartition(gradient_list, -(estimatedNumCrossings*windowsize))[-(estimatedNumCrossings*windowsize):])
    #remove indices that are within windowsize of each other
    ind = np.delete(ind, np.argwhere(np.ediff1d(ind) <= windowsize) + 1)

    ind_std = np.sort(np.argpartition(std_list, -estimatedNumCrossings*windowsize)[-estimatedNumCrossings*windowsize:])
    
    #remove indices that are within windowsize of each other
    ind_std = np.delete(ind_std, np.argwhere(np.ediff1d(ind_std) <= windowsize) + 1)
    
    #crossing times found
    crossing_times = data.index[ind_std]

    ksmcoords = []
    for i in range(len(crossing_times)):
        ksm = data.loc[str(crossing_times[i])]
        ksm = ksm.values.tolist()
        ksmcoords.append([ksm[4], ksm[5], ksm[5]])

    return ind_std,crossing_times, ksmcoords

def smoothdata(data,windowsize,polyorder):
    """Returns a smoothed out form of the data"""

    return signal.savgol_filter(data,windowsize,polyorder)

def identify_crossings_2(orbit_data, windowsize, estimatedNumCrossings,polyorder=3, edge_ord=2, crossing_interval=200):
    """Retrieve times where Cassini crosses Saturn's bow shock on a prolonged basisin data that is
    restricted to a specified timeframe."""

    orbit_m = ndimage.median_filter(orbit_data['Btotal'], size=40) #apply median filter to data

    dif3 = np.gradient(smoothdata(orbit_m, windowsize, polyorder), 0.01, edge_order=edge_ord)

    h = np.quantile(dif3,1-(2*(estimatedNumCrossings+1)/len(dif3)))

    #find entries and exits
    entry,_ = find_peaks(dif3,height=h,distance=crossing_interval)
    exit,_ = find_peaks(-dif3,height=h,distance=crossing_interval)

    #plot on axes 
    fig, ax = plt.subplots()
    ax.plot(orbit_data["Btotal"],c='k')

    #ax.set_title("Bow Shock Crossings")
    ax.set_xlabel("SCET",fontsize = 12)
    ax.set_ylabel(r'$Total B-Field \left(nT\right)$',fontsize = 12);
    ax.plot(orbit_data.index[entry],orbit_data['Btotal'][entry] ,'x',color= "b",label='entry',markersize=12,linewidth=0)
    ax.plot(orbit_data.index[exit], orbit_data['Btotal'][exit] ,'x',color= "r",label='exit',markersize=12,linewidth=0)
    ax.legend()

    #ax.plot(entry,dif3[entry],'x',c='k')
    #ax.plot(exit,dif3[exit],'x',c='r');

    #extract entry and exit times
    entries = orbit_data.index[entry]
    exits = orbit_data.index[exit]

    cross_times=[]
    for i in range(len(entry)):
        cross_times.append(entry[i])
    for i in range(len(exit)):
        cross_times.append(exit[i])

    cross_times=np.sort(cross_times)
    cross_times2=orbit_data.index[cross_times]
    cross_times=[]
    for i in range(len(cross_times2)):
        cross_times3 = str(cross_times2[i].strftime('%Y-%m-%d %H:%M:%S'))
        cross_times.append(cross_times3)


    ksmcoords = []

    for i in range(len(cross_times)):
        ksm = orbit_data.loc[cross_times[i]]
        ksm = ksm.values.tolist()
        ksmcoords.append([ksm[4], ksm[5], ksm[6]])


    return entries, exits, ax, ksmcoords,cross_times

def calc_theta(coordinates):
    """Calculates the values of r and cos(theta) given a coordinate."""
    coordinates = np.array(coordinates)
    #calculate r
    r = np.sqrt(np.sum(np.square(coordinates), axis=1))

    #using trigonometry calculate cos(theta)
    cos_theta = (coordinates[:,0])/r

    return cos_theta, r

def calc_L(coordinates):
    """Calculates the values of r and cos(theta) for given coordinates. Then uses the following equation $L=R(\epsilon cos(\theta))$    
    (3.0) with an uncertainty of $L=R(\delta \epsilon cos(\theta))$ (3.1) and returns this in a n dimentional array  with the first value               corresponding to the L values and second corresponding to the uncertainty in L values."""
    #obtain cos(theta) and r
    cos_theta, r = calc_theta(coordinates)

    #Master's model - equations (3.0/3.1)
    epsilon = 1.05
    unc_epsilon = 0.09
    L = np.array(r*(1+epsilon*cos_theta))
    unc_L = np.array(r*unc_epsilon*cos_theta)
    print(L)
    print(unc_L)
    #join the values of L and its uncertainties 
    array_L = np.vstack((L, unc_L)).T

    return array_L

def SolarWind(L):
    """Calculates the values of dynamic pressure using the following equation $e^{-4.2ln(\frac{200L}{5043})}$ (4) and associated uncertainty:         $e^{-4.2ln(\frac{200\delta L}{5043})}$ (4.1). Producing a n dimensional array with the first value corresponding to the dynamic pressure 
    associated for each L and second corresponding to the uncertainty in those pressure values."""
    #Master's model - equation (4)
    P = np.exp(-4.2*np.log(200*L[:,0]/5043))
    unc_P = L[:,1]*(1/L[:,0])*np.exp(-4.2*np.log(200*L[:,0]/5043)) #*(-4.2)
    print(P)
    print(unc_P)
    #join the values of P and its uncertainties
    array_P = np.vstack((P, unc_P)).T

    return array_P


def RSNeq(x):
    y = x[:,1]
    x = x[:,0]
    #eq. 5
    return 12.3*(x**(-1/4.3)), (-12.3/4.3)*(x**(-5.3/4.3))*y

def RMP(x0, epsilon):
    #eq. 6
    return x0 +(l/(1+epsilon))

def masters(data):
    #model unadjusted equation

    eps= np.linalg.norm(KSM)/np.linalg.norm(data["X","Y","Z"].values.tolist())
    c_array =scipy.optimize.curve_fit(RSNeq,PSW,RSN)
    #figure out plasma mass density estimation from nick
    #plot for epsilon
    #plot fig 6 for c1 c2
    m, c = np.linalg.lstsq(PSW, RSN, rcond=None)[0]
    epsilon

    P_sw = []
    R_sn = []
    plt.figure()
    plt.plot(P_sw, R_sn, 'k.')
    plt.errorbar(xerr=P_err)
    y = np.linspace(min(P_sw), max(P_sw), 10000)
    plt.plot(P_sw, y, 'k-')
    plt.show()
    #adjusted equation
    #addd to the saturn plot in ksm above.

def crossing_time(entries, exits):
    '''this function output the entry and exits time using in a list'''
    cross_array = []
    for i in range(len(entries)):
        entry = str(entries[i].strftime('%Y-%m-%d %H:%M:%S'))
        exit = str(exits[i].strftime('%Y-%m-%d %H:%M:%S'))
        cross_array.append(entry)
        cross_array.append(exit)
    print(cross_array)
    return cross_array
def extract_B_vector(start_time, end_time, data):
    """This function extracts the B vector for the given time interval"""
    start_index =  list(data.index.strftime('%Y-%m-%d %H:%M:%S')).index(start_time)
    end_index = list(data.index.strftime('%Y-%m-%d %H:%M:%S')).index(end_time)

    B_X = data["BX"][start_index+1:end_index]
    B_Y = data["BY"][start_index+1:end_index]
    B_Z = data["BZ"][start_index+1:end_index]
    B_vector = np.array([B_X,B_Y,B_Z])
    return B_vector

def divide(data, num_of_intervals):
    """This function will divide the upstream/downstream intervals into subintervals"""
    sub_3vectors = []
    num_for_each_interval = len(data[0])//num_of_intervals
    for components in data:
        comp_sub = []
        for j in range(num_of_intervals):
            if j != num_of_intervals-1:
                comp_sub.append(components[j*num_for_each_interval:(j+1)*num_for_each_interval])
            else:
                comp_sub.append(components[j*num_for_each_interval:])
        sub_3vectors.append(comp_sub)
    return sub_3vectors

def average_B(data):
    """This function calculates the average upstream and downstream magnetic field vectors for each subinterval.
    Our data is in the format of[[BX1, BX2, ...],[BY1, BY2, ...],[BZ1, BZ2, ...]] and we want to output is [[averageBX1, ..., average BX8],[averageBY1, ..., average BY8],[averageBZ1, ..., average BZ8]]"""
    B_average = []
    for components in data:
        component = []
        for subintervals in components:
            ave_int = np.sum(subintervals)/len(subintervals)
            component.append(ave_int)
        B_average.append(component)

    return B_average

def nCP(BD_average,BU_average):
    """This function calculates the coplanarity normal to the shock surface using eq(3)."""
    cross_product = np.cross(np.subtract(BD_average,BU_average), np.cross(BD_average,BU_average))
    nCP = cross_product/np.linalg.norm(cross_product)
    return nCP
def ncp_shock(BD,BU,num_intervals,cross_num):
    divided_BD_vector = divide(BD, num_intervals)
    divided_BU_vector = divide(BU, num_intervals)
    ave_BD = average_B(divided_BD_vector)
    ave_BU = average_B(divided_BU_vector)

# calculates 64 coplanarity normals and shock values for each possible combination from both sets of BU and BD values.
    ncp, shock_value = subinterval_values(ave_BU,ave_BD)

# calculates the average shock value and its standard deviation.
    ave_shock = np.mean(shock_value)
    print(f"S={cross_num} θ_BN value = {ave_shock}")
    std = np.std(shock_value)
    print(f"Standard deviation = {std}")
    return ncp,shock_value, ave_shock,std

def shock_angle(BU_average,nCP):
    """This function calculates the shock angle, which is the angle between the upstream solar wind BU, and the coplanarity normal to the shock surface nCP."""
    #The shock angle is calculated by rearranging the dot product formula a.b = |a||b|cos(theta) and calculating theta, where theta is the shock angle.
    shock_angle = 180 - np.arccos(np.dot(BU_average,nCP)/(np.linalg.norm(BU_average)*np.linalg.norm(nCP)))*180/np.pi
    return shock_angle

def subinterval_values(BU_div,BD_div):
    """Takes in the sub-interval values of BD and BU, then calculates the shock angle and coplanarity normal for each iteration
       The input dimension is: [[BXs][BYs][BZs]]"""
    nCP_values = []
    shock_values = []
    for i in range(len(BU_div[0])):
        for j in range(len(BU_div[0])):
            BU_data = np.array([BU_div[0][i], BU_div[1][i], BU_div[2][i]])
            BD_data = np.array([BD_div[0][j], BD_div[1][j], BD_div[2][j]])

            nCP_values.append(nCP(BD_data,BU_data))
            shock_values.append(shock_angle(BU_data,nCP_values[-1]))
    return nCP_values,shock_values
def extract_B_vector3(crossing_time, data, mins,offset):
    """This function extracts the B vector for the given time interval"""
    crossing_index = np.linspace(0, 1, len(crossing_time))
    crossing_index1 = np.linspace(0, 1, len(crossing_time))
    B_vector1 = []
    B_vector2 = []
    for i in range(len(crossing_time)):
        crossing_index1[i] = list(
            data.index.strftime('%Y-%m-%d %H:%M:%S')).index(crossing_time[i])
        crossing_index[i] = int(crossing_index1[i])
        a = crossing_index[i]

        B_X1 = data["BX"][int(a) + offset:int(a) + mins]
        B_Y1 = data["BY"][int(a) + offset:int(a) + mins]
        B_Z1 = data["BZ"][int(a) + offset:int(a) + mins]
        B_vector1.append(np.array([B_X1, B_Y1, B_Z1]))  #B after crossing
        B_X2 = data["BX"][int(a) - mins:int(a) - offset]
        B_Y2 = data["BY"][int(a) - mins:int(a) - offset]
        B_Z2 = data["BZ"][int(a) - mins:int(a) - offset]
        B_vector2.append(np.array([B_X2, B_Y2, B_Z2]))  #B before crossing
    return B_vector1, B_vector2


def ncp_shock3(B1, B2, num_intervals, cross_num, crossing_type):
    ncp = []
    shock_value = []
    ave_shock = np.linspace(0, 1, len(B1))
    std = np.linspace(0, 1, len(B1))
    ave_ncp = np.zeros((len(B1),3))
    std_ncp = np.zeros((len(B1),3))
    for i in range(len(B1)):
        if crossing_type[i] == 'entry':
            divided_BD_vector = divide(B1[i], num_intervals)
            divided_BU_vector = divide(B2[i], num_intervals)
            ave_BD = average_B(divided_BD_vector)
            ave_BU = average_B(divided_BU_vector)
        elif crossing_type[i] == 'exit':
            divided_BD_vector = divide(B2[i], num_intervals)
            divided_BU_vector = divide(B1[i], num_intervals)
            ave_BD = average_B(divided_BD_vector)
            ave_BU = average_B(divided_BU_vector)


# calculates 64 coplanarity normals and shock values for each possible combination from both sets of BU and BD values.
        ncp1, shock_value1 = subinterval_values(ave_BU, ave_BD)
        ncp.append(ncp1)
        shock_value.append(shock_value1)

        # calculates the average shock value and its standard deviation.
        ave_shock[i] = np.mean(shock_value1)
        ave_ncp[i] = sum(ncp[i])/len(ncp[i])
        print(f"S={cross_num[i]} θ_BN value = {ave_shock[i]}")  # value for S2
        std[i] = np.std(shock_value1)
        
        for j in range(3):
            sum2=0
            for k in range(64):
                sum1=(ncp1[k][j] - ave_ncp[i][j])**2
                sum2=sum2+sum1
            std_ncp[i][j]= np.sqrt(sum2/(64))
        print(f"Standard deviation = {std[i]}")
    return ncp,ave_ncp,std_ncp ,shock_value,ave_shock, std
def Plot_Dist_for_theta(theta_list):
    '''The theta list contains lists of theta values for different crossings
      In the format of this : [[estimats for crossing 1][estimates for crossing 2][...][...][...]]'''
    kurtosis_list = []
    skewness_list = []
    for crossing in theta_list:
        length = len(crossing)
        ave_shock = np.sum(crossing)/length
        std = np.sqrt(np.sum((crossing - ave_shock)**2/length))
        kurtosis = np.sum([(theta - ave_shock)**4/length for theta in crossing])/std**4
        skewness = np.sum([(theta - ave_shock)**3/length for theta in crossing])/std**3
        kurtosis_list.append(kurtosis)
        skewness_list.append(skewness)
    plt.figure(figsize = (15,8))
    plt.plot(np.array(kurtosis_list),np.array(skewness_list),'+',markersize = 25,c='k')
    plt.xlim(1,5)
    plt.ylim(-2,2)
    plt.axhline(0, color= "black", ls="--")
    plt.axvline(3, color='k', linestyle='--')
    for i in range(len(kurtosis_list)):
        S='S'+str(i+1)
        plt.annotate(S, (kurtosis_list[i],skewness_list[i]+0.01) )
def plotcross(day_data_2004,cross_array,point1,point2,ave_ncp,std_ncp):
    theta=np.linspace(-np.pi/1.4,np.pi/1.4,1000)


    a=1.5
    X=day_data_2004.loc[cross_array[point2-1],'X']
    RHO=day_data_2004.loc[cross_array[point2-1],'Rs']
    at_1=np.arctan(RHO/X)
    if at_1<0:
        at_1=at_1+np.pi
    l_1=(RHO*(1+(a*np.cos(at_1))))/(np.sin(at_1))
    X=day_data_2004.loc[cross_array[point1-1],'X']
    RHO=day_data_2004.loc[cross_array[point1-1],'Rs']
    at_2=np.arctan(RHO/X)
    if at_2<0:
        at_2=at_2+np.pi
    l_2=(RHO*(1+(a*np.cos(at_2))))/(np.sin(at_2))
    l=[abs(l_1),abs(l_2)]
    
    

    for i in range(len(l)):
        r=l[i]/(1+(a*np.cos(theta)))
        x_model=r*np.cos(theta)
        rho_model=r*np.sin(theta)
        plt.plot(x_model,rho_model,'-.',color='k',label='$S_{BS}$')
    a=0.92
    X=day_data_2004.loc[cross_array[point2-1],'X']
    RHO=day_data_2004.loc[cross_array[point2-1],'Rs']
    at_1=np.arctan(RHO/X)
    if at_1<0:
        at_1=at_1+np.pi
    l_1=(RHO*(1+(a*np.cos(at_1))))/(np.sin(at_1))
    X=day_data_2004.loc[cross_array[point1-1],'X']
    RHO=day_data_2004.loc[cross_array[point1-1],'Rs']
    at_2=np.arctan(RHO/X)
    if at_2<0:
        at_2=at_2+np.pi
    l_2=(RHO*(1+(a*np.cos(at_2))))/(np.sin(at_2))
    l=[abs(l_1),abs(l_2)]
    for i in range(len(l)):
        r=l[i]/(1+(a*np.cos(theta)))
        x_model=r*np.cos(theta)
        rho_model=r*np.sin(theta)
        plt.plot(x_model,rho_model,'-.',color='r',label='H')
    plt.legend()

    plt.gca().invert_yaxis()
    
def plotncp(day_data,cross_array,scal,ave_ncp,std_ncp,point1,point2):
    plt.figure(figsize=(10,12))
    a=[]
    b=[]
    c=[]

    #x.invert_yaxis()
    #ax.plot(day_data['X'],day_data['Rs'],color='k',label='orbit path')
    for i in range(len(cross_array)):
        X=day_data.loc[cross_array[i],'X']
        RHO=day_data.loc[cross_array[i],'Rs']
        Y=day_data.loc[cross_array[i],'Y']
        Z=day_data.loc[cross_array[i],'Z']
        plt.plot(X,RHO,'o',color='k')
        S='S'+str(i+1)
        plt.annotate(S, (X-0.7,RHO ))
        x_ncp=ave_ncp[i][0]
        y_ncp=ave_ncp[i][1]
        z_ncp=ave_ncp[i][2]
        sx_ncp=std_ncp[i][0]
        sy_ncp=std_ncp[i][1]
        sz_ncp=std_ncp[i][2]
        
        dot=[Y,Z]/RHO
        rho_ncp=np.dot([y_ncp,z_ncp],dot)
        srho_ncp=np.dot([sy_ncp,sz_ncp],dot)
        errorr=srho_ncp*scal
        errorx=sx_ncp*scal
        plt.plot(X,RHO,'o',color='k')
        if (x_ncp*scal)>=0:
            plt.plot([X,X+(x_ncp*scal)],[RHO,RHO+(rho_ncp*scal)],color='k')
            plt.errorbar(X+(x_ncp*scal),RHO+(rho_ncp*scal),xerr=errorx,yerr=errorr,color='k')
        else:
            plt.plot([X,X+(x_ncp*-scal)],[RHO,RHO+(rho_ncp*-scal)],color='k')
            plt.errorbar(X+(x_ncp*-scal),RHO+(rho_ncp*-scal),xerr=errorx,yerr=errorr,color='k')
        
        a.append(x_ncp)
        b.append(rho_ncp)
        c.append(srho_ncp)
    #ax.legend()
    
    
    plt.gca().invert_yaxis
    plotcross(day_data,cross_array,point1,point2,ave_ncp,std_ncp)
    return a,b,c
def calCPM(day_data_2004,cross_array,ave_ncp,rho_ncp,std_ncp,srho_ncp):
    theta=np.linspace(-np.pi/1.2,np.pi/1.2,1000)
    X=day_data_2004.loc[cross_array[0],'X']
    RHO=day_data_2004.loc[cross_array[0],'Rs']
    at=np.arctan(RHO/X)
    a=0.92
    t=[]
    st=[]
    l=(X*(1+(a*np.cos(at))))/(np.cos(at))

    for i in range(len(rho_ncp)):
        X=day_data_2004.loc[cross_array[i],'X']
        RHO=day_data_2004.loc[cross_array[i],'Rs']
        at2=np.arctan(RHO/X)
        if at2<0:
            at2=at2+np.pi
        l_1=abs((RHO*(1+(a*np.cos(at2))))/(np.sin(at2)))
        dxdt=-(l_1*np.sin(at2))/((1+a*np.cos(at2))**2)
        drhodt=(l_1*(np.cos(at2)+a))/((1+a*np.cos(at2))**2)
        m2=drhodt/dxdt
        m1=rho_ncp[i]/ave_ncp[i][0]
        cross1=dxdt/(((dxdt*dxdt)+(drhodt*drhodt))**(0.5))
        cross2=drhodt/(((dxdt*dxdt)+(drhodt*drhodt))**(0.5))
        nx_ncp=ave_ncp[i][0]/np.linalg.norm([ave_ncp[i][0],rho_ncp[i]])
        n=np.linalg.norm([ave_ncp[i][0],rho_ncp[i]])
        n2=np.linalg.norm([cross1,cross2])
        nrho_ncp=rho_ncp[i]/np.linalg.norm([ave_ncp[i][0],rho_ncp[i]])
        angle=np.arccos(np.dot([cross1,cross2],[nx_ncp,nrho_ncp]))
        sdt_xangle=((cross1*cross2*nx_ncp)-(nrho_ncp**3))/((1-(((nx_ncp*nrho_ncp)+(cross1*cross2))**2))**0.5)
        sdt_rhoangle=((cross1*cross2*nrho_ncp)-(nx_ncp**3))/((1-(((nx_ncp*nrho_ncp)+(cross1*cross2))**2))**0.5)
        nsx_ncp=std_ncp[i][0]#/np.linalg.norm([std_ncp[i][0],srho_ncp[i]])
        nsrho_ncp=srho_ncp[i]#/np.linalg.norm([std_ncp[i][0],srho_ncp[i]])
    
    
    
        std_angle=(((sdt_xangle*(nsx_ncp))**2)+((sdt_rhoangle*nsrho_ncp)**2))**(0.5)
    
        if np.degrees(std_angle)>90:
            std_angle=abs(std_angle-(np.pi))

    
        t.append(abs(abs(np.degrees(angle))-90))
        st.append(np.degrees(std_angle))
    print('Theta_CPM ',t)
    print('Std',st)
    X=day_data_2004.loc[cross_array[0],'X']
    RHO=day_data_2004.loc[cross_array[0],'Rs']
    at2=np.arctan(RHO/X)
    dxdt=-(l*np.sin(at2))/((1+a*np.cos(at2))**2)
    drhodt=(l*(np.cos(at2)+a))/((1+a*np.cos(at2))**2)
    r=l/(1+(a*np.cos(theta)))
    x_model=r*np.cos(theta)
    rho_model=r*np.sin(theta)

    plt.figure()
    plt.plot(r*np.cos(at),r*np.sin(at),label='Normal')
    plt.plot(X,RHO,'.')
    plt.plot(x_model,rho_model,'-.',color='k',label='H model')
    plt.plot([X-(dxdt),X,X+(dxdt)],[RHO-(drhodt),RHO,RHO+(drhodt)],'-.',color='g',label='Tangent')
    
    for i in range(len(srho_ncp)):
        if ave_ncp[i][0]>=0:
            plt.plot([X,X+(ave_ncp[i][0])],[RHO,RHO+(rho_ncp[i])],'-.',color='r')
            S='NCP S'+str(i+1)
            plt.annotate(S, (X+(ave_ncp[i][0]),RHO+(rho_ncp[i])), fontsize=6)
        else:
            plt.plot([X,X-(ave_ncp[i][0])],[RHO,RHO-(rho_ncp[i])],'-.',color='r')
            S='NCP S'+str(i+1)
            plt.annotate(S, (X-(ave_ncp[i][0]),RHO-(rho_ncp[i])), fontsize=6)
            
        
    
    plt.legend()
    return t,st

#### additional functinos in orbits notebook
def Plot_Dist_for_theta(theta_list):
    '''The theta list contains lists of theta values for different crossings
      In the format of this : [[estimats for crossing 1][estimates for crossing 2][...][...][...]]'''
    kurtosis_list = []
    skewness_list = []
    for crossing in theta_list:
        length = len(crossing)
        ave_shock = np.sum(crossing)/length
        std = np.sqrt(np.sum((crossing - ave_shock)**2/length))
        kurtosis = np.sum([(theta - ave_shock)**4/length for theta in crossing])/std**4
        skewness = np.sum([(theta - ave_shock)**3/length for theta in crossing])/std**3
        kurtosis_list.append(kurtosis)
        skewness_list.append(skewness)
    plt.figure(figsize = (15,8))
    plt.plot(np.array(kurtosis_list),np.array(skewness_list),'+',markersize = 25,c='k')
    plt.xlabel("Kurtosis of ${\Theta}_{BN}$")
    plt.ylabel("Skewness of ${\Theta}_{BN}$")
    plt.xlim(1,5)
    plt.ylim(-2,2)
    plt.axhline(0, color= "black", ls="--")
    plt.axvline(3, color='k', linestyle='--')
    for i in range(len(kurtosis_list)):
        S='S'+str(i+1)
        plt.annotate(S, (kurtosis_list[i],skewness_list[i]+0.01) )

def selectBvector(start_time, end_time, time_width, data, exclude_time = 7):
    """
    Finds region of minimum B field deviation within a bow shock crossing
    
    """

    start_index =  list(data.index.strftime('%Y-%m-%d %H:%M:%S')).index(start_time)
    end_index = list(data.index.strftime('%Y-%m-%d %H:%M:%S')).index(end_time)
    number_sub = (end_index-start_index-2*exclude_time)//time_width
    std_list = []
    B_total_list = []
    B_vector_list = []
    for i in range(number_sub):
        if i+1 == number_sub:
            B_total = data["Btotal"][(start_index+exclude_time)+time_width*i:end_index-exclude_time]
            BX = data["BX"][(start_index+exclude_time)+time_width*i:end_index-exclude_time]
            BY = data["BY"][(start_index+exclude_time)+time_width*i:end_index-exclude_time]
            BZ = data["BZ"][(start_index+exclude_time)+time_width*i:end_index-exclude_time]
            B_vector_list.append([BX,BY,BZ])
            B_total_list.append(B_total)
        else:
            B_total = data["Btotal"][(start_index+exclude_time)+time_width*i:(start_index+exclude_time)+time_width*(i+1)]
            BX = data["BY"][(start_index+exclude_time)+time_width*i:(start_index+exclude_time)+time_width*(i+1)]
            BY = data["BY"][(start_index+exclude_time)+time_width*i:(start_index+exclude_time)+time_width*(i+1)]
            BZ = data["BZ"][(start_index+exclude_time)+time_width*i:(start_index+exclude_time)+time_width*(i+1)]
            B_vector_list.append([BX,BY,BZ])
            B_total_list.append(B_total)
    for i in B_total_list: ## find the subinterval with the smallest Btotal std
        ave_i = np.sum(i)/len(i)
        std = np.sqrt(np.sum((i - ave_i)**2/len(i)))
        std_list.append(std)
    best_interval = np.argmin(std_list)
    best_vector_interval = B_vector_list[best_interval]
    return best_vector_interval

def ncp_shock4(BU,BD,num_intervals,cross_num):
    '''outputs the norm vector, shock value estimates, average shock value and the standard deviation for shock values'''
    divided_BU_vector = divide(BU, num_intervals)
    divided_BD_vector = divide(BD, num_intervals)
    ave_BD = average_B(divided_BD_vector)
    ave_BU = average_B(divided_BU_vector)



# calculates 64 coplanarity normals and shock values for each possible combination from both sets of BU and BD values.
    ncp, shock_value = subinterval_values(ave_BU,ave_BD)

# calculates the average shock value and its standard deviation.
    ave_shock = np.mean(shock_value)
    print(f"S={cross_num} θ_BN value = {ave_shock}")# value for S2
    std = np.std(shock_value)
    print(f"Standard deviation = {std}")    
    return ncp,shock_value,ave_shock,std



