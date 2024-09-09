import scipy.constants
from rtlsdr import RtlSdr
import time
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from multiprocessing import Process, Barrier, Queue
import queue
import math
import sys
import argparse

# from scipy.io import loadmat

np.set_printoptions(threshold=sys.maxsize)
"""
    Classes/functions for reading RTL-SDR data. Each receiver is spawned into its own process and data is passed to a
    shared queue. 
"""
def StartProc(numRadios, F, Fs, sampleLength, gain, sampleQueue):
    proc = []
    barrier = Barrier(numRadios)  # barrier for each channel/radio

    # start reading samples from each radio
    for i in range(numRadios):
        # name = "chan" + str(i) + "_proc"
        p = RadioHandler(i + 1, F, Fs, gain, sampleQueue, sampleLength, barrier)
        p.start()
        time.sleep(0.1)  # I don't know why librtlsdr craps out without this delay

"""
    Class for RTL-SDR.
    Args:
      radioNum: serial number
      centerFreq: tuning frequency
      sampRate: sample rate
      gain: default gain
"""
class RadioHandler(Process):
    def __init__(self,
                 radioNum,
                 centerFreq,
                 sampleRate,
                 gain,
                 sampleQ,
                 sampleLength,
                 barrier):

        Process.__init__(self)

        self.centerFreq = centerFreq
        self.radioNum = radioNum
        self.barrier = barrier

        self.deviceIndex = RtlSdr.get_device_index_by_serial(str(self.radioNum))

        self.sampleRate = sampleRate
        self.gain = gain
        self.sdr = 0

        self.sampleQ = sampleQ
        self.sampleLength = sampleLength

    """
        Callback for async sample reads, puts radio number and data into queue
    """
    def __call__(self, samples, radioNum):
        # put data in queue with format: (context, samples[...])
        self.sampleQ.put(np.append(np.array(radioNum), np.array(samples)))

    def run(self):
        # callback = GetSamplesCallback(self.sampleQueue, self.commandQueue)
        self.sdr = RtlSdr(device_index=self.deviceIndex,
                          test_mode_enabled=False,
                          serial_number=None,
                          dithering_enabled=False)

        self.sdr.set_center_freq(self.centerFreq)
        self.sdr.set_agc_mode(False)
        # self.sdr.set_manual_gain_enabled(True)
        self.sdr.set_gain(self.gain)
        self.sdr.set_sample_rate(self.sampleRate)
        self.sdr.set_bandwidth(self.sampleRate)

        self.barrier.wait()  # wait for all processes to reach this point, then start reads at the same time
        self.sdr.read_samples_async(self.__call__, num_samples=self.sampleLength, context=int(self.radioNum))

"""
    Class to compile samples for higher level DSP functions
    Args:
      numRadios: total number of radios (eg. 1-4)
      sampleLength: length of each async read
      sampleQ: queue where callback puts samples into
"""
class CollectSamples:
    def __init__(self, numRadios, sampleLength, sampleQ=Queue()):
        # create empty array of samples to contain data from all radios (format: row = radio, column = samples)
        self.sampleLength = sampleLength
        self.samples = np.zeros((numRadios, self.sampleLength), dtype=complex)
        self.sampleQ = sampleQ
        self.numRadios = numRadios
        self.maxRetries = 3

        # array for keeping record of which samples grabbed from the queue, each entry represents ith radio
        self.gotSamples = np.full(numRadios,False)

    # returns true or false depending on whether we have a new batch of samples available
    def SamplesAvailable(self):
        # check if we have samples from all radios (all True)
        if all(self.gotSamples):
            return True
        else:
            return False

    # returns samples if we have completed a collection from all radios
    def ReturnSamples(self):
        if all(self.gotSamples):
            self.gotSamples = np.full(self.numRadios,False)  # reset gotSamples to all false for next collection
            # print(self.samples)
            return self.samples

    # checks queue for new data, only gets one
    def CheckQueue(self):
        # if data in queue
        if not self.sampleQ.empty():
            fail = 0  # record how many times we retry queue object

            # iterate over queue size to grab multiple sample data sets
            for i in range(self.sampleQ.qsize()):
                try:
                    inData = np.array(self.sampleQ.get())  # retrieve one item from queue
                    index = int(np.real(inData[0])) - 1  # which radio the data originated from (0 to N-1)

                    # if we haven't collected data from that radio yet, get samples, else skip that data
                    if not self.gotSamples[index]:
                        self.samples[index] = inData[1:self.sampleLength + 1]
                        self.gotSamples[index] = True

                    fail = 0  # reset fail counter

                except queue.Empty:
                    if fail < self.maxRetries:
                        i -= 1  # reset index to last value so this index is looped over again
                        fail += 1  # increment fail counter
                        print("Queue retry")

"""
    Functions for plotting data.
"""
class PlotPSD:
    def __init__(self, Fc, Fs):
        self.Fc = Fc
        self.Fs = Fs

        plt.ion()

        self.startTime = time.time()
        self.lastTime = self.startTime
        self.plotInterval = 0.5
        self.currentTime = 0
        self.previousTime = 0

    def plotPSD(self, data):
        # Update PSD plot
        self.currentTime = time.time()

        if self.currentTime - self.previousTime >= self.plotInterval:
            self.previousTime = self.currentTime

            plt.figure(1)

            plt.clf()
            # plt.magnitude_spectrum(data[0], Fs=self.Fs/1e6, scale='dB', Fc=self.Fc/1e6)
            plt.psd(data[0], Fs=self.Fs / 1e6, Fc=self.Fc / 1e6, scale_by_freq=False)
            plt.title('Magnitude Spectrum')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Relative power (dBFS)')
            plt.ylim([-100, 0])
            #plt.legend(["1", "2", "3", "4"], loc="upper right")
            plt.draw()
            plt.pause(0.002)

class PlotTime:
    def __init__(self, Fs, maxPlotTime):
        #maxPlotTime = 1e-4

        sampleInterval = 1/Fs
        self.t = np.arange(0, maxPlotTime, sampleInterval)

        self.startTime = time.time()
        self.lastTime = self.startTime
        self.plotInterval = 1
        self.currentTime = 0
        self.previousTime = 0

    def plotTime(self, data):
        self.currentTime = time.time()

        if self.currentTime - self.previousTime >= self.plotInterval:
            self.previousTime = self.currentTime

            plt.figure(2)
            plt.clf()

            for i in range(len(data)):
                # print(data[i])
                plt.plot(self.t, np.abs(data[i][:len(self.t)]), alpha=0.3, marker='x')

            # plt.plot(self.t, np.abs(data[0][:len(self.t)]), alpha=0.3, marker='x')
            # plt.plot(self.t, np.abs(data[1][:len(self.t)]), alpha=0.3, marker='x')

            plt.title('Signal Over Time')
            plt.xlabel('Time (sec)')
            plt.ylabel('Amplitude')
            plt.legend(["1", "2", "3", "4"], loc="upper right")
            plt.xlim([0.02, 0.021])
            plt.ylim([-0.05, 0.3])
            plt.draw()
            plt.pause(0.002)

class PlotCorr:
    def __init__(self, correlationLength):
        self.xValues = np.arange(0, 2 * correlationLength - 1, 1)

        self.startTime = time.time()
        self.lastTime = self.startTime
        self.plotInterval = 1
        self.currentTime = 0
        self.previousTime = 0

        self.figReal = plt.figure(3)
        self.figImag = plt.figure(6)

        self.axReal = self.figReal.add_subplot()
        self.axImag = self.figImag.add_subplot()

    def plotCorr(self, data):
        self.currentTime = time.time()

        if self.currentTime - self.previousTime >= self.plotInterval:
            self.previousTime = self.currentTime

            self.axReal.cla()
            self.axImag.cla()

            self.axReal.plot(np.real(data[0]), alpha=0.5)
            self.axReal.plot(np.real(data[1]), alpha=0.5)
            self.axReal.plot(np.real(data[2]), alpha=0.5)

            self.axImag.plot(np.imag(data[0]), alpha=0.5)
            self.axImag.plot(np.imag(data[1]), alpha=0.5)
            self.axImag.plot(np.imag(data[2]), alpha=0.5)

            self.axReal.set_xlim([4100, 4200])
            self.axImag.set_xlim([4100, 4200])

            self.axReal.set_title('Correlation between channels (real)')
            self.axImag.set_title('Correlation between channels (imaginary)')

            self.axReal.set_xlabel('Sample Delay')
            self.axImag.set_xlabel('Sample Delay')

            self.axReal.set_ylabel('Correlation Factor')
            self.axImag.set_ylabel('Correlation Factor')

            self.axReal.legend(["1 & 2", "1 & 3", "1 & 4"], loc="upper right")
            self.axImag.legend(["1 & 2", "1 & 3", "1 & 4"], loc="upper right")

            # plt.figure(3)
            # plt.clf()
            #
            # for i in range(len(data)):
            #     plt.plot(self.xValues, data[i], alpha=0.3)
            # # plt.plot(self.xValues, np.real(data[2]))
            # # plt.plot(self.xValues, np.imag(data[2]))
            #
            #
            # plt.title('Correlation between channels')
            # plt.xlabel('Sample Delay')
            # plt.ylabel('Correlation Factor')
            # plt.legend(["1 & 2", "1 & 3", "1 & 4"], loc="upper right")
            # plt.draw()
            # plt.pause(0.002)

class PlotAOA:
    def __init__(self, numPoints, scanAngle):
        self.startTime = time.time()
        self.lastTime = self.startTime
        self.plotInterval = 1
        self.currentTime = 0
        self.previousTime = 0

        self.thetaScan = np.linspace(0, scanAngle,
                                     numPoints)  # 1000 different thetas between -90 and +90 degrees
        self.thetaScanRad = self.thetaScan * np.pi / 180  # convert to radians

        self.fig = plt.figure(5)
        self.ax = self.fig.add_subplot(projection='polar')

    def plotAoa(self, data):
        self.currentTime = time.time()

        if self.currentTime - self.previousTime >= self.plotInterval:
            self.previousTime = self.currentTime

            thetaMax = self.thetaScan[np.argmax(data)]

            # print(np.round(thetaMax, decimals=3))

            plt.figure(4)
            plt.clf()
            plt.plot(self.thetaScan, data)  # plot angle in degrees
            plt.plot([thetaMax], [np.max(data)], color='red', marker='o')  # highlight peak
            #plt.xlim([-90, 90])
            plt.annotate(np.round(thetaMax, decimals=1), (thetaMax, np.max(data)))

            plt.draw()
            plt.pause(0.002)

    def plotAoaPolar(self, data):
        self.currentTime = time.time()

        if self.currentTime - self.previousTime >= self.plotInterval:
            self.previousTime = self.currentTime

            thetaMax = self.thetaScanRad[np.argmax(data)]

            self.ax.cla()
            self.ax.semilogy(self.thetaScanRad, data)
            self.ax.semilogy([thetaMax], [np.max(data)], color='red', marker='o')  # highlight peak
            self.ax.annotate(str(np.round(thetaMax * np.pi / 180, decimals=1)),
                            xy=(float(thetaMax * np.pi / 180), float(np.max(data))),
                             xycoords='polar')
            self.ax.set_theta_zero_location('N')  # make 0 degrees point up
            self.ax.set_theta_direction(-1)  # increase clockwise
            self.ax.set_rlabel_position(45)  # Move grid labels away from other labels
            self.ax.grid(True)
            self.ax.set_thetagrids(np.arange(0, 360, 30))
            # self.ax.set_thetamin(-90)  # only show top half
            # self.ax.set_thetamax(90)

            self.ax.set_title("Angle of Arrival")


"""
    DSP Functions: filter, decimate, correlate, and correct delay.
    Args:
        numRadios: number of receive channels
        sampleLength: length of data before down sampling
        downsampleLength: length of data after down sampling
        downsampleFactor: how much data is down sampled, factor = sampleLength / downsampleLength
        
"""
class DSP:
    def __init__(self, numRadios, sampleLength, downsampleLength, downsampleFactor, filterType):
        self.numRadios = numRadios
        self.sampleLength = sampleLength

        self.filtData = np.zeros((self.numRadios, sampleLength), dtype=complex)

        # VHF Filters
        if filterType == 'huge':
            self.LPFilterCoeff = np.array([[1, -1.98881793496472, 1, 1, -1.97971988633092, 0.985072577263330],
                                           [1, -1.98676631359207, 1, 1, -1.94795766338301, 0.953651336698271],
                                           [1, -1.98085690041635, 1, 1, -1.91055780634462, 0.917023894428133],
                                           [1, -1.96276323213348, 1, 1, -1.86662464003702, 0.874252620327643],
                                           [1, -1.86621618507353, 1, 1, -1.82456592538589, 0.833425332818471],
                                           [1, 1, 0, 1, -0.902892712697377, 0]])

            self.LPFilterScale = np.array([0.478685369430631,
                                           0.430240912452897,
                                           0.337776442903418,
                                           0.204850762503578,
                                           0.0662218179191029,
                                           0.0485536436513117,
                                           1])

        elif filterType == 'wide':
            # Chebyshev type II lowpass, Fpass = 10 khz
            self.LPFilterCoeff = np.array([[1, -1.99842620999190, 1, 1, -1.99361794783738, 0.994373885297106],
                                           [1, -1.99813663647775, 1, 1, -1.98154345787021, 0.982355635997320],
                                           [1, -1.99730112695137, 1, 1, -1.96710136046824, 0.968034736838839],
                                           [1, -1.99472962956153, 1, 1, -1.94975297011701, 0.950869704640288],
                                           [1, -1.98066017927976, 1, 1, -1.93270699768855, 0.934021876209467],
                                           [1, 1, 0, 1, -0.962474075814749, 0]])

            self.LPFilterScale = np.array([0.480329304313391,
                                           0.435866709537057,
                                           0.345839301730525,
                                           0.211889190013269,
                                           0.067988144251164,
                                           0.018762962092626,
                                           1])

        elif filterType == 'small':
            # Fpass = 2.5khz
            self.LPFilterCoeff = np.array([[1, -1.99982332806554, 1, 1, -1.99773568321239, 0.997797798652930],
                                           [1, -1.99977154345942, 1, 1, -1.99305802609194, 0.993125369045612],
                                           [1, -1.99958532328847, 1, 1, -1.98780856961204, 0.987886023909904],
                                           [1, -1.99853570937150, 1, 1, -1.98278076598366, 0.982870051909482],
                                           [1, 1, 0, 1, -0.990251881075268, 0]])

            self.LPFilterScale = np.array([0.351586349761871,
                                           0.294773585809826,
                                           0.186782367357668,
                                           0.0609755495799319,
                                           0.00487405946236583,
                                           1])

        elif filterType == 'narrow':
            # Fpass = 1khz
            self.LPFilterCoeff = np.array([[1, -1.99998409930002, 1, 1, -1.99930004608542, 0.999308291419286],
                                           [1, -1.99997943837701, 1, 1, -1.99773532899739, 0.997744665757954],
                                           [1, -1.99996267654478, 1, 1, -1.99565101759790, 0.995662737313477],
                                           [1, -1.99986817335323, 1, 1, -1.99312325433483, 0.993138368941815],
                                           [1, 1, 0, 1, -0.995868694807982, 0]])

            self.LPFilterScale = np.array([0.518551628354966,
                                           0.454086750501123,
                                           0.314004035960798,
                                           0.114655172928567,
                                           0.00206565259600907,
                                           1])

        elif filterType == 'normal':
            # Fpass = 5 khz
            self.LPFilterCoeff = np.array([[1, -1.99929334535726, 1, 1, -1.99535226018595, 0.995600458358106],
                                           [1, -1.99908625212125, 1, 1, -1.98602948505367, 0.986297943532363],
                                           [1, -1.99834166684626, 1, 1, -1.97561032377051, 0.975918284143802],
                                           [1, -1.99414876230346, 1, 1, -1.96567775478440, 0.966031872106796],
                                           [1, 1, 0, 1, -0.980597503956064, 0]])

            self.LPFilterScale = np.array([0.351229804689344,
                                           0.293799290740268,
                                           0.185704767828524,
                                           0.0605200712665570,
                                           0.00970124802196806,
                                           1])

        self.LPFilterInitCondData = signal.sosfilt_zi(self.LPFilterCoeff)  # compute initial conditions for filter
        self.filtResponseLengthData = 20

        # Decimate
        self.downsampleFactor = downsampleFactor
        self.downsampledData = np.zeros((self.numRadios, downsampleLength), dtype=complex)

        # Calibration filter, Fpass = 25khz
        # Wide filter so we get lots of noise power
        self.calFilterCoeffIIR = np.array([[1, -1.98881793496472, 1, 1, -1.97971988633092, 0.985072577263330],
                                        [1, -1.98676631359207, 1, 1, -1.94795766338301, 0.953651336698271],
                                        [1, -1.98085690041635, 1, 1, -1.91055780634462, 0.917023894428133],
                                        [1, -1.96276323213348, 1, 1, -1.86662464003702, 0.874252620327643],
                                        [1, -1.86621618507353, 1, 1, -1.82456592538589, 0.833425332818471],
                                        [1, 1, 0, 1, -0.902892712697377, 0]])

        self.calFilterScaleIIR = np.array([0.478685369430631,
                                        0.430240912452897,
                                        0.337776442903418,
                                        0.204850762503578,
                                        0.0662218179191029,
                                        0.0485536436513117,
                                        1])


        # mat = loadmat('5k_FIR_30db.mat')
        # self.FIRfilter = mat["Num"].squeeze()

        self.calFilterInitCond = signal.sosfilt_zi(self.calFilterCoeffIIR)  # compute initial conditions for filter
        self.calFiltResponseLength = 20

        # Decimate
        # self.downsampleFactor = downsampleFactor
        # self.downsampledData = np.zeros((self.numRadios, downsampleLength), dtype=complex)

        self.numRadiosMinusOne = self.numRadios - 1
        self.xcorr = np.zeros((self.numRadiosMinusOne, 2 * downsampleLength - 1), dtype=complex)
        self.lags = np.zeros((self.numRadiosMinusOne, 2 * downsampleLength - 1))

        self.lag = np.zeros(self.numRadiosMinusOne)

        self.downsampleLength = downsampleLength
        self.firstCall = True
        self.movingAvgWindow = 20
        self.phaseMovingAvg = np.zeros((self.movingAvgWindow, self.numRadiosMinusOne))
        self.movingAvgCounter = 0
        self.avgLag = np.zeros(self.numRadios)

        self.phaseDiff = np.zeros(self.numRadiosMinusOne)

        self.window = np.hamming(downsampleLength)

        # Phase correction check
        self.xcorrCheck = np.zeros((self.numRadiosMinusOne, 2 * downsampleLength - 1), dtype=complex)
        self.lagsCheck = np.zeros((self.numRadiosMinusOne, 2 * downsampleLength - 1))
        self.lagCheck = np.zeros(self.numRadiosMinusOne)

    def FilterData(self, samples):
        for i in range(self.numRadios):
            # compute initial filter condition based on input samples to minimize transient
            zi = self.LPFilterInitCondData * np.average(samples[i][:self.filtResponseLengthData])

            # apply filter with initial condition, multiply times scale vector
            self.filtData[i], zf = signal.sosfilt(self.LPFilterCoeff, samples[i], zi=zi)
            self.filtData[i] *= np.prod(self.LPFilterScale)

        return self.filtData

    def FilterCalIIR(self, samples):
        for i in range(self.numRadios):
            # compute initial filter condition based on input samples to minimize transient
            zi = self.calFilterInitCond * np.average(samples[i][:self.calFiltResponseLength])

            # apply filter with initial condition, multiply times scale vector
            self.filtData[i], zf = signal.sosfilt(self.calFilterCoeffIIR, samples[i], zi=zi)
            self.filtData[i] *= np.prod(self.calFilterScaleIIR)

        return self.filtData

    # def FilterCalFIR(self, samples):
    #     for i in range(self.numRadios):
    #         self.filtData[i] = signal.filtfilt(self.FIRfilter, 1, samples[i])
    #
    #     return self.filtData

    # Downsample data
    def DecimateData(self, samples):
        for i in range(self.numRadios):
            # decimate by a factor of 24 to 100 kHz
            self.downsampledData[i] = signal.decimate(samples[i], self.downsampleFactor, ftype='fir', zero_phase=True)

        return self.downsampledData

    # Function to determine delay offset between channels, and amplitude normalization
    def DetermineSync(self, samples):
        # chan0_conj = np.conj(samples[0])

        # iterate over each radio to correlate with channel one
        for i in range(self.numRadiosMinusOne):
            # tmpPhase = []
            self.xcorr[i] = signal.correlate(samples[0], samples[i + 1], method="fft")  # compute cross correlation of channel 1 with other channels
            self.lags[i] = signal.correlation_lags(samples[0].size, samples[i + 1].size, mode="full")  # compute lags for all correlation offsets
            self.lag[i] = self.lags[i][np.argmax(np.abs(self.xcorr[i]))]  # find correlation peak where data matches
            tmpSamples = np.roll(samples[i], int(self.avgLag[i - 1]))  # circularly shift array
            xcorr = signal.correlate(samples[0], tmpSamples, method="fft")
            argmax = np.argmax(np.abs(xcorr))
            self.phaseDiff[i] = -np.angle(xcorr[argmax]) / np.pi

        return self.lag, self.xcorr

    # Identical to DetermineOffset, only to check correlation
    def CheckSync(self, samples):
        print("Test diff: ")

        # iterate over each radio to correlate with channel one
        for i in range(self.numRadiosMinusOne):
            self.xcorrCheck[i] = signal.correlate(samples[0], samples[i + 1], method="fft")  # compute cross correlation of channel 1 with other channels
            self.lagsCheck[i] = signal.correlation_lags(samples[0].size, samples[i + 1].size, mode="full")  # compute lags for all correlation offsets
            argmax = np.argmax(np.abs(self.xcorrCheck[i]))
            self.lagCheck[i] = -np.angle(self.xcorr[i][argmax]) / np.pi

        return self.lagCheck, self.xcorrCheck

    # Function to shift data so it is coherent between channels
    def CorrectSync(self, samples):
        newSamples = np.zeros(samples.shape, dtype=complex)
        newSamples[0] = samples[0]  # copy channel 1 samples to output

        for i in range(1, self.numRadios):
            newSamples[i] = np.roll(samples[i], int(self.lag[i - 1]))  # circularly shift array
            newSamples[i] *= np.exp(-1j*self.phaseDiff[i - 1])

        return newSamples

"""
    DSP Functions for AoA.
    Args:
        F: center frequency
        numRadios: number of receive channels
        numSignals: number of signals to look for
        antSpacing: antenna spacing between elements in inches (for all array geometries)
        scanAngle: +/- angle to scan for in the steering vector
        numPoints: number of points in steering vector
        sampleLength: length of input chunks/data
        arrayType: 'linear' for uniform linear array; 'circular' for uniform circular array 
"""
class DSP_AOA:  # TODO: add null steering
    def __init__(self, F, numRadios, numSignals, antSpacing, scanAngle, numPoints, sampleLength, arrayType):
        self.numRadios = numRadios
        self.numSignals = numSignals

        wavelength = scipy.constants.speed_of_light / F

        self.batchSize = 1389  # 4167 / 3
        self.length = sampleLength

        self.thetaScan = np.linspace(0, scanAngle, numPoints)  # 1000 different thetas between -90 and +90 degrees
        self.thetaScan *= np.pi / 180  # convert to radians

        self.normAntSpacing = (antSpacing * 0.0254) / wavelength  # antSpacing argument in inches

        if arrayType == 'linear':
            self.arrayType = 1

        elif arrayType == 'circular':
            self.arrayType = 2

            sf = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2 * np.pi / self.numRadios)))
            r = self.normAntSpacing * sf
            self.x = r * sf * np.cos(2 * np.pi / self.numRadios * np.arange(self.numRadios))
            self.y = -1 * r * sf * np.sin(2 * np.pi / self.numRadios * np.arange(self.numRadios))


        self.window = np.hamming(self.numRadios)

    def MUSIC(self, samples):
        avg = []

        for i in range(self.batchSize, self.length-self.batchSize, self.batchSize):
            # part that doesn't change with theta_i
            R = np.cov(samples[:, i:i+self.batchSize])  # Calc covariance matrix. gives a Nr x Nr covariance matrix
            w, v = np.linalg.eig(R)  # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
            eig_val_order = np.argsort(np.abs(w))  # find order of magnitude of eigenvalues
            v = v[:, eig_val_order]  # sort eigenvectors using this order
            # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
            V = np.zeros((self.numRadios, self.numRadios - self.numSignals), dtype=np.complex64)
            for j in range(self.numRadios - self.numSignals):
                V[:, j] = v[:, j]

            s = 0
            results = []

            for theta_i in self.thetaScan:
                if self.arrayType == 1:
                    s = np.exp(-2j * np.pi * self.normAntSpacing * np.arange(self.numRadios) * np.sin(theta_i))  # Steering Vector
                    s = s.reshape(-1, 1)

                elif self.arrayType == 2:
                    s = np.exp(1j * 2 * np.pi * (self.x * np.cos(theta_i) + self.y * np.sin(theta_i)))
                    s = s.reshape(-1, 1)  # Nrx1

                metric = 1 / (s.conj().T @ V @ V.conj().T @ s)  # The main MUSIC equation
                metric = np.abs(metric.squeeze())  # take magnitude
                metric = 10 * np.log10(metric)  # convert to dB
                results.append(metric)

            results /= np.max(results)  # normalize

            avg.append(results)

        avgAoa = sum(avg)/len(avg)

        return avgAoa


"""
    Main loop
"""

def main():
    parser = argparse.ArgumentParser("rtl_sdr_doa", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f",
                        "--center_frequency",
                        help="Tuning frequency in Hz.",
                        type=lambda fc: int(float(fc)),
                        default=100e6)
    parser.add_argument("-b",
                        "--bandwidth",
                        help="Bandwidth in Hz.",
                        type=lambda bw: int(float(bw)),
                        default=2.4e6)
    parser.add_argument("-g",
                        "--gain",
                        help="Hardware gain, valid options: 0-49 dB.",
                        type=int,
                        default=15)
    parser.add_argument("-c",
                        "--number_channels",
                        type=int,
                        default=4)
    parser.add_argument("-fbw",
                        "--filt_width",
                        help="Filter bandwidth: huge: 25 kHz, wide: 10 kHz, normal: 5 kHz, small: 2.5 kHz, narrow: 1 kHz.",
                        type=str,
                        default="narrow")
    parser.add_argument("-s",
                        "--antenna_spacing",
                        help="Antenna spacing in inches.",
                        type=float,
                        default=7)
    parser.add_argument("-a",
                        "--array_type",
                        help="linear or circular.",
                        type=str,
                        default="circular")
    parser.add_argument("-sig",
                        "--num_signals",
                        help="Number of signals of interest (up to number_channels - 1).",
                        type=int,
                        default=1)


    args = parser.parse_args()

    # Device configuration
    # sampleRate = 2.4e6  # Msps
    # centerFrequency = 162.550e6  # MHz
    # centerFrequency = 460.800e6  # MHz
    # defaultGain = 15

    sampleRate = args.bandwidth  # Msps
    centerFrequency = args.center_frequency  # MHz
    defaultGain = args.gain

    # Array configuration
    # numRadios = 4
    # antSpacing = 7  # inches
    # arrayType = 'circular'
    # numSignals = 1

    numRadios = args.number_channels
    antSpacing = args.antenna_spacing  # inches
    arrayType = args.array_type
    numSignals = args.num_signals

    # Filter type
    # Options:
    #   huge:   25 kHz
    #   wide:   10 kHz
    #   normal: 5 kHz  (default)
    #   small:  2.5 kHz
    #   narrow: 1 kHz
    # filterType = 'normal'
    filterType = args.filt_width

    # DSP configuration
    sampleLength = 100000
    downsampleFactor = 24  # downsampled Fs = 100kHz
    downsampleFs = sampleRate / downsampleFactor
    downsampleLength = math.ceil(sampleLength / downsampleFactor)  # round up downsampled length

    # DoA configuration
    scanAngle = 360  # +/-x degrees  TODO: make this better for circular array
    numPoints = 2000  # AoA theta points in range

    # Debug plot instantiations
    psdSampled = PlotPSD(centerFrequency, sampleRate)
    psdDownsampled = PlotPSD(centerFrequency, downsampleFs)
    # timeplot = PlotTime(sampleRate, sampleLength/sampleRate)

    aoaPlot = PlotAOA(numPoints, scanAngle)

    radiosSynchronized = False

    # Start samples queue
    sampleQ = Queue()

    # Start new process for each channel
    StartProc(numRadios, centerFrequency, sampleRate, sampleLength, defaultGain, sampleQ)

    # Collect and manage new data from queue
    collect = CollectSamples(numRadios, sampleLength, sampleQ)

    # Run DSP
    dsp = DSP(numRadios, sampleLength, downsampleLength, downsampleFactor, filterType)

    # Run direction finding
    dsp_aoa = DSP_AOA(centerFrequency, numRadios, numSignals, antSpacing, scanAngle, numPoints, downsampleLength, arrayType)

    # filteredSamples = np.zeros((numRadios, downsampleLength), dtype=complex)
    downsampledSamples = np.zeros((numRadios, downsampleLength), dtype=complex)
    # angles = np.array(numPoints)

    count = 0

    print('Calibrating...')

    #for i in range(1000000):

    while True:
        collect.CheckQueue()  # check queue for new data
        if collect.SamplesAvailable():  #
            samples = collect.ReturnSamples()

            # Calibration
            if not radiosSynchronized:
                filteredSamples = dsp.FilterCalIIR(samples)  # Lowpass filter incoming samples using calibration filter
                #filteredSamples = dsp.FilterCalFIR(samples)
                downsampledSamples = dsp.DecimateData(filteredSamples)  # Decimate samples
                offsets, xcorr = dsp.DetermineSync(downsampledSamples)  # Calculate delay between channels
                phaseCorrectedSamples = dsp.CorrectSync(downsampledSamples)  # Correct delay offset from last step

                # Stop synchronizing after 150 loops
                if count == 150:
                    radiosSynchronized = True
                    print("Sync: " + str(radiosSynchronized))

                    offsetsCorrected, xcorrCorrected = dsp.CheckSync(
                        phaseCorrectedSamples)  # Sanity check phase alignment
                    print("Corrected channel delay value: " + str(offsetsCorrected))

                count += 1

            # Direction finding
            else:
                filteredSamples = dsp.FilterData(samples)
                #filteredSamples = dsp.FilterCalFIR(samples)
                downsampledSamples = dsp.DecimateData(filteredSamples)
                phaseCorrectedSamples = dsp.CorrectSync(downsampledSamples)
                angles = dsp_aoa.MUSIC(phaseCorrectedSamples)

                aoaPlot.plotAoaPolar(angles)

        """
            Debug plots
        """

        # Time domain plots
        # timeplotFiltered.plotTime(phaseCorrectedSamples)
        # timeplotFiltered.plotTime(downsampledSamples)
        # timeplot.plotTime(samples)

        # Frequency domain plots
        # psdSampled.plotPSD(samples)
        psdDownsampled.plotPSD(downsampledSamples)

        # Cross-correlation plots
        # corrplot.plotCorr(xcorr)
        # corrplot.plotCorr(xcorrCorrected)


    # for i in range(4):
    #     proc[i].join()


if __name__ == '__main__':
    main()
