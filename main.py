import datetime

import serial
import time
import os
import busio
import board
import adafruit_amg88xx
import spidev
from lib_nrf24 import NRF24
from picamera import PiCamera
import RPi.GPIO as GPIO
import classifier
import pynmea2
from gpiozero import LED, Button

EPIDEMIC_LABELS = ["sick", "healthy"]
ANIMAL_LABELS = ["animal", "human", "null"]

# 1 = Sickness
# 2 = High Temperature
# 3 = Animal Detected
# 4 = Human Detected

WARNING_MODES = ['1', '2', '3', '4']

SWITCH_PORT = 3

COOLING_MIN = 60
DETECTION_MIN = 24

MIN_PROBABILITY_FIRE = 65
MIN_PROBABILITY_EPIDEMIC = 55

STATUS_TRAP = False

CAMERA_PATH = ""
EPIDEMIC_TENSORFLOW_PATH = ""
ANIMAL_DETECTION_PATH = ""
HUMAN_DETECTION_PATH = ""

epidemicClassifier = classifier.TensorflowLiteClassificationModel(image_size=(180, 180),
                                                                  model_path=EPIDEMIC_TENSORFLOW_PATH,
                                                                  labels=EPIDEMIC_LABELS)
animalClassifier = classifier.TensorflowLiteClassificationModel(image_size=(180, 180),
                                                                model_path=ANIMAL_DETECTION_PATH,
                                                                labels=ANIMAL_LABELS)

PIPES = [[0xe7, 0xe7, 0xe7, 0xe7, 0xe7]]

button = Button(10)
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PORT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
radio = NRF24(GPIO, spidev.Spidev())
camera = PiCamera()
serialPort = "/dev/ttyAMA0"

radio.begin(0, 17)
radio.setRetries(15, 15)  # number of retries in case of unsuccessfull connection
radio.setPayloadSize(16)  # bytes
radio.setChannel(0x60)  # channel
radio.setDataRate(NRF24.BR_250KBPS)  # change the bandwidth in order to increase distance
radio.setPALevel(NRF24.PA_MIN)  # for power usage

radio.setAutoAck(True)
radio.enableDynamicPayloads()
radio.enableAckPayload()

radio.openWritingPipe(PIPES[0])
radio.printDetails()

print("Wait for modules to initialize ")
time.sleep(5)


def testIRSensor():
    while True:
        for row in amg.pixels:
            print(['{0:.1f}'.format(temp) for temp in row])
            print("")
        print("\n")
        time.sleep(1)


def testGPSModule():
    x = 0
    while x < 20:
        x += 1
        ser = serial.Serial(serialPort, baudrate=9600, timeout=0.5)
        newdata = ser.readline()
        if newdata[0:6] == "$GPGLL":
            newmsg = pynmea2.parse(newdata)
            latitude = newmsg.latitude
            longtitude = newmsg.longtitude
            gps = "Latitude=" + str(latitude) + "Lontitude=" + str(longtitude)
            print(gps)


def testAnimalClassifier():
    print("Wait for 5 seconds")
    time.sleep(5)
    camera.capture(CAMERA_PATH)
    RESULT = animalClassifier.run_from_filepath(CAMERA_PATH)
    print(RESULT[0][0])
    os.remove(CAMERA_PATH)


def testSicknessDetector():
    print("Wait for 5 seconds")
    camera.capture(CAMERA_PATH)
    RESULT = epidemicClassifier.run_from_filepath(CAMERA_PATH)
    print(RESULT[0][0])
    os.remove(CAMERA_PATH)


def changeStatus():
    global STATUS_TRAP
    STATUS_TRAP = not STATUS_TRAP


def scoutMode():
    ser = serial.Serial(serialPort, baudrate=9600, timeout=0.5)
    newdata = ser.readline()
    if newdata[0:6] == "$GPRMC":
        newmsg = pynmea2.parse(newdata)
        latitude = newmsg.latitude
        longtitude = newmsg.longitude
    pixels = amg.pixels
    print("Reading pixels")
    maxValue = max(max(pixels))
    print("Got the highest Value")
    camera.capture(CAMERA_PATH)
    print("Captured succesfully")
    SICKNESS_RESULT = epidemicClassifier.run_from_filepath(CAMERA_PATH)
    print("Checking sickness")
    os.remove(CAMERA_PATH)
    print("Deleted the image")
    if maxValue >= COOLING_MIN:
        os.remove(CAMERA_PATH)
        print("Deleted the image")
        print("High temperature detected, " + str(datetime.time))
        radio.write(list(str(latitude)))
        print("Sent latitude")
        radio.write(list(str(longtitude)))
        print("Sent longtitude")
        radio.write(WARNING_MODES[2])
    if SICKNESS_RESULT[0][0] == EPIDEMIC_LABELS[0]:
        os.remove(CAMERA_PATH)
        print("Deleted the image")
        print("Sickness detected, " + str(datetime.time))
        radio.write(list(str(latitude)))
        print("Sent latitude, " + str(datetime.time))
        radio.write(list(str(longtitude)))
        print("Sent longtitude, " + str(datetime.time))
        radio.write(WARNING_MODES[1])


def trapMode():
    time.sleep(.1)
    ser = serial.Serial(serialPort, baudrate=9600, timeout=0.5)
    newdata = ser.readline()
    if newdata[0:6] == "$GPRMC":
        newmsg = pynmea2.parse(newdata)
        latitude = newmsg.latitude
        longtitude = newmsg.longitude
    pixels = amg.pixels
    print("Reading pixels")
    maxValue = max(max(pixels))
    print("Got the maximum value")
    if maxValue >= DETECTION_MIN:
        camera.capture(CAMERA_PATH)
        print("Image captured")
        ANIMAL_DETECTION_RESULT = animalClassifier.run_from_filepath(CAMERA_PATH)
        if ANIMAL_DETECTION_RESULT[0][0] == ANIMAL_LABELS[0]:
            print("Animal detected, " + str(datetime.time))
            radio.write(list(str(latitude)))
            print("Sent the latitude" + str(datetime.time))
            radio.write(list(str(longtitude)))
            print("Sent the longtitude" + str(datetime.time))
            radio.write(WARNING_MODES[3])
        elif ANIMAL_DETECTION_RESULT[0][0] == ANIMAL_LABELS[1]:
            print("Human detected, " + str(datetime.time))
            radio.write(list(str(latitude)))
            print("Sent the latitude" + str(datetime.time))
            radio.write(list(str(longtitude)))
            print("Sent the longtitude" + str(datetime.time))
            radio.write(WARNING_MODES[4])


def main():
    while True:
        try:
            button.when_pressed = changeStatus
        finally:
            pass
        if STATUS_TRAP:
            trapMode()
        else:
            scoutMode()
