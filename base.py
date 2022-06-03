import smtplib
from lib_nrf24 import NRF24
from email.message import EmailMessage
from RPi.GPIO import GPIO
import spidev
import time

# 1 = Sickness
# 2 = High Temperature
# 3 = Animal Detected
# 4 = Human Detected

WARNING_MODES = ['1', '2', '3', '4']
PIPES = [[0xe7, 0xe7, 0xe7, 0xe7, 0xe7]]

message = EmailMessage()

message['Subject'] = 'ALERT'
message['From'] = 'OKS'
message['To'] = 'ataberk.ozsasal@gmail.com'

GPIO.setmode(GPIO.BCM)
radio = NRF24(GPIO, spidev.Spidev())
radio.begin(0, 17)
radio.setRetries(15, 15)
radio.setPayloadSize(16)
radio.setChannel(0x60)
radio.setDataRate(NRF24.BR_250KBPS)
radio.setPALevel(NRF24.PA_MIN)

radio.setAutoAck(True)
radio.enableDynamicPayloads()
radio.enableAckPayload()

radio.openReadingPipe(1, PIPES[0])
radio.startListening()

while True:

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login("ataberkozsasalprojetest@gmail.com", "ProjeStrabon")

    while not radio.available(0):
        time.sleep(1 / 100)
    receivedMessage = []
    radio.read(receivedMessage, radio.getDynamicPayloadSize())
    latitude = str(receivedMessage)
    print("Received latitude")
    receivedMessage = []
    radio.read(receivedMessage, radio.getDynamicPayloadSize())
    longtitude = str(receivedMessage)
    print("Received longtitude")
    receivedMessage = []
    radio.read(receivedMessage, radio.getDynamicPayloadSize())
    print("Received status")
    if chr(receivedMessage[0]) == WARNING_MODES[0]:
        message.set_content("Sickness detected in the following coordinates: \n" +
                            "latitude: " + latitude + "\n" +
                            "longtitude: " + longtitude)
        server.send_message(message)
        print("Sent the message!")
        server.quit()
    elif chr(receivedMessage[0]) == WARNING_MODES[1]:
        message.set_content("High Temperature detected in the following coordinates: \n" +
                            "latitude: " + latitude + "\n" +
                            "longtitude: " + longtitude)
        server.send_message(message)
        print("Sent the message!")
        server.quit()
    elif chr(receivedMessage[0]) == WARNING_MODES[2]:
        message.set_content("Animal detected in the following coordinates: \n" +
                            "latitude: " + latitude + "\n" +
                            "longtitude: " + longtitude)
        server.send_message(message)
        print("Sent the message!")
        server.quit()
    elif chr(receivedMessage[0]) == WARNING_MODES[3]:
        message.set_content("Human detected in the following coordinates: \n" +
                            "latitude: " + latitude + "\n" +
                            "longtitude: " + longtitude)
        server.send_message(message)
        print("Sent the message!")
        server.quit()
    else:
        continue

