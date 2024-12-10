import RPi.GPIO 
from time import sleep

relay = 16

print('Turning the relay on and off for 5 seconds')
print('Press q to exit')


RPi.GPIO.setmode(RPi.GPIO.BCM)
RPi.GPIO.setup(relay, RPi.GPIO.OUT)


''' Relay input pin ---> Active Low
    We want relay to be defaultly on
    SO High (true) --> relay on --> led on
    if  low      ----> relay off --> led off

'''
def relay_on():
    RPi.GPIO.output(relay, True)
    sleep(10) 
    RPi.GPIO.output(relay, False)
    sleep(0.1)




