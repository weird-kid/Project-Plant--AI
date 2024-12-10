from time import sleep
import RPi.GPIO as gpio

in1 = 17
in2 = 22
in3 = 23
in4 = 24

''' choosing pin Numbering scheme and setting the correspoding GPIO pins as OUTPUT'''

def init():
    gpio.setmode(gpio.BCM)
    gpio.setup(in3, gpio.OUT)
    gpio.setup(in4, gpio.OUT)

  
def run():
    gpio.output(in3, False)
    gpio.output(in4, True)

def stop():
    gpio.output(in3, True)
    gpio.output(in4, True)



def water_motor():

    gpio.setwarnings(False)
    init()
    run()
    sleep(5)
    stop()

if __name__ == '__main__':
    print('Function water_motor is not called')
