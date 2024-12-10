
from gpiozero import MCP3208
from time import sleep


def Temp_humid():
    moist = MCP3208(channel=1)
    return  int(moist.value*100)
