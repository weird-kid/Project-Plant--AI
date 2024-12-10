import smbus2
import time

# Define the I2C bus
bus = smbus2.SMBus(1)  # 1 indicates /dev/i2c-1

# AHT10 address and commands
AHT10_ADDRESS = 0x38
AHT10_CMD_INIT = [0xE1, 0x08, 0x00]
AHT10_CMD_MEASURE = [0xAC, 0x33, 0x00]
AHT10_CMD_SOFT_RESET = [0xBA]

def aht10_init():
    bus.write_i2c_block_data(AHT10_ADDRESS, 0x00, AHT10_CMD_INIT)
    time.sleep(0.05)  # Wait for initialization

def aht10_measure():
    bus.write_i2c_block_data(AHT10_ADDRESS, 0x00, AHT10_CMD_MEASURE)
    time.sleep(0.1)  # Wait for measurement to complete
    data = bus.read_i2c_block_data(AHT10_ADDRESS, 0x00, 6)
    return data

def parse_data(data):
    humidity = ((data[1] << 12) | (data[2] << 4) | (data[3] >> 4)) / (1 << 20) * 100
    temperature = (((data[3] & 0x0F) << 16) | (data[4] << 8) | data[5]) / (1 << 20) * 200 - 50
    return humidity, temperature

def main():
    aht10_init()
    data = aht10_measure()
    humidity, temperature = parse_data(data)
    humidity = int(humidity)
    temperature = int(temperature)

    return humidity,temperature

