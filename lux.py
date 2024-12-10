import time
import smbus

# Pi has 2 buses, one for ARM and other for GPU 
bus = smbus.SMBus(1)

# TSL2561 address -----> 0x39(57)
# Select control register, 0x00(00) with command register, 0x80(128)
#		0x03(03)	Power ON mode
bus.write_byte_data(0x39, 0x00 | 0x80, 0x03)

# TSL2561 address -----> 0x39(57)
# Select timing register, 0x01(01) with command register, 0x80(128)
#		0x02(02)	Nominal integration time = 402ms
bus.write_byte_data(0x39, 0x01 | 0x80, 0x02)

time.sleep(0.5)

while True:
    # Read data back from 0x0C(12) with command register, 0x80(128), 2 bytes
    # ch0 LSB, ch0 MSB
    data = bus.read_i2c_block_data(0x39, 0x0C | 0x80, 2)

    # Read data back from 0x0E(14) with command register, 0x80(128), 2 bytes
    # ch1 LSB, ch1 MSB
    data1 = bus.read_i2c_block_data(0x39, 0x0E | 0x80, 2)

    # So, basically they are left shifting data[1] (8 byte MSB) and then adding it to LSB (8 bytes) data[0] 
    ch0 = data[1] * 256 + data[0]
    ch1 = data1[1] * 256 + data1[0]

    # Output data to screen
    print (f'Full Spectrum(IR + Visible) :{ch0} lux')
    print (f'Infrared Value :{ch1} lux')
    print (f'Visible Value : lux {(ch0 - ch1)}')
