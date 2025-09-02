import smbus
import time

class PCA9685:
    # PCA9685 레지스터
    __MODE1 = 0x00
    __PRESCALE = 0xFE

    def __init__(self, address=0x40, busnum=1, freq=50):
        self.address = address
        self.bus = smbus.SMBus(busnum)
        self.set_pwm_freq(freq)

    def set_pwm_freq(self, freq):
        prescale_val = int(round(25000000.0 / (4096.0 * freq) - 1))
        oldmode = self.bus.read_byte_data(self.address, self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # sleep
        self.bus.write_byte_data(self.address, self.__MODE1, newmode)
        self.bus.write_byte_data(self.address, self.__PRESCALE, prescale_val)
        self.bus.write_byte_data(self.address, self.__MODE1, oldmode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.__MODE1, oldmode | 0x80)

    def set_pwm(self, channel, on, off):
        # 0x06~0x09: channel 0, 0x0A~0x0D: channel 1, ... 4byte씩
        base = 0x06 + 4 * channel
        self.bus.write_byte_data(self.address, base, on & 0xFF)
        self.bus.write_byte_data(self.address, base + 1, on >> 8)
        self.bus.write_byte_data(self.address, base + 2, off & 0xFF)
        self.bus.write_byte_data(self.address, base + 3, off >> 8)

    def init_pwm(self):
        for ch in range(16):
           self.set_pwm(ch,0,0)



