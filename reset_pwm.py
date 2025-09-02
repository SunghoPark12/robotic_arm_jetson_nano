import math
from pca9685 import PCA9685

pwm = PCA9685(address=0x40, busnum=1, freq=50)  # 주소 반드시 확인!
pwm.init_pwm()
