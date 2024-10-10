import numpy as np
import matplotlib.pyplot as plt
import random

test = np.ones((32,32))
test2 = np.zeros((32,32))

def generate_random_variables():
    # 生成随机变量 a、b、c、d
    a = random.randint(0, 30)
    b = random.randint(0, 30)
    c = random.randint(1, 31 - a)
    d = random.randint(1, 31 - b)

    return a, b, c, d

def randomcutt(pressure,recordings):

    pressure_out = np.zeros_like(pressure)
    len = pressure.shape[0]
    for i in range(len):
        basic = pressure[i].copy()
        recording = int(recordings[i])

        ind_record = np.argwhere(np.squeeze(recordings)==recording)

        pair = np.squeeze(pressure[ind_record[random.randint(0,ind_record.shape[0]-1)]])

        a, b, c, d = generate_random_variables()

        # 处理边界情况，确保不越界
        c = min(c, 31 - a)
        d = min(d, 31 - b)
        e = random.randint(-a,0) if a+c == 31 else random.randint(-a, 31 - (a + c))
        f = random.randint(-b,0) if b+d == 31 else random.randint(-b, 31 - (b + d))
        m = np.random.beta(1.0, 1.0)

        for xx in range(c):
            for yy in range(d):
                basic[a+xx][b+yy] = m*pair[a+e+xx][b+f+yy] + (1-m)*basic[a+xx][b+yy]

        pressure_out[i] = basic


    return pressure_out
