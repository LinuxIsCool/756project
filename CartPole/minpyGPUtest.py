import minpy.numpy as np
import minpy.numpy.random as random
from minpy.context import cpu, gpu
import time

n = 100

with cpu():
    x_cpu = random.rand(1024, 1024) - 0.5
    y_cpu = random.rand(1024, 1024) - 0.5

    # dry run
    for i in range(10):
        z_cpu = np.dot(x_cpu, y_cpu)
    z_cpu.asnumpy()

    # real run
    t0 = time.time()
    for i in range(n):
        z_cpu = np.dot(x_cpu, y_cpu)
    z_cpu.asnumpy()
    t1 = time.time()

with gpu(0):
    x_gpu0 = random.rand(1024, 1024) - 0.5
    y_gpu0 = random.rand(1024, 1024) - 0.5

    # dry run
    for i in range(10):
        z_gpu0 = np.dot(x_gpu0, y_gpu0)
    z_gpu0.asnumpy()

    # real run
    t2 = time.time()
    for i in range(n):
        z_gpu0 = np.dot(x_gpu0, y_gpu0)
    z_gpu0.asnumpy()
    t3 = time.time()

print("run on cpu: %.6f s/iter" % ((t1 - t0) / n))
print("run on gpu: %.6f s/iter" % ((t3 - t2) / n))
