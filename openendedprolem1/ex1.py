import numpy as np
import matplotlib.pyplot as plt
#1. x[n] = 2 for even n 3 for odd n
n = np.arange(-3, 7)
x1 = np.where(n % 2 == 0, 2, 3)
plt.figure(figsize=(5,3))
plt.stem(n, x1)
plt.title("Signal 1")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
# 2.Staircase signal
n2 = np.arange(0, 6)
x2 = [1, 2, 3, 3, 2, 1]
plt.figure(figsize=(5,3))
plt.step(n2, x2, where='post')
plt.title("Signal 2")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
# 3.Piecewise signal
n3 = np.linspace(-2, 2, 400)
x3 = np.piecewise(
    n3,
    [n3 < -1,
     (n3 >= -1) & (n3 <= 1),
     (n3 > 1) & (n3 <= 2)],
    [-2,
     lambda n: 2*n,
     2]
)
plt.figure(figsize=(5,3))
plt.plot(n3, x3, linewidth=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("Signal 3")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
# 4.x[n] = 4u(n) - u(n-3) - 5u(n-7)
n4 = np.arange(-2, 11)
u = lambda n: np.where(n >= 0, 1, 0)
x4 = 4*u(n4) - u(n4-3) - 5*u(n4-7)
plt.figure(figsize=(5,3))
plt.stem(n4, x4)
plt.title("Signal 4")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
# 5. x[n] = δ(n) + 3δ(n-1) + 5δ(n+1)
n5 = np.arange(-3, 4)
delta = lambda n: np.where(n == 0, 1, 0)
x5 = delta(n5) + 3*delta(n5-1) + 5*delta(n5+1)
plt.figure(figsize=(5,3))
plt.stem(n5, x5)
plt.title("Signal 5")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()