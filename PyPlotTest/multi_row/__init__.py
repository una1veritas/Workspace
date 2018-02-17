import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-np.pi, np.pi, 1000)

x1 = np.sin(2*t)
x2 = np.cos(2*t)

fig, axes = plt.subplots(ncols=1,nrows=2, figsize=(10,4))

axes[0].plot(t, x1, linewidth=2)
axes[0].set_title('sin')
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
axes[0].set_xlim(-np.pi, np.pi)
axes[0].grid(True)

axes[1].plot(t, x2, linewidth=2)
axes[1].set_title('cos')
axes[1].set_xlabel('t')
axes[1].set_ylabel('x')
axes[1].set_xlim(-np.pi, np.pi)
axes[1].grid(True)

fig.show()
input('?')