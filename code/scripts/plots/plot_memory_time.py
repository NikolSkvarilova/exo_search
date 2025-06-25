import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])

"""Plot the time and memory usage plots.
"""

# Data
downsampling = [2, 3, 5, 10]  # step
time = [30, 12, 312 / 60, 50 / 60]  # min
memory = [12, 4, 2.5, 1.5]  # GB

# Training time
plt.figure(dpi=300)
plt.title("Training time")
plt.xlabel("Downsampling (moving average with step n)")
plt.ylabel("Time [m]")
plt.bar(downsampling, time, color="blue")
plt.tight_layout()
plt.savefig("figures/downsampling_time.pdf")
plt.close()

# Memory usage
plt.figure(dpi=300)
plt.title("RAM usage during training")
plt.xlabel("Downsampling (moving average with step n)")
plt.ylabel("RAM usage [GB]")
plt.bar(downsampling, memory, color="blue")
plt.tight_layout()
plt.savefig("figures/downsampling_memory.pdf")
plt.close()
