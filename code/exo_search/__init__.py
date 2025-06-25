import matplotlib.pyplot as plt
import sys

# Check Python version
if sys.version_info[0] != 3 or sys.version_info[1] != 10:
    raise SystemExit("Bad Python version. Please ensure Python 3.10.")

# Try using scienceplots
try:
    import scienceplots

    plt.style.use(["science", "no-latex"])
except:
    pass

# Setup for matplotlib plots
plt.rcParams["figure.dpi"] = 300
