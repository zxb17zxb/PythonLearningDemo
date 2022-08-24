import matplotlib.pyplot as plt
import random
import pandas as pd
from plotly.offline import plot
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import signal
from alphawaves.dataset import AlphaWaves
import numpy as np
import random
import pandas as pd
from plotly.offline import plot

# Create a random list of values.
vals = [random.randint(0, 10) for _ in range(100)]

# Create a test DataFrame.
df = pd.DataFrame({'count': vals})
df['cumsum'] = df['count'].cumsum()