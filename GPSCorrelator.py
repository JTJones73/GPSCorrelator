import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sci


# Constants
SAMPLE_RATE = 4.092e6  # 4 MHz sampling rate
CODE_RATE = 1.023e6  # 1.023 MHz GPS C/A code rate
CODE_LENGTH = 1023  # Length of C/A code

def generate_prn():
    """Generates 1023-length C/A Codes for GPS PRNs 1-37."""
    tap = np.array([
        [2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9], [3, 10], [2, 3],
        [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [1, 4], [2, 5], [3, 6], [4, 7],
        [5, 8], [6, 9], [1, 3], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 6], [2, 7],
        [3, 8], [4, 9], [5, 10], [4, 10], [1, 7], [2, 8], [4, 10]
    ])
    
    # G1 LFSR parameters
    s = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    L = 2**10 - 1  # Length of one C/A code sequence (1023)

    # G2 LFSR parameters
    t = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    
    # Prepare output matrix for all 37 PRNs
    g = np.zeros((37, L), dtype=int)
    
    # Generate C/A Code sequences for all PRNs
    for prn in range(1, 38):  # PRNs 1 to 37
        tap_sel = tap[prn - 1] - 1  # Adjust for Python's zero-based indexing
        g1 = np.ones(10, dtype=int)  # Re-initialize G1 for each PRN
        q = np.ones(10, dtype=int)  # Re-initialize G2 for each PRN
        
        for inc in range(L):
            g2 = np.mod(np.sum(q[tap_sel]), 2)
            g[prn - 1, inc] = np.mod(g1[-1] + g2, 2)
            g1 = np.roll(g1, -1)
            g1[0] = np.mod(np.sum(g1 * s), 2)
            q = np.roll(q, -1)
            q[0] = np.mod(np.sum(q * t), 2)

    return g * 2 - 1

prn_code = generate_prn()
def load_iq_samples(filename, num_samples=16000):
    iq_data = np.fromfile(filename, np.int16).astype(np.float64).view(np.complex128)[:num_samples]
    iq_data = sci.resample(iq_data, 16368)
    print(f"[DEBUG] Loaded {len(iq_data)} IQ samples")  # Confirm sample count
    return iq_data

def resample_prn_code(prn_code, target_length):
    samples_per_chip = int(SAMPLE_RATE / CODE_RATE)
    resampled_code = np.repeat(prn_code, samples_per_chip)
    return resampled_code[:target_length]

def gps_correlator(iq_samples, prn_id, doppler_shift=0):
    """Correlates the IQ samples with the PRN code of the specified satellite and applies Doppler shift."""
    code_samples = resample_prn_code(prn_code[prn_id], len(iq_samples))

    #apply doppler shift
    time = np.arange(len(iq_samples)) / SAMPLE_RATE
    iq_shifted = iq_samples * np.exp(-1j * 2 * np.pi * doppler_shift * time)

    #convert to phase  OR not
    #for i in range(len(iq_shifted)):
    #   iq_shifted[i] = np.complex128(np.arctan2(iq_shifted[i].imag, iq_shifted[i].real))
    correlation = correlate(iq_shifted, code_samples, mode='valid')

    #max_corr_value = np.max(np.abs(correlation))
    #print(f"[DEBUG] Max correlation for PRN {prn_id}: {max_corr_value}")  # Peak correlation value
    return np.abs(correlation)

def plot_correlation(correlation, prn_id, plot_type='2d'):
    if plot_type == '3d':
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Create 3d plot
        x = np.arange(correlation.shape[1])  # Sample indices
        y = np.linspace(-1250 * 4, 1250 * 4, correlation.shape[0])  # Doppler shifts
        X, Y = np.meshgrid(x, y)
        Z = np.abs(correlation)  # Correlation magnitudes

        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"3D GPS Correlation Result for PRN ID {prn_id}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Doppler Shift (Hz)")
        ax.set_zlabel("Correlation Magnitude")
    else:
        #2d plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(correlation).max(axis=0))  # Plot max correlation across Doppler bins
        plt.title(f"2D GPS Correlation Result for PRN ID {prn_id}")
        plt.xlabel("Sample Index")
        plt.ylabel("Max Correlation Magnitude")

    plt.show()

iq_samples = load_iq_samples('GPS-L1-2022-03-27.sigmf-data')
for i in range(1,32):
    print(f"\n[INFO] Running correlation for PRN ID {i}")
    correlation = np.zeros((5000, 12277), np.complex128)
    for j in range(5000):
        doppler_shift = (j - 2500) * 2  #doppler shift in Hz
        correlation[j] = gps_correlator(iq_samples, i, doppler_shift=doppler_shift)
    
    print("Max correlation magnitude:", np.max(np.abs(correlation)))
    #plot_correlation(correlation, i, plot_type='3d')