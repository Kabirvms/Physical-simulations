import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_pulsar_data(filename):
    """Loads pulsar data from a file (ASCII, one value per line)."""
    return np.loadtxt(filename)

def compute_fourier(x_data, N_total, sample_interval=0.004):
    """Computes the Fourier transform using recurrence relations."""
    # Initialize arrays
    A_k = np.zeros(N_total)
    B_k = np.zeros(N_total)
    intensity = np.zeros(N_total)

    # Compute Fourier coefficients using recurrence relations
    for k in range(1, N_total):
        theta = 2 * np.pi * k / N_total
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        U_n = np.zeros(N_total + 2)


        for n in range(N_total - 1, -1, -1):
            U_n[n] = x_data[n] + 2 * cos_theta * U_n[n + 1] - U_n[n + 2]
        
        # Compute and normalise the coefficients
        A_k[k] = (U_n[0] - U_n[1] * cos_theta) / N_total
        B_k[k] = (U_n[1] * sin_theta) / N_total
        intensity[k] = A_k[k]**2 + B_k[k]**2
    frequency_array = np.arange(N_total) / (N_total * sample_interval)
    return intensity, frequency_array

def detect_peaks(intensity, height_fraction=0.1, distance_fraction=0.05):
    """Detects peaks in the intensity spectrum."""
    # Use scipy's find_peaks to detect peaks
    max_intensity = np.max(intensity)
    min_height = max_intensity * height_fraction
    min_distance = int(len(intensity) * distance_fraction)
    peaks_indices, _ = find_peaks(intensity, height=min_height, distance=min_distance)
    return peaks_indices

def gen_synthetic_data(N_total=256, time_step=0.001):



    """Generates synthetic pulsar data see the report for details on parameters """
    # Frequencies in Hz
    freq1 = 100  # PSR J0437−4715
    freq2 = 130 # PSR B1937+21

    # Amplitudes (relative units)
    amp1 = 1.3   # PSR J0437−4715
    amp2 = 1.0  # PSR B1937+21

    # Background noise parameters
    background_mean = 10.0
    background_std = 2.0

    # Generate time array
    time_array = np.linspace(0, N_total * time_step, N_total)

    # Generate the signals
    angular_freq1 = 2 * np.pi * freq1
    angular_freq2 = 2 * np.pi * freq2

    # Calculate signal, add noise and background
    signal = abs(amp1 * np.sin(angular_freq1 * time_array)) + abs(amp2 * np.cos(angular_freq2 * time_array))
    noise = np.random.normal(0, 1, N_total)
    background = np.random.normal(background_mean, background_std, N_total)
    intensity_array = np.abs(signal) + noise + background

    plt.plot(time_array, intensity_array)
    plt.title('Synthetic Pulsar Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (arbitrary units)')
    plt.savefig('synthetic_pulsar_data.png')
    plt.show()
    return intensity_array, time_array

def estimate_frequency_uncertainty(frequencies, intensity, peak_index):
    """Estimate frequency uncertainty as FWHM of the peak."""

    # Find the peak frequency and its FWHM

    half_max = intensity[peak_index] / 2
    left = peak_index


    while left > 0 and intensity[left] > half_max:
        left -= 1
    right = peak_index
    while right < len(intensity) - 1 and intensity[right] > half_max:
        right += 1
    fwhm = frequencies[right] - frequencies[left]

    # Estimate frequency uncertainty as half of the FWHM
    freq_uncertainty = fwhm / 2
    return freq_uncertainty

def calculate_snr(intensity, peak_index, noise_width=10):
    """Estimate SNR by comparing peak height to local noise."""
    peak_height = intensity[peak_index]
    left = max(0, peak_index - noise_width)
    right = min(len(intensity), peak_index + noise_width)
    noise = np.concatenate([intensity[left:peak_index], intensity[peak_index+1:right]])
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)
    snr = (peak_height - noise_mean) / noise_std
    return snr

def summarize_peaks(frequencies, intensity, peaks_indices):
    print(f"{'Peak #':<6} {'Frequency (Hz)':<15} {'Uncertainty (Hz)':<18} {'SNR':<8}")
    snr_array = np.zeros(len(peaks_indices))
    for i, index in enumerate(peaks_indices):
        freq_unc = estimate_frequency_uncertainty(frequencies, intensity, index)
        snr = calculate_snr(intensity, index)
        snr_array[i] = snr

        print(f"{i+1:<6} {frequencies[index]:<15.3f} {freq_unc:<18.3e} {snr:<8.2f}")
    print(f"Average SNR: {np.mean(snr_array):.2f} ± {np.std(snr_array):.2f}")

def phase_binning(data, period, number_of_bins=10, sample_interval=0.004):
    """Perform phase binning on the data using the given period."""
    total_length = len(data)
    bins = np.zeros(number_of_bins)
    bin_counts = np.zeros(number_of_bins)
    for index in range(total_length):
        time = index * sample_interval
        phase = (time % period) / period
        bin_index = int(phase * number_of_bins)
        if bin_index == number_of_bins:
            bin_index = 0
        bins[bin_index] += data[index]
        bin_counts[bin_index] += 1
    mask = bin_counts > 0
    bins[mask] /= bin_counts[mask]
    return bins, bin_counts


def find_period(data, period_guess, delta=0.028, steps=100, num_bins=100, sample_interval=0.004):
    """Refine the period estimate using phase binning."""
    periods = np.linspace(period_guess - delta, period_guess + delta, steps)
    variations = np.zeros(steps)
    for index, period in enumerate(periods):
        bins, _ = phase_binning(data, period, num_bins, sample_interval)
        variations[index] = np.max(bins) - np.min(bins)
    best_index = np.argmax(variations)
    best_period = periods[best_index]
    max_variation = variations[best_index]
    # Estimate period uncertainty as FWHM of the variation curve
    half_max = max_variation / 2
    left = best_index
    while left > 0 and variations[left] < half_max:
        left -= 1
    right = best_index
    while right < len(variations) - 1 and variations[right] < half_max:
        right += 1
    fwhm = periods[right] - periods[left]
    period_uncertainty = fwhm / 2
    plt.figure(figsize=(10, 6))
    plt.plot(periods, variations)
    plt.axvline(x=best_period, color='r', linestyle='--', label=f'Best Period: {best_period:.6f}s',alpha=0.5)
    plt.title('Period Refinement via Phase Binning')
    plt.xlabel('Period (s)')
    plt.ylabel('Bin-to-bin Variation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('period_refinement.png')
    plt.show()
    print(f"Best period: {best_period:.6f} s ± {period_uncertainty:.2e} s")
    return best_period, period_uncertainty

def plot_bin(data, period, number_of_bins=10, sample_interval=0.004):
    bins, count = phase_binning(data, period, number_of_bins, sample_interval)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(bins)), bins, capsize=5)
    ax.set_title(f'Pulsar Waveform (Period: {period:.6f}s)')
    ax.set_xlabel('Phase Bin')
    ax.set_ylabel('Average Intensity')
    ax.set_xticks(np.arange(number_of_bins))
    ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig('pulsar_waveform_with_error.png')
    plt.show()
    return

def plot_fourier_transform(data, sample_interval=0.004):
    no_data_points = len(data)
    intensity, frequencies = compute_fourier(data, no_data_points, sample_interval)
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, intensity, label='Power Spectrum')
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (Arbitrary Units)')
    plt.legend()
    plt.savefig('power_spectrum.png')
    plt.show()
    peaks_indices = detect_peaks(intensity)
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, intensity, label='Power Spectrum')
    plt.scatter(frequencies[peaks_indices], intensity[peaks_indices], color='red', label='Detected Peaks', zorder=5)
    plt.title('Power Spectrum with Detected Peaks')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()

    plt.savefig('power_spectrum_peaks.png')
    plt.show()
    return frequencies, intensity, peaks_indices

def main(pulsar_data):
    frequencies, intensity, peaks_indices = plot_fourier_transform(pulsar_data)
    summarize_peaks(frequencies, intensity, peaks_indices)
    fundamental_freq = frequencies[peaks_indices[0]]
    period_guess = 1 / fundamental_freq
    best_period, period_uncertainty = find_period(pulsar_data, period_guess)
    plot_bin(pulsar_data, best_period)

if __name__ == "__main__":
    #print("Generating synthetic pulsar data...")
    #print("Running analysis on the synthetic data...")
    #data = gen_synthetic_data(N_total=256, time_step=0.004)[0]
    #main(data)

    print("Analysing real data")
    pulsar_data = load_pulsar_data('pulsar.dat')
    plt.plot(np.linspace(0, len(pulsar_data)*0.004, len(pulsar_data)), pulsar_data)
    plt.title('Pulsar Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (arbitrary units)')
    plt.savefig('pulsar_data_plot.png')
    plt.show()
    main(pulsar_data)
    print("End of script.")
    
