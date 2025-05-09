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
    freq2 = 130 

    # Amplitudes (relative units)
    amp1 = 1.3   # PSR J0437−4715
    amp2 = 1.0 

    # Background noise parameters
    background_mean = 10.0
    background_std = 2.0

    # Generate time array
    time_array = np.linspace(0, N_total * time_step, N_total)

    # Generate the signals
    angular_freq1 = 2 * np.pi * freq1
    angular_freq2 = 2 * np.pi * freq2

    # Calculate signal
    signal = abs(amp1 * np.sin(angular_freq1 * time_array)) + abs(amp2 * np.cos(angular_freq2 * time_array))
    # Add noise and background
    noise = np.random.normal(0, 1, N_total)
    background = np.random.normal(background_mean, background_std, N_total)

    # adds to intensity array
    intensity_array = np.abs(signal) + noise + background

    #plots the data for verification
    plt.plot(time_array, intensity_array)
    plt.title('Synthetic Pulsar Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (arbitrary units)')
    plt.savefig('synthetic_pulsar_data.png')
    plt.show()
    return intensity_array, time_array

def estimate_frequency_uncertainty(frequencies, intensity, peak_index):
    """Estimate frequency uncertainty as FWHM of the peak."""

    #find the half maximum
    half_max = intensity[peak_index] / 2
    left = peak_index

    # Find the left edge of the peak
    while left > 0 and intensity[left] > half_max:
        left -= 1
    right = peak_index

    # Find the right edge of the peak
    while right < len(intensity) - 1 and intensity[right] > half_max:
        right += 1
    fwhm = frequencies[right] - frequencies[left]

    # Estimate frequency uncertainty as half of the FWHM
    freq_uncertainty = fwhm / 2
    return freq_uncertainty

def calculate_snr(intensity, peak_index, noise_width=10):
    """Estimate SNR by comparing peak height to local noise."""

    #finds the peak height
    peak_height = intensity[peak_index]
    
    #finds the left and right with noise width
    left = max(0, peak_index - noise_width)
    right = min(len(intensity), peak_index + noise_width)

    # Calculate noise as the average of the surrounding values
    noise = np.concatenate([intensity[left:peak_index], intensity[peak_index+1:right]])
    noise_mean = np.mean(noise)

    # Calculates the noise standard deviation
    noise_std = np.std(noise)

    # Calculate SNR
    snr = (peak_height - noise_mean) / noise_std
    return snr

def print_data(frequencies, intensity, peaks_indices):
    """Prints the detected peaks and their properties."""

    #prints frequencies and intensities, snr and uncertainty
    print(f"{'Peak #':<6} {'Frequency (Hz)':<15} {'Uncertainty (Hz)':<18} {'SNR':<8}")
    snr_array = np.zeros(len(peaks_indices))
    for i, index in enumerate(peaks_indices):
        freq_unc = estimate_frequency_uncertainty(frequencies, intensity, index)
        snr = calculate_snr(intensity, index)
        snr_array[i] = snr
        print(f"{i+1:<6} {frequencies[index]:<15.3f} {freq_unc:<18.3e} {snr:<8.2f}")

    # Calculate and print average SNR
    print(f"Average SNR: {np.mean(snr_array):.2f} ± {np.std(snr_array):.2f}")

def phase_binning(data, period, number_of_bins=10, sample_interval=0.004):
    """Perform phase binning on the data using the given period."""
    # initlaises variables
    total_length = len(data)
    bins = np.zeros(number_of_bins)
    bin_counts = np.zeros(number_of_bins)

    #finds current bin and segrigates data into bins
    for index in range(total_length):
        time = index * sample_interval
        phase = (time % period) / period
        bin_index = int(phase * number_of_bins)

        # enfoces bin bounds
        if bin_index == number_of_bins:
            bin_index = 0
        # adds data to the bin
        bins[bin_index] += data[index]
        bin_counts[bin_index] += 1
    
    mask = bin_counts > 0
    bins[mask] /= bin_counts[mask]
    return bins, bin_counts


def find_period(data, period_guess, delta=0.028, steps=100, num_bins=10, sample_interval=0.004):
    """Refine the period estimate using phase binning."""
    #initialies array with all "guesses" and variation array
    periods = np.linspace(period_guess - delta, period_guess + delta, steps)
    variations = np.zeros(steps)

    #computes for each guess
    for index, period in enumerate(periods):
        bins, _ = phase_binning(data, period, num_bins, sample_interval)
        variations[index] = np.max(bins) - np.min(bins)

    #finds the best period and its variation
    best_index = np.argmax(variations)
    best_period = periods[best_index]
    max_variation = variations[best_index]

    # Print the best period and its variation
    print(f"Best period: {best_period:.6f} s with variation: {max_variation:.6f}")

    # Estimate period uncertainty as FWHM of the variation curve should have used objects but this was faster as its not much code
    
    #see comments in estimate_frequency_uncertainty
    half_max = max_variation / 2
    left = best_index
    while left > 0 and variations[left] < half_max:
        left -= 1
    right = best_index
    while right < len(variations) - 1 and variations[right] < half_max:
        right += 1
    fwhm = periods[right] - periods[left]
    period_uncertainty = fwhm / 2

    # Plot the variation curve
    plt.figure(figsize=(10, 6))
    plt.plot(periods, variations)

    #highlights the best period
    plt.axvline(x=best_period, color='r', linestyle='--', label=f'Best Period: {best_period:.6f}s',alpha=0.5)
    plt.title('Period Refinement via Phase Binning')
    plt.xlabel('Period (s)')
    plt.ylabel('Bin-to-bin Variation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('period_variation_with_phase_bining.png')
    plt.show()

    # Print the best period and its uncertainty again 
    print(f"Best period: {best_period:.6f} s ± {period_uncertainty:.2e} s")
    return best_period, period_uncertainty

def plot_bin(data, period, number_of_bins=10, sample_interval=0.004):
    """plots binned data"""

    # Perform phase binning for a given period
    bins, count = phase_binning(data, period, number_of_bins, sample_interval)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(len(bins)), bins, capsize=5)
    ax.set_title(f'Pulsar Waveform (Period: {period:.6f}s)')
    ax.set_xlabel('Phase Bin')
    ax.set_ylabel('Average Intensity')
    ax.set_xticks(np.arange(number_of_bins))
    ax.grid(True, axis='y')
    fig.tight_layout()
    fig.savefig('best_bin_plot.png')
    plt.show()
    return

def plot_fourier_transform(data, sample_interval=0.004):
    no_data_points = len(data)

    # Compute the Fourier transform
    intensity, frequencies = compute_fourier(data, no_data_points, sample_interval)

    # Plot the power spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, intensity, label='Power Spectrum')
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (Arbitrary Units)')
    plt.legend()
    plt.savefig('power_spectrum.png')
    plt.show()

    # Detect peaks in the power spectrum
    peaks_indices = detect_peaks(intensity) # peaks are based on an arbitrary threshold see detect_peaks function

    # Print the detected peaks
    plt.figure(figsize=(10, 4))
    #plots the power spectrum
    plt.plot(frequencies, intensity, label='Power Spectrum')

    #plots the detected peaks
    plt.scatter(frequencies[peaks_indices], intensity[peaks_indices], color='red', label='Detected Peaks', zorder=5)
    plt.title('Power Spectrum with Detected Peaks')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.savefig('power_spectrum_peaks.png')
    plt.show()
    return frequencies, intensity, peaks_indices

def main(pulsar_data):
    """Main run function that takes any data and attempts to find a period of rotation"""
    
    #Runs the Fourier transform funtion and plots the data
    frequencies, intensity, peaks_indices = plot_fourier_transform(pulsar_data)

    #prints the data
    print_data(frequencies, intensity, peaks_indices)

    #Ues the findemetal frequancey as a guess for the period (not other guesses were also considered)
    fundamental_freq = frequencies[peaks_indices[0]]

    #converts the frequency to a period
    period_guess = 1 / fundamental_freq

    #find the best period using the phase binning
    best_period, period_uncertainty = find_period(pulsar_data, period_guess)
    plot_bin(pulsar_data, best_period)

if __name__ == "__main__":
    #print("Synthetic pulsar data")
    #data = gen_synthetic_data(N_total=256, time_step=0.004)[0]
    #main(data)

    print("real data")
    pulsar_data = load_pulsar_data('pulsar.dat')
    plt.plot(np.linspace(0, len(pulsar_data)*0.004, len(pulsar_data)), pulsar_data)
    plt.title('Pulsar Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (arbitrary units)')
    plt.savefig('pulsar_data_plot.png')
    plt.show()
    main(pulsar_data)
    print("Completed")
    
