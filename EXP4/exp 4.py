import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, bilinear, freqz, cheby1, lfilter


# Design Butterworth Filter
def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):

    # Design the analog Butterworth filter
    analog_b, analog_a = butter(
        filter_order,
        cutoff_frequency,
        analog=True,
        btype='low'
    )

    # Perform bilinear transformation
    digital_b, digital_a = bilinear(
        analog_b,
        analog_a,
        sampling_frequency
    )

    return digital_b, digital_a


# Design Chebyshev Filter
def design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
):

    # Design the analog Chebyshev Type-I filter
    analog_b, analog_a = cheby1(
        filter_order,
        ripple,
        cutoff_frequency,
        analog=True,
        btype='low'
    )

    # Perform bilinear transformation
    digital_b, digital_a = bilinear(
        analog_b,
        analog_a,
        sampling_frequency
    )

    return digital_b, digital_a


# Plot Filter Response
def plot_filter_response(
    digital_b,
    digital_a,
    sampling_frequency,
    filter_name
):

    # Frequency response
    frequency, magnitude_response = freqz(
        digital_b,
        digital_a,
        fs=sampling_frequency
    )

    # Magnitude response
    plt.figure(figsize=(10, 6))

    plt.plot(
        frequency,
        np.abs(magnitude_response)
    )

    plt.title(filter_name + " Magnitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.show()


    # Impulse response
    impulse_input = np.zeros(100)
    impulse_input[0] = 1

    impulse_response = lfilter(
        digital_b,
        digital_a,
        impulse_input
    )

    # Plot impulse response
    plt.figure(figsize=(10, 6))

    plt.stem(
        range(len(impulse_response)),
        impulse_response
    )

    plt.title(filter_name + " Impulse Response")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.show()


# Filter specifications
filter_order = 4
cutoff_frequency = 1000
sampling_frequency = 8000
ripple = 0.5


# -------------------------------
# Butterworth Filter
# -------------------------------

digital_b, digital_a = design_butterworth_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency
)

print("Butterworth Filter Coefficients")
print("Numerator coefficients:")
print(digital_b)

print("Denominator coefficients:")
print(digital_a)

plot_filter_response(
    digital_b,
    digital_a,
    sampling_frequency,
    "Butterworth"
)


# -------------------------------
# Chebyshev Filter
# -------------------------------

digital_b, digital_a = design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
)

print("\nChebyshev Filter Coefficients")
print("Numerator coefficients:")
print(digital_b)

print("Denominator coefficients:")
print(digital_a)

plot_filter_response(
    digital_b,
    digital_a,
    sampling_frequency,
    "Chebyshev"
)


# Save Chebyshev filter coefficients
filter_path = "filter_coefficients.txt"

np.savetxt(
    filter_path,
    np.vstack((digital_b, digital_a)),
    delimiter=","
)

print("\nFilter coefficients saved at:", filter_path)