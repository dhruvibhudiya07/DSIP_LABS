from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fft import fft, ifft

# =====================================
# Load WAV File
# =====================================
#rate, samples = wavfile.read("audio2.wav")
rate, samples = wavfile.read(r"D:\DSIP_LABs\EXP2\audio5.wav")
# Convert stereo to mono if needed
if len(samples.shape) == 2:
    samples = np.mean(samples, axis=1)

samples = samples.astype(np.float32)

# Normalize
samples = samples / np.max(np.abs(samples))

# =====================================
# Multiple Impulse Responses
# =====================================
impulse_responses = {
    "IR1": np.array([1, 0, 1, 0, 1], dtype=np.float32),
    "IR2": np.array([0.5, 1, 0.5], dtype=np.float32),
    "IR3": np.array([1, -0.5, 0.25], dtype=np.float32)
}

# =====================================
# Process Each Impulse Response
# =====================================
for name, kernel in impulse_responses.items():

    print(f"\nProcessing {name}...")

    # -----------------------------
    # Convolution
    # -----------------------------
    convolved = convolve(samples, kernel, mode="same")

    # Normalize
    convolved = convolved / np.max(np.abs(convolved))

    # Save Convolved Audio
    wavfile.write(
        f"{name}_Convolved.wav",
        rate,
        (convolved * 32767).astype(np.int16)
    )

    # -----------------------------
    # Inverse Filtering
    # -----------------------------
    kernel_pad = np.zeros(len(convolved))
    kernel_pad[:len(kernel)] = kernel

    H = fft(kernel_pad)

    # Avoid division by zero
    epsilon = 1e-8
    H[np.abs(H) < epsilon] = epsilon

    Y = fft(convolved)

    recovered = np.real(ifft(Y / H))

    # Normalize
    recovered = recovered / np.max(np.abs(recovered))

    # Save Recovered Audio
    wavfile.write(
        f"{name}_Recovered.wav",
        rate,
        (recovered * 32767).astype(np.int16)
    )

    # -----------------------------
    # Plot Results
    # -----------------------------
    plt.figure(figsize=(12,8))

    plt.subplot(3,1,1)
    plt.plot(samples)
    plt.title("Original Audio")

    plt.subplot(3,1,2)
    plt.plot(convolved)
    plt.title(f"{name} - Convolved Audio")

    plt.subplot(3,1,3)
    plt.plot(recovered)
    plt.title(f"{name} - After Inverse Filtering")

    plt.tight_layout()
    plt.show()

print("\nProcessing Completed Successfully!")

print("\nGenerated Files:")
for ir in impulse_responses.keys():
    print(f"{ir}_Convolved.wav")
    print(f"{ir}_Recovered.wav")