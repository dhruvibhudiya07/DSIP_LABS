import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import correlate
# 1. FILE PATHS
base_dir = os.path.dirname(os.path.abspath(__file__))
# YOUR ACTUAL FILE NAMES
original_file = os.path.join(base_dir, "originl1.mp3")
karaoke_file = os.path.join(base_dir, "karaoke.mp3")
different_file = os.path.join(base_dir, "different.mp3")
# 2. CHECK FILE
for file in [original_file, karaoke_file, different_file]:
    if not os.path.exists(file):
        print("File not found:", file)
        raise FileNotFoundError(file)
print("All three audio files found successfully!")
# =========================================================
# 3. LOAD AUDIO FILES
# =========================================================
sr = 22050
original, _ = librosa.load(
    original_file,
    sr=sr,
    mono=True
)
karaoke, _ = librosa.load(
    karaoke_file,
    sr=sr,
    mono=True
)
different, _ = librosa.load(
    different_file,
    sr=sr,
    mono=True
)
print("Audio files loaded successfully!")
# =========================================================
# 4. USE FIRST 15 SECONDS
# =========================================================

duration = 15

samples = sr * duration

original = original[:samples]

karaoke = karaoke[:samples]

different = different[:samples]


# =========================================================
# 5. MAKE SAME LENGTH
# =========================================================

length = min(
    len(original),
    len(karaoke),
    len(different)
)

original = original[:length]

karaoke = karaoke[:length]

different = different[:length]


print("Common signal length:", length)


# =========================================================
# 6. NORMALIZE AUDIO
# =========================================================

def normalize(signal):

    maximum = np.max(np.abs(signal))

    if maximum != 0:
        signal = signal / maximum

    return signal


original = normalize(original)

karaoke = normalize(karaoke)

different = normalize(different)


# =========================================================
# 7. NORMALIZED CROSS-CORRELATION
# =========================================================

def normalized_correlation(x, y):

    corr = correlate(
        x,
        y,
        mode="full"
    )

    value = np.max(np.abs(corr)) / (
        np.linalg.norm(x) *
        np.linalg.norm(y)
    )

    return corr, value


# =========================================================
# THREE COMPARISONS
# =========================================================

corr_original_karaoke, value_original_karaoke = \
    normalized_correlation(
        original,
        karaoke
    )


corr_original_different, value_original_different = \
    normalized_correlation(
        original,
        different
    )


corr_karaoke_different, value_karaoke_different = \
    normalized_correlation(
        karaoke,
        different
    )


# =========================================================
# 8. AUTO-CORRELATION
# =========================================================

auto_original = correlate(
    original,
    original,
    mode="full"
)

auto_karaoke = correlate(
    karaoke,
    karaoke,
    mode="full"
)

auto_different = correlate(
    different,
    different,
    mode="full"
)
# AUTO-CORRELATION
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(auto_original)
plt.title("Autocorrelation - Original")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(auto_karaoke)
plt.title("Autocorrelation - Karaoke")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(auto_different)
plt.title("Autocorrelation - Different Song")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.tight_layout()
plt.show()
# CROSS-CORRELATION
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(corr_original_karaoke)
plt.title(
    "Cross Correlation: Original vs Karaoke"
)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(corr_original_different)
plt.title(
    "Cross Correlation: Original vs Different"
)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(corr_karaoke_different)
plt.title(
    "Cross Correlation: Karaoke vs Different"
)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.tight_layout()
plt.show()
# NORMALIZED CORRELATION VALUES
print("       NORMALIZED CORRELATION VALUES")
print(
    "Original vs Karaoke   :",
    round(value_original_karaoke, 4)
)
print(
    "Original vs Different :",
    round(value_original_different, 4)
)
print(
    "Karaoke vs Different  :",
    round(value_karaoke_different, 4)
)
# NORMALIZED CORRELATION COMPARISON
pairs = [
    "Original\nKaraoke",
    "Original\nDifferent",
    "Karaoke\nDifferent"
]
values = [
    value_original_karaoke,
    value_original_different,
    value_karaoke_different
]
plt.figure(figsize=(8, 5))
bars = plt.bar(
    pairs,
    values
)
plt.title(
    "Normalized Correlation Comparison"
)
plt.ylabel(
    "Correlation Value"
)
plt.ylim(0, 1)
plt.grid(
    axis="y"
)
# Display values on bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )
plt.tight_layout()
plt.show()
print("              OBSERVATION")
if value_original_karaoke > value_original_different:
    print(
        "Original and Karaoke have higher correlation."
    )
    print(
        "This indicates that the two tracks are more "
        "similar than the Original and Different Song."
    )
else:
    print(
        "Original and Karaoke do not have the highest "
        "correlation."
    )
print("\nCorrelation value interpretation:")
print("Closer to 1  -> High similarity")
print("Closer to 0  -> Low similarity")
print(
    "Negative value -> Opposite relationship"
)
print("       EXPERIMENT COMPLETED")