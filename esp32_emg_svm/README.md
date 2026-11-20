# ESP32 EMG Linear SVM (Arduino IDE)

This folder contains everything needed to compile and flash the trained linear SVM (`RawInput.txt`, 11 classes, 2 EMG channels, 207 causal features) onto an ESP32 using **Arduino IDE**.

## Files in this folder
- `esp32_emg_svm.ino` — Main Arduino sketch (ADC sampling at 1 kHz + serial output / optional serial test input).
- `emg_svm_classifier.h` — Portable C++11 ring-buffer feature extractor and linear SVM inference engine.
- `svm_model.h` — Auto-generated model weights, normalization constants, and class names exported from `linear_svm.joblib`.

## Flashing via Arduino IDE

1. **Install ESP32 Board Support** (if not already installed):
   - Open Arduino IDE → **File → Preferences**.
   - Under *Additional Boards Manager URLs*, add:
     `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
   - Open **Tools → Board → Boards Manager**, search for **esp32**, and install **esp32 by Espressif Systems**.

2. **Open the Sketch**:
   - Open `esp32_emg_svm/esp32_emg_svm.ino` in Arduino IDE (`svm_model.h` and `emg_svm_classifier.h` will open automatically as tabs).

3. **Check Configuration at the Top of `esp32_emg_svm.ino`**:
   - `EMG_PINS`: defaults to `{ 34, 35 }` (ADC1 pins on classic ESP32). Change these if your EMG sensors are wired to different pins or if you are using an ESP32-S3 / ESP32-C3.
   - `SAMPLE_RATE_HZ`: defaults to `1000` Hz. Set this to match the sampling rate used when recording `RawInput.txt`.
   - `PREDICT_EVERY_N_SAMPLES`: defaults to `50` (prints a prediction every 50 samples, i.e., 20 Hz at 1 kHz).
   - `USE_SERIAL_TEST_INPUT`: set to `0` for live ADC sensors, or `1` to send `ch0,ch1` voltage lines over the Serial Monitor for bench testing without electrodes.

4. **Select Board & Port, then Upload**:
   - **Tools → Board → esp32** → select your ESP32 board (e.g., *ESP32 Dev Module*).
   - **Tools → Port** → select the USB COM port for your ESP32.
   - Click **Upload** (`→`).

5. **View Live Predictions**:
   - Open **Tools → Serial Monitor** and set the baud rate to **115200**.
   - After warming up the 400-sample sliding window, it prints lines like:
     `Grip: grip4 (class 4, margin=2.841)`

## Updating the Model After Retraining

Whenever you retrain the model (`python train_svm.py`), regenerate `svm_model.h` from the project root before re-uploading in Arduino IDE:

```bash
python export_esp32.py
```

If you retrained on multiple sessions (e.g., `day1` + `day2`), pass `--session day1` (or omit `--session` to export the shared base weights).
