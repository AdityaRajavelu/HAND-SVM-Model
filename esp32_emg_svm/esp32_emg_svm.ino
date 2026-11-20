#include <Arduino.h>
#include "emg_svm_classifier.h"

// =============================================================================
// USER CONFIGURATION
// =============================================================================
// Classic ESP32 ADC1 pins (safe to use alongside WiFi). Update if using ESP32-S3/C3
// or wired to different analog pins.
static const int EMG_PINS[EMG_NUM_CHANNELS] = { 34, 35 };

// Match the sampling rate used when recording NewPosition.txt so the 10..400 sample
// sliding windows span the same durations.
static constexpr uint32_t SAMPLE_RATE_HZ = 1000;
static constexpr uint32_t PREDICT_EVERY_N_SAMPLES = 50;  // 20 Hz at 1000 Hz
static constexpr uint32_t SERIAL_BAUD = 115200;

// Set to 1 to feed test lines like "0.269,0.209" over USB Serial Monitor
// instead of reading live ADC pins.
#define USE_SERIAL_TEST_INPUT 0
// =============================================================================

// Global so the ~6.5 KB ring buffers live in static BSS RAM, not the loopTask stack.
EmgSvmClassifier classifier;

static constexpr uint32_t SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;
static uint32_t last_sample_us = 0;

// Hook your motor / servo / prosthetic hand controller here.
void onGripPredicted(int cls, const char* name, float confidence, int runner_up_cls, float runner_up_conf) {
    (void)cls;
    // Example: only actuate a servo/motor when confidence is above a threshold:
    // if (confidence < 0.70f) return;

    Serial.print("Grip: ");
    Serial.print(name);
    Serial.print(" | Confidence: ");
    Serial.print(confidence * 100.0f, 1);
    Serial.print("% (2nd: ");
    Serial.print(EmgSvmClassifier::className(runner_up_cls));
    Serial.print(" ");
    Serial.print(runner_up_conf * 100.0f, 1);
    Serial.println("%)");
}

static void handleSample(const float raw[EMG_NUM_CHANNELS]) {
    classifier.pushSample(raw);

    if ((classifier.sampleCount() % PREDICT_EVERY_N_SAMPLES) != 0) {
        return;
    }

    if (!classifier.isReady()) {
        Serial.print("Warming up window: ");
        Serial.print(classifier.sampleCount());
        Serial.print(" / ");
        Serial.println(EMG_MAX_WINDOW);
        return;
    }

    float confidence = 0.0f;
    float probs[EMG_NUM_CLASSES];
    const int pred = classifier.predictWithConfidence(&confidence, probs, nullptr);

    int second_cls = (pred == 0) ? 1 : 0;
    for (int c = 0; c < EMG_NUM_CLASSES; ++c) {
        if (c != pred && probs[c] > probs[second_cls]) {
            second_cls = c;
        }
    }
    onGripPredicted(pred, EmgSvmClassifier::className(pred), confidence, second_cls, probs[second_cls]);
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(200);

#if !USE_SERIAL_TEST_INPUT
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);  // ~0 - 3.1 V range, matching NewPosition.txt
    for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
        pinMode(EMG_PINS[c], INPUT);
    }
#endif

    classifier.reset();
    last_sample_us = micros();

    Serial.println("ESP32 EMG Linear SVM ready.");
#if USE_SERIAL_TEST_INPUT
    Serial.println("Serial test mode active: send 'ch0,ch1' lines.");
#endif
}

void loop() {
#if USE_SERIAL_TEST_INPUT
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) {
            return;
        }
        int comma = line.indexOf(',');
        if (comma < 0) {
            return;
        }
        float raw[EMG_NUM_CHANNELS];
        raw[0] = line.substring(0, comma).toFloat();
        raw[1] = line.substring(comma + 1).toFloat();
        handleSample(raw);
    }
#else
    const uint32_t now_us = micros();
    if (static_cast<uint32_t>(now_us - last_sample_us) >= SAMPLE_INTERVAL_US) {
        last_sample_us += SAMPLE_INTERVAL_US;
        // Guard against falling permanently behind if loop() stalls briefly
        if (static_cast<uint32_t>(now_us - last_sample_us) > SAMPLE_INTERVAL_US * 4UL) {
            last_sample_us = now_us;
        }

        float raw[EMG_NUM_CHANNELS];
        for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
            raw[c] = static_cast<float>(analogReadMilliVolts(EMG_PINS[c])) * 0.001f;
        }
        handleSample(raw);
    }
#endif
}
