#pragma once
// Portable C++11 inference engine for the EMG linear SVM exported by export_esp32.py.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "svm_model.h"

class EmgSvmClassifier {
public:
    EmgSvmClassifier() { reset(); }

    void reset() {
        for (int t = 0; t < EMG_MAX_WINDOW; ++t) {
            for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
                n_buf_[t][c] = 0.0f;
                dn_buf_[t][c] = 0.0f;
            }
        }
        for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
            prev_n_[c] = 0.0f;
            norm_mu_[c] = EMG_NORM_MU[c];
            norm_sd_[c] = (EMG_NORM_SD[c] > 1e-9f) ? EMG_NORM_SD[c] : 1e-9f;
        }
        head_ = 0;
        sample_count_ = 0;
    }

    // Override per-channel normalization stats at runtime (e.g., after calibration).
    void setNormalization(const float mu[EMG_NUM_CHANNELS], const float sd[EMG_NUM_CHANNELS]) {
        for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
            norm_mu_[c] = mu[c];
            norm_sd_[c] = (sd[c] > 1e-9f) ? sd[c] : 1e-9f;
        }
    }

    bool isReady() const {
        return sample_count_ >= static_cast<uint32_t>(EMG_MAX_WINDOW);
    }

    uint32_t sampleCount() const { return sample_count_; }

    // Push one new sample (voltage per channel matching training units).
    void pushSample(const float raw[EMG_NUM_CHANNELS]) {
        for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
            const float clipped = (raw[c] > 0.0f) ? raw[c] : 0.0f;
            const float l = log1pf(clipped);
            const float n = (l - norm_mu_[c]) / norm_sd_[c];
            const float dn = (sample_count_ == 0) ? 0.0f : fabsf(n - prev_n_[c]);
            n_buf_[head_][c] = n;
            dn_buf_[head_][c] = dn;
            prev_n_[c] = n;
        }
        head_ = (head_ + 1) % EMG_MAX_WINDOW;
        ++sample_count_;
    }

    // Compute 207 causal features from the ring buffer and run LinearSVC.
    // Returns predicted class index [0 .. EMG_NUM_CLASSES-1], or -1 if no samples yet.
    int predict(float scores_out[EMG_NUM_CLASSES] = nullptr) const {
        if (sample_count_ == 0) {
            return -1;
        }

        const int avail = (sample_count_ < static_cast<uint32_t>(EMG_MAX_WINDOW))
            ? static_cast<int>(sample_count_)
            : EMG_MAX_WINDOW;
        const int cur_idx = (head_ == 0) ? (EMG_MAX_WINDOW - 1) : (head_ - 1);

        float features[EMG_NUM_FEATURES];
        float poly_in[EMG_POLY_INPUT_DIM];
        int f_pos = 0;
        int p_pos = 0;

        float n_cur[EMG_NUM_CHANNELS];
        for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
            n_cur[c] = n_buf_[cur_idx][c];
            features[f_pos++] = n_cur[c];
            poly_in[p_pos++] = n_cur[c];
        }
        for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
            features[f_pos++] = dn_buf_[cur_idx][c];
        }

        float sum_d[EMG_NUM_CHANNELS] = {0.0f};
        float sum_d2[EMG_NUM_CHANNELS] = {0.0f};
        float sum_dn[EMG_NUM_CHANNELS] = {0.0f};
        float cur_max[EMG_NUM_CHANNELS];
        float cur_min[EMG_NUM_CHANNELS];

        int idx = cur_idx;
        int w = 0;
        for (int age = 0; age < EMG_MAX_WINDOW; ++age) {
            for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
                const float v = n_buf_[idx][c];
                // Shifted variance around n_cur avoids float32 cancellation on flat channels.
                const float d = v - n_cur[c];
                sum_d[c] += d;
                sum_d2[c] += d * d;
                sum_dn[c] += dn_buf_[idx][c];
                if (age == 0 || v > cur_max[c]) {
                    cur_max[c] = v;
                }
                if (age == 0 || v < cur_min[c]) {
                    cur_min[c] = v;
                }
            }

            if (w < EMG_NUM_WINDOWS && (age + 1) == EMG_WINDOWS[w]) {
                const float inv_k = 1.0f / static_cast<float>(EMG_WINDOWS[w]);
                float mean_n[EMG_NUM_CHANNELS];
                float std_n[EMG_NUM_CHANNELS];
                float wl[EMG_NUM_CHANNELS];

                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
                    const float mean_d = sum_d[c] * inv_k;
                    mean_n[c] = n_cur[c] + mean_d;
                    const float var_n = sum_d2[c] * inv_k - mean_d * mean_d;
                    std_n[c] = sqrtf(fmaxf(var_n, 0.0f));
                    wl[c] = sum_dn[c] * inv_k;
                }

                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = mean_n[c];
                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = n_cur[c] - mean_n[c];
                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = std_n[c];
                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = wl[c];
                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = cur_max[c];
                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = cur_min[c];
                for (int c = 0; c < EMG_NUM_CHANNELS; ++c) features[f_pos++] = cur_max[c] - cur_min[c];

                if (EMG_IS_POLY_WINDOW[w] != 0) {
                    for (int c = 0; c < EMG_NUM_CHANNELS; ++c) poly_in[p_pos++] = mean_n[c];
                    for (int c = 0; c < EMG_NUM_CHANNELS; ++c) poly_in[p_pos++] = std_n[c];
                }
                ++w;
            }

            if (age + 1 < avail) {
                idx = (idx == 0) ? (EMG_MAX_WINDOW - 1) : (idx - 1);
            }
        }

        // Degree-2 polynomial features (matches sklearn PolynomialFeatures(2, include_bias=False))
        for (int i = 0; i < EMG_POLY_INPUT_DIM; ++i) {
            features[f_pos++] = poly_in[i];
        }
        for (int i = 0; i < EMG_POLY_INPUT_DIM; ++i) {
            const float pi = poly_in[i];
            for (int j = i; j < EMG_POLY_INPUT_DIM; ++j) {
                features[f_pos++] = pi * poly_in[j];
            }
        }

        int best_cls = 0;
        float best_score = 0.0f;
        for (int cls = 0; cls < EMG_NUM_CLASSES; ++cls) {
            float score = EMG_SVM_BIAS[cls];
            const float* w_row = EMG_SVM_WEIGHTS[cls];
            for (int f = 0; f < EMG_NUM_FEATURES; ++f) {
                score += w_row[f] * features[f];
            }
            if (scores_out != nullptr) {
                scores_out[cls] = score;
            }
            if (cls == 0 || score > best_score) {
                best_score = score;
                best_cls = cls;
            }
        }
        return best_cls;
    }

    // Compute predicted class index [0 .. EMG_NUM_CLASSES-1] and calibrated softmax
    // confidence in [0.0, 1.0]. Optionally outputs full probabilities and raw scores.
    int predictWithConfidence(
        float* confidence_out,
        float probs_out[EMG_NUM_CLASSES] = nullptr,
        float scores_out[EMG_NUM_CLASSES] = nullptr) const
    {
        float scores_local[EMG_NUM_CLASSES];
        const int pred = predict(scores_local);
        if (pred < 0) {
            if (confidence_out != nullptr) {
                *confidence_out = 0.0f;
            }
            return -1;
        }
        if (scores_out != nullptr) {
            for (int c = 0; c < EMG_NUM_CLASSES; ++c) {
                scores_out[c] = scores_local[c];
            }
        }

        const float max_s = scores_local[pred];
        const float inv_t = (EMG_SOFTMAX_TEMP > 1e-6f) ? (1.0f / EMG_SOFTMAX_TEMP) : 1.0f;
        float probs_local[EMG_NUM_CLASSES];
        float sum_exp = 0.0f;
        for (int c = 0; c < EMG_NUM_CLASSES; ++c) {
            const float e = expf((scores_local[c] - max_s) * inv_t);
            probs_local[c] = e;
            sum_exp += e;
        }
        const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int c = 0; c < EMG_NUM_CLASSES; ++c) {
            probs_local[c] *= inv_sum;
            if (probs_out != nullptr) {
                probs_out[c] = probs_local[c];
            }
        }
        if (confidence_out != nullptr) {
            *confidence_out = probs_local[pred];
        }
        return pred;
    }

    static const char* className(int cls) {
        if (cls >= 0 && cls < EMG_NUM_CLASSES) {
            return EMG_CLASS_NAMES[cls];
        }
        return "unknown";
    }

private:
    float n_buf_[EMG_MAX_WINDOW][EMG_NUM_CHANNELS];
    float dn_buf_[EMG_MAX_WINDOW][EMG_NUM_CHANNELS];
    float prev_n_[EMG_NUM_CHANNELS];
    float norm_mu_[EMG_NUM_CHANNELS];
    float norm_sd_[EMG_NUM_CHANNELS];
    int head_;
    uint32_t sample_count_;
};
