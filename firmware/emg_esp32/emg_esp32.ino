#include <math.h>
#include <stdint.h>
#include "model_data.h"

float mu_cal[N_FEAT];
float sd_cal[N_FEAT];
float h_cal[N_CAL * H2];
float proto[N_CLASS * H2];
float w_ft[N_CLASS * H2];
float b_ft[N_CLASS];
int32_t qw_ft[N_CLASS * H2];
int32_t qb_ft[N_CLASS];

enum Head { HEAD_ORIGINAL, HEAD_PROTO, HEAD_FINETUNED };

void normalize(const float *x, const float *mu, const float *sd, float *out) {
  for (int i = 0; i < N_FEAT; i++) out[i] = (x[i] - mu[i]) / sd[i];
}

void hidden_float(const float *x, float *h2) {
  float h1[H1];
  for (int j = 0; j < H1; j++) {
    float s = B0[j];
    for (int i = 0; i < N_FEAT; i++) s += W0[j * N_FEAT + i] * x[i];
    h1[j] = s > 0 ? s : 0;
  }
  for (int j = 0; j < H2; j++) {
    float s = B1[j];
    for (int i = 0; i < H1; i++) s += W1[j * H1 + i] * h1[i];
    h2[j] = s > 0 ? s : 0;
  }
}

int32_t quantize(float v, float scale) {
  float q = floorf(v / scale + 0.5f);
  if (q > 127) q = 127;
  if (q < -127) q = -127;
  return (int32_t)q;
}

void dense_int8(const int8_t *w, const int32_t *b, const int32_t *in, int n_in, int n_out, int32_t *acc) {
  for (int j = 0; j < n_out; j++) {
    int32_t s = b[j];
    for (int i = 0; i < n_in; i++) s += (int32_t)w[j * n_in + i] * in[i];
    acc[j] = s;
  }
}

void requant_relu(const int32_t *acc, int n, float mul, float next_scale, int32_t *out) {
  for (int j = 0; j < n; j++) {
    float r = (float)acc[j] * mul;
    if (r < 0) r = 0;
    out[j] = quantize(r, next_scale);
  }
}

void hidden_int8(const float *x, int32_t *q2) {
  int32_t q0[N_FEAT], a1[H1], q1[H1], a2[H2];
  for (int i = 0; i < N_FEAT; i++) q0[i] = quantize(x[i], S_ACT[0]);
  dense_int8(QW0, QB0, q0, N_FEAT, H1, a1);
  requant_relu(a1, H1, S_MUL[0], S_ACT[1], q1);
  dense_int8(QW1, QB1, q1, H1, H2, a2);
  requant_relu(a2, H2, S_MUL[1], S_ACT[2], q2);
}

void hidden(const float *x, bool use_int8, float *h) {
  if (use_int8) {
    int32_t q[H2];
    hidden_int8(x, q);
    for (int j = 0; j < H2; j++) h[j] = (float)q[j] * S_ACT[2];
  } else {
    hidden_float(x, h);
  }
}

int argmax_f(const float *v, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
  return best;
}

int predict(const float *x, bool use_int8, Head head) {
  if (head == HEAD_PROTO) {
    float h[H2];
    hidden(x, use_int8, h);
    int best = 0;
    float best_d = 0;
    for (int c = 0; c < N_CLASS; c++) {
      float d = 0;
      for (int j = 0; j < H2; j++) {
        float t = h[j] - proto[c * H2 + j];
        d += t * t;
      }
      if (c == 0 || d < best_d) {
        best_d = d;
        best = c;
      }
    }
    return best;
  }
  if (use_int8) {
    int32_t q[H2], a[N_CLASS];
    hidden_int8(x, q);
    for (int c = 0; c < N_CLASS; c++) {
      int32_t s = head == HEAD_FINETUNED ? qb_ft[c] : QB2[c];
      for (int j = 0; j < H2; j++) {
        int32_t w = head == HEAD_FINETUNED ? qw_ft[c * H2 + j] : (int32_t)QW2[c * H2 + j];
        s += w * q[j];
      }
      a[c] = s;
    }
    int best = 0;
    for (int c = 1; c < N_CLASS; c++) if (a[c] > a[best]) best = c;
    return best;
  }
  float h[H2], o[N_CLASS];
  hidden_float(x, h);
  for (int c = 0; c < N_CLASS; c++) {
    float s = head == HEAD_FINETUNED ? b_ft[c] : B2[c];
    for (int j = 0; j < H2; j++) s += (head == HEAD_FINETUNED ? w_ft[c * H2 + j] : W2[c * H2 + j]) * h[j];
    o[c] = s;
  }
  return argmax_f(o, N_CLASS);
}

void calibrate_norm() {
  for (int f = 0; f < N_FEAT; f++) {
    double s = 0, s2 = 0;
    for (int n = 0; n < N_CAL; n++) {
      double v = CAL_X[n * N_FEAT + f];
      s += v;
      s2 += v * v;
    }
    double m = s / N_CAL;
    double var = s2 / N_CAL - m * m;
    if (var < 0) var = 0;
    mu_cal[f] = (float)m;
    sd_cal[f] = (float)sqrt(var) + 1e-8f;
  }
}

void compute_hidden_cal(bool use_int8) {
  float x[N_FEAT];
  for (int n = 0; n < N_CAL; n++) {
    normalize(&CAL_X[n * N_FEAT], mu_cal, sd_cal, x);
    hidden(x, use_int8, &h_cal[n * H2]);
  }
}

void calibrate_proto() {
  for (int c = 0; c < N_CLASS; c++) {
    int count = 0;
    double acc[H2];
    for (int j = 0; j < H2; j++) acc[j] = 0;
    for (int n = 0; n < N_CAL; n++) {
      if (CAL_Y[n] != c) continue;
      count++;
      for (int j = 0; j < H2; j++) acc[j] += h_cal[n * H2 + j];
    }
    for (int j = 0; j < H2; j++) proto[c * H2 + j] = count ? (float)(acc[j] / count) : 0;
  }
}

void calibrate_finetune(bool use_int8) {
  for (int i = 0; i < N_CLASS * H2; i++) w_ft[i] = W2[i];
  for (int c = 0; c < N_CLASS; c++) b_ft[c] = B2[c];
  float z[N_CLASS], p[N_CLASS];
  for (int e = 0; e < EPOCHS; e++) {
    for (int k = 0; k < N_CAL; k++) {
      int n = CAL_ORDER[k];
      const float *h = &h_cal[n * H2];
      for (int c = 0; c < N_CLASS; c++) {
        float s = b_ft[c];
        for (int j = 0; j < H2; j++) s += w_ft[c * H2 + j] * h[j];
        z[c] = s;
      }
      float zmax = z[argmax_f(z, N_CLASS)];
      float sum = 0;
      for (int c = 0; c < N_CLASS; c++) {
        p[c] = expf(z[c] - zmax);
        sum += p[c];
      }
      for (int c = 0; c < N_CLASS; c++) p[c] /= sum;
      p[CAL_Y[n]] -= 1.0f;
      for (int c = 0; c < N_CLASS; c++) {
        float g = LR * p[c];
        for (int j = 0; j < H2; j++) w_ft[c * H2 + j] -= g * h[j];
        b_ft[c] -= g;
      }
    }
  }
  if (use_int8) {
    float wmax = 0;
    for (int i = 0; i < N_CLASS * H2; i++) if (fabsf(w_ft[i]) > wmax) wmax = fabsf(w_ft[i]);
    float sw = wmax / 127.0f;
    for (int i = 0; i < N_CLASS * H2; i++) qw_ft[i] = quantize(w_ft[i], sw);
    float sb = S_ACT[2] * sw;
    for (int c = 0; c < N_CLASS; c++) qb_ft[c] = (int32_t)floorf(b_ft[c] / sb + 0.5f);
  }
}

float evaluate(const float *mu, const float *sd, bool use_int8, Head head, float *us) {
  float x[N_FEAT];
  int correct = 0;
  unsigned long total = 0;
  for (int n = 0; n < N_TEST; n++) {
    normalize(&TEST_X[n * N_FEAT], mu, sd, x);
    unsigned long t0 = micros();
    int p = predict(x, use_int8, head);
    total += micros() - t0;
    if (p == TEST_Y[n]) correct++;
  }
  *us = (float)total / N_TEST;
  return 100.0f * correct / N_TEST;
}

void report(const char *name, float acc, float expected, float us_inf, long us_cal) {
  Serial.print(name);
  Serial.print(" | acc ");
  Serial.print(acc, 2);
  Serial.print(" % (Colab ");
  Serial.print(expected, 2);
  Serial.print(") | inference ");
  Serial.print(us_inf, 1);
  Serial.print(" us | calibration ");
  Serial.print(us_cal);
  Serial.println(" us");
}

void run_all(bool use_int8) {
  const float ef[4] = {EXP_F0, EXP_F1, EXP_F2, EXP_F3};
  const float ei[4] = {EXP_I0, EXP_I1, EXP_I2, EXP_I3};
  const float *e = use_int8 ? ei : ef;
  float us, a;
  unsigned long t0, t_norm, t_hidden, t_proto, t_ft;

  Serial.println(use_int8 ? "===== INT8 =====" : "===== Float32 =====");
  a = evaluate(TRAIN_MU, TRAIN_SD, use_int8, HEAD_ORIGINAL, &us);
  report("No calibration                  ", a, e[0], us, 0);

  t0 = micros();
  calibrate_norm();
  t_norm = micros() - t0;
  a = evaluate(mu_cal, sd_cal, use_int8, HEAD_ORIGINAL, &us);
  report("Re-normalization                ", a, e[1], us, t_norm);

  t0 = micros();
  compute_hidden_cal(use_int8);
  t_hidden = micros() - t0;

  t0 = micros();
  calibrate_proto();
  t_proto = micros() - t0;
  a = evaluate(mu_cal, sd_cal, use_int8, HEAD_PROTO, &us);
  report("Re-norm + prototypes            ", a, e[2], us, t_norm + t_hidden + t_proto);

  t0 = micros();
  calibrate_finetune(use_int8);
  t_ft = micros() - t0;
  a = evaluate(mu_cal, sd_cal, use_int8, HEAD_FINETUNED, &us);
  report("Re-norm + last-layer fine-tuning", a, e[3], us, t_norm + t_hidden + t_ft);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("EMG gesture recognition: on-device calibration test");
  Serial.print("Test windows: ");
  Serial.print(N_TEST);
  Serial.print(", calibration windows: ");
  Serial.print(N_CAL);
  Serial.print(", fine-tuning epochs: ");
  Serial.println(EPOCHS);
  int n_w = N_FEAT * H1 + H1 * H2 + H2 * N_CLASS;
  int n_b = H1 + H2 + N_CLASS;
  Serial.print("Model size float32: ");
  Serial.print((n_w + n_b) * 4);
  Serial.print(" bytes, INT8: ");
  Serial.print(n_w + n_b * 4);
  Serial.println(" bytes");
  int ram_cal = (int)(sizeof(mu_cal) + sizeof(sd_cal) + sizeof(h_cal) + sizeof(proto) + sizeof(w_ft) + sizeof(b_ft) + sizeof(qw_ft) + sizeof(qb_ft));
  Serial.print("RAM used by calibration buffers: ");
  Serial.print(ram_cal);
  Serial.println(" bytes");

  run_all(false);
  run_all(true);

  float f[N_FEAT];
  unsigned long t0 = micros();
  for (int k = 0; k < 10; k++) {
    for (int c = 0; c < N_CH; c++) {
      float sa = 0, s2 = 0, wl = 0;
      int zc = 0;
      for (int t = 0; t < WIN; t++) {
        float v = DEMO_WINDOW[t * N_CH + c];
        sa += fabsf(v);
        s2 += v * v;
        if (t > 0) {
          float p = DEMO_WINDOW[(t - 1) * N_CH + c];
          wl += fabsf(v - p);
          if (v * p < 0) zc++;
        }
      }
      f[c] = sa / WIN;
      f[N_CH + c] = sqrtf(s2 / WIN);
      f[2 * N_CH + c] = wl;
      f[3 * N_CH + c] = (float)zc;
    }
  }
  float t_feat = (micros() - t0) / 10.0f;
  float max_err = 0;
  for (int i = 0; i < N_FEAT; i++) {
    float e = fabsf(f[i] - DEMO_FEATURES[i]) / (fabsf(DEMO_FEATURES[i]) + 1e-12f);
    if (e > max_err) max_err = e;
  }
  Serial.print("Feature extraction for one window: ");
  Serial.print(t_feat, 1);
  Serial.print(" us, max relative difference from Colab: ");
  Serial.println(max_err, 6);
  Serial.println("Done");
}

void loop() {}