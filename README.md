# EMG gesture recognition on ESP32

Bachelor thesis, Computer Engineering, Ala-Too International University.
Topic: cross-session robust and energy-aware EMG gesture recognition on a microcontroller.

Data: NinaPro DB6 (https://ninapro.hevs.ch/instructions/DB6.html). The .mat files are not in this repository.

How to run:
1. Put S1_D1_T1.mat ... S1_D5_T2.mat into MyDrive/ninapro/ on Google Drive.
2. Run notebooks/01_features.ipynb in Colab. It saves the features to Drive.
3. Run notebooks/02_experiments.ipynb. It saves the result tables and downloads model_data.h.
4. Put model_data.h into firmware/emg_esp32/, upload emg_esp32.ino from Arduino IDE, open Serial Monitor at 115200.

Code is in src/emg, the ESP32 sketch is in firmware/emg_esp32, tables and board output go to results/.