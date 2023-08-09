This repository contains the code which was used for a research project at UNSW (COMP9991 T1 2023 and COMP9992 T2 2023).
- Student : Louis Milhiet z5374550
- Supervisor : Dr Alan Blair
- Cosupervisor : Dr Alekandar Ignjatovic

The project :

The aim of this project is to investigate the use of Chromatic Derivatives, in combination with deep neural networks, for speech processing and phoneme recognition.

Chromatic Derivatives (CD's) are numerically stable differential operators which correspond to families of orthogonal polynomials. They can be robustly evaluated, even for operators of very high degree, using FIR filterbanks consisting of broad band comb-like filters.

Due to their orthogonality, numerical stability and time shift invariance, CD's have recently been proposed for speech and signal processing, as an alternative to raw signal input or Mel-Frequency Cepstral Coefficients.

This project will explore the use of CD's for phoneme recognition on the TIMIT dataset. Neural network models including CNN and LSTM will be tested; as well as preprocessing which may include just the raw CD components, and/or a spectrogram derived from their pairwise correlations.


The repository : 

This repository is a clone of the SpeechBrain repository --> https://github.com/speechbrain/speechbrain/tree/develop

The files folders which have been modified are :

- speechbrain > lobes > feature.py 
- files added in the speechbrain > lobes > tina_chroma_filters.py
- files added in recipes > TIMIT > ASR
- results.ods contains all results for models that have been trained during the project. This file is in the main folder.

