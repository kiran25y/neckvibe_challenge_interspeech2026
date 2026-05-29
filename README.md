
NeckVibe Challenge Submission
Team: Seoultech - Medisensing Inc
Date: February 12, 2026

CONTENTS:
  1. pvh.py                      - PVH tabular (CatBoost) training + OOF export
  2. npvh.py                     - NPVH tabular (CatBoost) training + OOF export
  3. mil_pvh.py                  - PVH Enhanced MIL training (5-fold) + OOF export
  4. mil_npvh.py                 - NPVH Enhanced MIL training (5-fold) + OOF export
  5. ensemble.py                 - Builds stacking ensemble artifacts from OOF
  6. inference.py                - Test inference and submission CSV generation
  7. requirements.txt            - Python dependencies
  8. README.txt                  - This file

REQUIREMENTS:

Python 3.10+

Install:
  pip install -r requirements.txt

Core packages:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - catboost
  - torch

GPU is recommended for MIL training but not required for inference.


EXPECTED DATA LAYOUT:


/workspace/NeckVibeChallenge/
│
├── Labels/
│   └── Train.csv
│
├── Features/                                  (TRAIN .mat files)
│   ├── NV001_20130101.mat
│   ├── NV001_20130102.mat
│   └── ...
│
└── Test_set/
    ├── Test_features/                         (TEST .mat files)
    │   ├── NV064_20140101.mat
    │   └── ...
    ├── Test_features_index.csv
    └── Test_results_template/
        ├── Task1_PVH_detection.csv
        └── Task2_NPVH_detection.csv


PROJECT OUTPUT FOLDER STRUCTURE:


/Seoultech-Medisensing Inc/
│
├── source code/
│   ├── pvh_catboost.py
│   ├── npvh_catboost.py
│   ├── mil_pvh.py
│   ├── mil_npvh.py
│   ├── ensemble.py
│   └── inference.py
│
├── models/
├── npvh_catboost/
│   ├── npvh_catboost_final.cbm
│   ├── npvh_features_final.json
│   └── npvh_oof_final.csv
├── pvh_catboost/
│   ├── pvh_catboost.cbm
│   ├── pvh_features.json
│   └── pvh_oof.csv
│
├── pvh_mil/
│   ├── mil_pvh_fold1.pt ... mil_pvh_fold5.pt
│   └── mil_pvh_oof.csv
│
├── npvh_mil/
│   ├── mil_npvh_fold1.pt ... mil_npvh_fold5.pt
│   └── mil_npvh_oof.csv
│
├── pvh_ensemble_final/
│   ├── merged_oof_with_stack.csv
│   ├── ensemble_summary.json
│   ├── meta_model.pkl
│   ├── qt_tab.pkl
│   └── qt_mil.pkl
│
├── npvh_ensemble_final/
│   ├── merged_oof_with_stack.csv
│   ├── ensemble_summary.json
│   ├── meta_model.pkl
│   ├── qt_tab.pkl
│   └── qt_mil.pkl
│
└── Classification test results/
    ├── Task1_PVH_detection.csv
    └── Task2_NPVH_detection.csv


ENSEMBLE METHOD (IMPORTANT):

We do NOT use a fixed weighted mean.

Final predictions are generated using stacking:
  1) Apply QuantileTransformer (CDF → uniform[0,1]) separately to:
       - tabular probability
       - MIL probability
  2) Build meta-features:
       [ t, m, |t - m|, t*m ]
  3) Train LogisticRegression meta-model with class_weight="balanced"
  4) Apply the fitted QT + meta-model to test predictions

Note:
- For NPVH we prefer the rank-normalized tabular OOF ("oof_rank") due to fold scale drift.
- For PVH we use raw tabular OOF ("oof_raw") by default.


USAGE:


1) Train Tabular Models (CatBoost)
---------------------------------
PVH:
  python pvh.py

NPVH:
  python npvh.py

Inputs:
  - /workspace/NeckVibeChallenge/Labels/Train.csv
  - /workspace/NeckVibeChallenge/Features/*.mat


2) Train MIL Models (Enhanced MIL, 5-fold)
-----------------------------------------
PVH:
  python mil_pvh.py

NPVH:
  python mil_npvh.py


3) Build Stacking Ensemble Artifacts (from OOF)
-----------------------------------------------
  python ensemble.py

This creates:
  - meta_model.pkl
  - qt_tab.pkl
  - qt_mil.pkl
for both PVH and NPVH:

  /workspace/final_neckvibe/pvh_ensemble_final/
  /workspace/final_neckvibe/npvh_ensemble_final/


4) Generate Test Predictions + Submission CSVs
----------------------------------------------
  python inference.py

Inputs:
  - /workspace/NeckVibeChallenge/Test_set/Test_features/*.mat
  - /workspace/NeckVibeChallenge/Test_set/Test_features_index.csv
  - /workspace/NeckVibeChallenge/Test_set/Test_results_template/*.csv
  - Trained model artifacts (CatBoost + MIL + stacking PKLs)

Outputs:
  - /workspace/final_neckvibe/submission_outputs/Task1_PVH_detection.csv
  - /workspace/final_neckvibe/submission_outputs/Task2_NPVH_detection.csv

EXPECTED OUTPUT FORMAT (SUBMISSION):


Task1_PVH_detection.csv:
  subjectID,pvh_probability,pvh_prediction
  NV064,0.87,1
  NV089,0.24,0
  ...

Task2_NPVH_detection.csv:
  subjectID,npvh_probability,npvh_prediction
  NV064,0.12,0
  NV089,0.78,1
  ...

Decision rule:
  prediction = 1 if probability >= 0.5 else 0
