import pandas as pd

# Load original dataset
df = pd.read_csv("diabetic_data.csv")

# -----------------------------
# 1. Patients table
# -----------------------------
patients = df[[
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight"
]].drop_duplicates(subset=["patient_nbr"]).copy()

# -----------------------------
# 2. Encounters table
# -----------------------------
encounters = df[[
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]].drop_duplicates(subset=["encounter_id"]).copy()

# -----------------------------
# 3. Clinical table
# -----------------------------
clinical = df[[
    "encounter_id",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult"
]].drop_duplicates(subset=["encounter_id"]).copy()

# -----------------------------
# 4. Medications / Outcomes table
# -----------------------------
medications_outcomes = df[[
    "encounter_id",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted"
]].drop_duplicates(subset=["encounter_id"]).copy()

# -----------------------------
# Save as parquet files
# -----------------------------
patients.to_parquet("patients.parquet", index=False)
encounters.to_parquet("encounters.parquet", index=False)
clinical.to_parquet("clinical.parquet", index=False)
medications_outcomes.to_parquet("medications_outcomes.parquet", index=False)

# -----------------------------
# Quick check
# -----------------------------
print("Saved parquet files successfully!")
