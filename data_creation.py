import logging
import pandas as pd

# Configure logging
# Logs are written to a file so pipeline activity and errors can be reviewed later.
logging.basicConfig(
    filename="data_creation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Data creation script started.")

try:
    # Load original dataset
    # Read the raw CSV file containing all diabetic hospital encounter records.
    df = pd.read_csv("diabetic_data.csv")
    logging.info(f"Raw dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 1. Patients table
    # Create a patient-level table with one row per unique patient.
    # These variables are intended to describe patient demographics.
    patients = df[[
        "patient_nbr",
        "race",
        "gender",
        "age",
        "weight"
    ]].drop_duplicates(subset=["patient_nbr"]).copy()

    logging.info(f"Patients table created successfully with {patients.shape[0]} rows.")

   
    # 2. Encounters table
    # Create an encounter-level table with one row per hospital encounter.
    # This table contains admission, discharge, and visit history variables.
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

    logging.info(f"Encounters table created successfully with {encounters.shape[0]} rows.")

    
    # 3. Clinical table
    # Create a clinical table with one row per encounter.
    # This table stores diagnoses, lab activity, and other clinical variables.
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

    logging.info(f"Clinical table created successfully with {clinical.shape[0]} rows.")

    # 4. Medications / Outcomes table
    # Create a medications and outcomes table with one row per encounter.
    # This table stores diabetes medication indicators and readmission outcome.
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

    logging.info(f"Medications/Outcomes table created successfully with {medications_outcomes.shape[0]} rows.")

    # Save as parquet files
    # Save each relational table as a parquet file for efficient storage and analysis.
    patients.to_parquet("patients.parquet", index=False)
    encounters.to_parquet("encounters.parquet", index=False)
    clinical.to_parquet("clinical.parquet", index=False)
    medications_outcomes.to_parquet("medications_outcomes.parquet", index=False)

    logging.info("All parquet files saved successfully.")

    # Quick validation output
    print("Saved parquet files successfully!")
    print("patients:", patients.shape)
    print("encounters:", encounters.shape)
    print("clinical:", clinical.shape)
    print("medications_outcomes:", medications_outcomes.shape)

    logging.info("Data creation script completed successfully.")

except FileNotFoundError as e:
    logging.error(f"Input file not found: {e}")
    raise

except KeyError as e:
    logging.error(f"Missing expected column in dataset: {e}")
    raise

except Exception as e:
    logging.error(f"Unexpected error during data creation: {e}")
    raise
