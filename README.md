# DS4320 Project 1: Identifying Factors Associated with 30-Day Hospital Readmission in Diabetic Patients

This repository contains a data science project focused on predicting 30-day hospital readmission risk among diabetic patients. The project develops a relational dataset from hospital encounter data and applies machine learning models to identify key factors associated with readmission. The repository includes the constructed dataset stored in a UVA OneDrive folder, a data processing and analysis pipeline implemented in Python and SQL using DuckDB, and a press release summarizing the findings for a general audience. Together, these components provide both a technical analysis and a clear, real-world interpretation of factors that contribute to hospital readmissions.

Grace Pitts

twg3sr

DOI: link**

[Press Release](press_release.md)

[Data](https://myuva-my.sharepoint.com/:f:/r/personal/twg3sr_virginia_edu/Documents/Project1%20Data?csf=1&web=1&e=v0GJec)

[Pipeline Notebook](hospital_readmission_pipeline.ipynb)
[Pipeline Markdown](hospital_readmission_pipeline.md)

This project is licensed under the MIT License. See the [License](LICENSE) file for details.




## Problem Definition
Hospitals want to identify patients who are likely to be readmitted.

Identify patient characteristics and hospital encounter patterns associated with 30-day readmission among patients with diabetes to support interventions that reduce readmission rates.

The general problem of predicting hospital readmission is broad and encompasses a wide range of factors, including patient demographics, medical conditions, treatments, and hospital-level practices. To make the project more focused and aligned with the available data, the problem was refined to examine 30-day readmission specifically among diabetic patients using hospital encounter data. This refinement uses a clearly defined group of patients with detailed data, which allows for a more focused and meaningful analysis. By narrowing the scope to diabetic patients, the project reduces complexity while still maintaining clinical relevance. This allows for clearer identification of patterns and more interpretable insights into factors associated with readmission risk.

Hospital readmissions are a major challenge in healthcare systems because they often indicate complications, poor care coordination, or inadequate follow-up treatment after discharge. Unplanned readmissions increase healthcare costs and can negatively impact patient outcomes. This issue is especially important for diabetic patients, who often require ongoing care and are at higher risk for complications that can lead to repeat hospital visits. Hospitals and healthcare providers are increasingly interested in understanding the factors that contribute to readmission so they can improve patient care and reduce unnecessary hospital visits. This project is motivated by the potential to use data analysis to better understand patterns in hospital admissions among diabetic patients and identify characteristics associated with higher readmission risk.

[Predicting Hospital Readmissions to Improve Patient Care](press_release.md)


## Domain Exposition

| Term | Definition |
|------|-----------|
| Hospital Readmission | When a patient is admitted to a hospital again within a specified period after discharge, typically within 30 days. |
| 30-Day Readmission | A common healthcare metric indicating whether a patient returns to the hospital within 30 days of discharge. |
| Patient Encounter | A single interaction between a patient and a healthcare provider, such as a hospital admission or visit. |
| Length of Stay (LOS) | The number of days a patient remains hospitalized during a single admission. |
| Diagnosis Code (ICD Code) | A standardized code used to classify diseases and medical conditions. |
| Comorbidity | The presence of one or more additional medical conditions alongside a primary condition. |
| Discharge | The process of releasing a patient from the hospital after treatment. |
| Readmission Rate | The percentage of patients who are readmitted within a specified time period after discharge. |
| Risk Factor | A characteristic or condition that increases the likelihood of hospital readmission. |
| Healthcare Outcome | A measurable result of healthcare services, such as recovery, complications, or readmission. |
| readmitted | Indicates whether a patient was readmitted within 30 days, after 30 days, or not readmitted. |
| time_in_hospital | The number of days a patient stayed in the hospital during a single encounter. |
| A1Cresult | A categorical variable indicating the patient’s A1C blood glucose test result level. |
| diabetesMed | Indicates whether the patient was prescribed diabetes medication during the encounter. |
| num_medications | The total number of medications prescribed to the patient during the encounter. |
| number_inpatient | The number of prior inpatient visits for the patient. |
| number_emergency | The number of prior emergency visits for the patient. |
| diag_1 / diag_2 / diag_3 | Primary and secondary diagnosis codes associated with the patient encounter. |

## Domain

This project belongs to the domain of healthcare analytics and hospital operations management. Hospitals collect large amounts of data during patient visits, including diagnoses, treatments, and patient demographics. Analyzing this data can help healthcare providers identify trends and improve decision-making. In particular, predicting or understanding hospital readmissions among diabetic patients has become a major focus because it can help hospitals improve patient care and reduce unnecessary healthcare spending.

| Title | Brief Description | Link |
|------|------------------|------|
| Hospital Readmission Risk and Risk Factors of People with a Primary or Secondary Discharge Diagnosis of Diabetes | Identifies key risk factors for 30-day readmission among diabetic patients, including comorbidities and length of stay. | [View Article](background_readings/Hospital Readmission Risk and Risk Factors of People with a Primary or Secondary Discharge Diagnosis of Diabetes.pdf) |
| A Systematic Review of Recent Studies on Hospital Readmissions of Patients With Diabetes | Summarizes recent research on diabetes-related readmissions and highlights major contributing factors. | [View Article](background_readings/A Systematic Review of Recent Studies on Hospital Readmissions of Patients With Diabetes.pdf) |
| Taking Steps in the Hospital to Prevent Diabetes-Related Readmissions | Describes strategies hospitals can use to reduce readmissions, such as patient education and discharge planning. | [View Article](background_readings/Taking Steps in the Hospital to Prevent Diabetes-Related Readmissions.md) |
| Diabetes | Provides an overview of diabetes, including causes, complications, and management. | [View Article](background_readings/diabetes.md) |
| Reducing Hospital Readmissions | Explains causes of hospital readmissions and general strategies to reduce them. | [View Article](background_readings/Reducing Hospital Readmissions.md) |




## Data Creation

For this project, I used data from the UCI Machine Learning Repository, specifically the Diabetes 130-US Hospitals Dataset. This dataset contains hospital encounter records for diabetic patients from 130 U.S. hospitals between 1999 and 2008, including patient demographics, clinical measurements, medications, and readmission outcomes. This dataset was selected because it includes the necessary variables to perform a meaningful analysis of hospital readmissions among diabetic patients and to support predictive modeling of readmission risk. The data was originally downloaded as a CSV file and loaded into Python using pandas. The dataset was then transformed into four separate relational tables: patients, encounters, clinical, and medications_outcomes. Each table was created by selecting relevant columns and removing duplicate records based on appropriate keys. The resulting tables were saved as parquet files for efficient storage and compatibility with DuckDB.

| File Name | Description | Link |
|----------|------------|------|
| data_creation.py | Loads the original dataset and splits it into four relational tables (patients, encounters, clinical, medications_outcomes), saving each as a parquet file | [View Code](data_creation.py) |
| diabetic_data.csv | Raw dataset used to construct the relational tables | [View Data](https://myuva-my.sharepoint.com/:f:/g/personal/twg3sr_virginia_edu/IgCBrNJwJpRyT5QfgCtf4UZVAf_5XkRYigD59zb_39zYzDU?e=bzV9zb) |

Bias in this dataset may arise in several ways. First, the dataset only includes patients with diabetes, which limits the ability to generalize findings to the broader population. Second, the data consists only of hospital encounter records, meaning it includes individuals who accessed healthcare and excludes those without access to medical services. Additionally, missing values in variables such as weight and medical specialty may introduce bias if certain groups are more likely to have incomplete data. Finally, the dataset spans from 1999 to 2008, so changes in medical practices over time may limit how well the data reflects current conditions.

Bias can be mitigated by carefully handling missing data, such as identifying patterns of missingness and avoiding inappropriate imputation. Results should be interpreted within the context of the dataset’s limitations, particularly its focus on diabetic patients and its historical time period. Stratifying analyses by demographic variables such as age, gender, or race can help identify differences across groups. Clearly communicating uncertainty and avoiding overgeneralization beyond the dataset can further reduce the impact of bias.

The dataset was split into four relational tables to reduce redundancy and better reflect real-world healthcare data structures. Patient-level attributes were separated from encounter-level data because a single patient may have multiple hospital visits. Clinical variables and medication data were further divided to improve organization and support more flexible analysis.

Parquet format was chosen for storage due to its efficiency and compatibility with DuckDB. A key decision was using patient_nbr and encounter_id as primary identifiers to define relationships between tables. However, some uncertainty remains because patient-level variables may vary across encounters, and using a single representation may not fully capture this variation. Additionally, missing values and inconsistencies in the original dataset may affect the reliability of certain features.


## Metadata
<img width="474" height="450" alt="Screenshot 2026-03-28 at 10 12 39 AM" src="https://github.com/user-attachments/assets/886cfdb3-0994-427a-a187-b35e69070a1c" />


| Table Name | Description | Link |
|------------|------------|------|
| patients | Patient-level demographic information, one row per patient | [patients.parquet](patients.parquet) |
| encounters | Encounter-level hospital visit information, one row per encounter | [encounters.parquet](encounters.parquet) |
| clinical | Clinical and diagnostic variables for each encounter | [clinical.parquet](clinical.parquet) |
| medications_outcomes | Medication data and readmission outcome for each encounter | [medications_outcomes.parquet](medications_outcomes.parquet) |


| Table | Name | Data Type | Description | Example |
|------|------|----------|------------|--------|
| patients | patient_nbr | integer | Unique identifier for each patient | 8222157 |
| patients | race | string | Patient race | Caucasian |
| patients | gender | string | Patient gender | Female |
| patients | age | string | Patient age range | [70-80) |
| patients | weight | string | Patient weight category | ? |
| encounters | encounter_id | integer | Unique identifier for each hospital encounter | 2278392 |
| encounters | patient_nbr | integer | Links to patients table | 8222157 |
| encounters | admission_type_id | integer | Encoded admission type | 6 |
| encounters | discharge_disposition_id | integer | Encoded discharge type | 25 |
| encounters | admission_source_id | integer | Encoded admission source | 1 |
| encounters | time_in_hospital | integer | Days spent in hospital | 1 |
| encounters | payer_code | string | Encoded payer type | ? |
| encounters | medical_specialty | string | Physician specialty | ? |
| encounters | number_outpatient | integer | Prior outpatient visits | 0 |
| encounters | number_emergency | integer | Prior emergency visits | 0 |
| encounters | number_inpatient | integer | Prior inpatient visits | 0 |
| clinical | encounter_id | integer | Links to encounters table | 2278392 |
| clinical | num_lab_procedures | integer | Number of lab procedures | 41 |
| clinical | num_procedures | integer | Number of procedures | 0 |
| clinical | num_medications | integer | Number of medications | 1 |
| clinical | number_diagnoses | integer | Number of diagnoses | 1 |
| clinical | diag_1 | string | Primary diagnosis | 250.83 |
| clinical | diag_2 | string | Secondary diagnosis | ? |
| clinical | diag_3 | string | Tertiary diagnosis | ? |
| clinical | max_glu_serum | string | Glucose test category | None |
| clinical | A1Cresult | string | A1C test category | None |
| medications_outcomes | encounter_id | integer | Links to encounters table | 2278392 |
| medications_outcomes | insulin | string | Insulin status | No |
| medications_outcomes | diabetesMed | string | Diabetes medication given | Yes |
| medications_outcomes | readmitted | string | Readmission outcome | NO |

**Uncertainty table needed


