# DS 4320 Project 1: Identifying Factors Associated with 30-Day Hospital Readmission in Diabetic Patients

**Executive Summary:** This repository contains a data science project focused on predicting 30-day hospital readmission risk among diabetic patients. It includes a relational dataset built from hospital encounter data, a Python and DuckDB pipeline for analysis, and a press release that summarizes key findings for a general audience.

**Name:** Grace Pitts

**Net ID:** twg3sr

**DOI:** link**

**Press Release:** [Press Release](press_release.md)

**Data:** [Data](https://myuva-my.sharepoint.com/:f:/r/personal/twg3sr_virginia_edu/Documents/Project1%20Data?csf=1&web=1&e=v0GJec)

**Pipeline Notebook:** [Pipeline Notebook](pipeline/hospital_readmission_pipeline.ipynb)

**Pipeline Markdown:** [Pipeline Markdown](pipeline/hospital_readmission_pipeline.md)

**License:** This project is licensed under the MIT License. See the [License](LICENSE) file for details.




## Problem Definition
**Initial Problem:** Predicting Hospital Readmission Risk.

**Refined Problem Statement**: Identify patient characteristics and hospital encounter patterns associated with 30-day readmission among patients with diabetes to support interventions that reduce readmission rates.

**Refinement Rationale:** The general problem of predicting hospital readmission is broad and encompasses a wide range of factors, including patient demographics, medical conditions, treatments, and hospital practices. To make the project more focused and aligned with the available data, the problem was refined to examine 30-day readmission specifically among diabetic patients using hospital encounter data. This refinement uses a clearly defined group of patients with detailed data, which allows for a more focused and meaningful analysis. By narrowing the scope to diabetic patients, the project reduces complexity while still maintaining clinical relevance. This allows for clearer identification of patterns and more interpretable insights into factors associated with readmission risk.

**Project Motivation**: Hospital readmissions are a major challenge in healthcare systems because they often indicate complications, poor care coordination, or inadequate follow-up treatment after discharge. Unplanned readmissions increase healthcare costs and can negatively impact patient outcomes. This issue is especially important for diabetic patients, who often require ongoing care and are at higher risk for complications that can lead to repeat hospital visits. Hospitals and healthcare providers are increasingly interested in understanding the factors that contribute to readmission so they can improve patient care and reduce unnecessary hospital visits. This project is motivated by the potential to use data analysis to better understand patterns in hospital admissions among diabetic patients and identify characteristics associated with higher readmission risk, which can hopefully help to lower the amount of readmissions.

**Press Release:** [Predicting Hospital Readmissions to Improve Patient Care](press_release.md)


## Domain Exposition

**Terminology:**
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

**Project Domain:** 
This project belongs to the domain of healthcare analytics and hospital operations management. Hospitals collect large amounts of data during patient visits, including diagnoses, treatments, and patient demographics. Analyzing this data can help healthcare providers identify trends and improve decision-making. In particular, predicting or understanding hospital readmissions among diabetic patients has become a major focus because it can help hospitals improve patient care and reduce unnecessary healthcare spending.

**Background Readings:**
Background readings can be found here: [link](https://myuva-my.sharepoint.com/:f:/g/personal/twg3sr_virginia_edu/IgDrX2KAueGcSIfQsf1QgLJoAfLvcSf2wZvuAKC5agkNjus?e=xINZUx)

**Reading Descriptions:**
| Title | Brief Description | Link |
|------|------------------|------|
| Hospital Readmission Risk and Risk Factors of People with a Primary or Secondary Discharge Diagnosis of Diabetes | Identifies key risk factors for 30-day readmission among diabetic patients, including comorbidities and length of stay. | [View Article](https://myuva-my.sharepoint.com/:b:/g/personal/twg3sr_virginia_edu/IQDW_bYGI8XGR6XX0mP4WtK3AcTbWXTKW95g29477YhPojE?e=GPei6N) |
| A Systematic Review of Recent Studies on Hospital Readmissions of Patients With Diabetes | Summarizes recent research on diabetes-related readmissions and highlights major contributing factors. | [View Article](https://myuva-my.sharepoint.com/:b:/g/personal/twg3sr_virginia_edu/IQA7UGv3SE3LQqKxZkSoF5ocAdT3l-PhHTAyQjm51hcRHGw?e=7HdjXy) |
| Taking Steps in the Hospital to Prevent Diabetes-Related Readmissions | Describes strategies hospitals can use to reduce readmissions, such as patient education and discharge planning. | [View Article](https://myuva-my.sharepoint.com/:b:/g/personal/twg3sr_virginia_edu/IQBYpmASwbRYTIfkQCJCTAhbAQRDT1hd-E0mhaCbCcP7rno?e=palDGw)|
| Diabetes | Provides an overview of diabetes, including causes, complications, and management. | [View Article](https://myuva-my.sharepoint.com/:b:/g/personal/twg3sr_virginia_edu/IQDkNFDBA3PIS6Pg9jkogTeyAVq7kUJ60i-tOxZh3Jt7zZE?e=aiqW9J) |
| Reducing Hospital Readmissions | Explains causes of hospital readmissions and general strategies to reduce them. | [View Article](https://myuva-my.sharepoint.com/:b:/g/personal/twg3sr_virginia_edu/IQDHoentpB16TacRvWjr7pb1AUd1EBYqaKtsl0bAvExPgvI?e=wuiTxf) |



## Data Creation

**Raw Data Accquisition:**
For this project, I used data from the UCI Machine Learning Repository, specifically the Diabetes 130-US Hospitals Dataset. This dataset contains hospital encounter records for diabetic patients from 130 U.S. hospitals between 1999 and 2008, including patient demographics, clinical measurements, medications, and readmission outcomes. This dataset was selected because it includes the necessary variables to perform a meaningful analysis of hospital readmissions among diabetic patients and to support predictive modeling of readmission risk. The data was originally downloaded as a CSV file and loaded into Python using pandas. The dataset was then transformed into four separate relational tables: patients, encounters, clinical, and medications_outcomes. Each table was created by selecting relevant columns and removing duplicate records based on appropriate keys. The resulting tables were saved as parquet files for efficient storage and compatibility with DuckDB.

**Code Provenance Table:**
| File Name | Description | Link |
|----------|------------|------|
| data_creation.py | Loads the original dataset and splits it into four relational tables (patients, encounters, clinical, medications_outcomes), saving each as a parquet file | [View Code](data_creation.py) |
| diabetic_data.csv | Raw dataset used to construct the relational tables | [View Data](https://myuva-my.sharepoint.com/:f:/g/personal/twg3sr_virginia_edu/IgCBrNJwJpRyT5QfgCtf4UZVAf_5XkRYigD59zb_39zYzDU?e=bzV9zb) |

**Bias Identification:**
Bias in this dataset could arise in several ways. One way is that the dataset only includes patients with diabetes, which limits the ability to generalize findings to the broader population. Another way bias could arise is because the data consists only of hospital encounter records, meaning it includes individuals who accessed healthcare and excludes those without access to medical services. Bias could also be introduced due to missing values in variables such as weight and medical specialty may introduce bias if certain groups are more likely to have incomplete data. Finally, the dataset spans from 1999 to 2008, so changes in medical practices over time may limit how well the data reflects current conditions.

**Bias Mitigation:**
Bias can be mitigated by carefully handling missing data, such as identifying patterns of missingness and avoiding inappropriate imputation. Results should be interpreted within the context of the dataset’s limitations, particularly its focus on diabetic patients and its time period. Stratifying analyses by demographic variables such as age, gender, or race can help identify differences across groups. Clearly communicating uncertainty and avoiding overgeneralization beyond the dataset can further reduce the impact of bias.

**Decision Rationale:** 
The dataset was split into four relational tables to reduce redundancy and better reflect real-world healthcare data structures. Patient-level attributes were separated from encounter-level data because a single patient may have multiple hospital visits. Clinical variables and medication data were further divided to improve organization and support more flexible analysis. Parquet format was chosen for storage due to its efficiency and compatibility with DuckDB. A key decision was using patient_nbr and encounter_id as primary identifiers to define relationships between tables. However, some uncertainty remains because patient-level variables may vary across encounters, and using a single representation may not fully capture this variation. Additionally, missing values and inconsistencies in the original dataset may affect the reliability of certain features.


## Metadata

**ERD:**
<img width="474" height="450" alt="Screenshot 2026-03-28 at 10 12 39 AM" src="https://github.com/user-attachments/assets/886cfdb3-0994-427a-a187-b35e69070a1c" />

**Data Table:**
| Table Name | Description | Link |
|------------|------------|------|
| patients | Patient-level demographic information, one row per patient | [patients.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/twg3sr_virginia_edu/IQBK_1NokQTsTq5WyU8SaKDgAZcyswwj80jzQTX26gCu4yo?e=3RUUWI) |
| encounters | Encounter-level hospital visit information, one row per encounter | [encounters.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/twg3sr_virginia_edu/IQDJ0r5RoG6yRakI6xiLKXDAAR5GGeIdbEu07zgr30-rjiQ?e=skdcoZ) |
| clinical | Clinical and diagnostic variables for each encounter | [clinical.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/twg3sr_virginia_edu/IQBOYMXyqq7wT6a8pt0p-RmdAc2JmX7h2agFWNhnGdSfOb8?e=O9lnPG) |
| medications_outcomes | Medication data and readmission outcome for each encounter | [medications_outcomes.parquet](https://myuva-my.sharepoint.com/:u:/g/personal/twg3sr_virginia_edu/IQCdTWMRxN2LT6B_aA1LwnofAfjj4mvQmlMKMoGLoDIdbUE?e=oNnyoZ) |

**Data Dictionary:**
| Table | Feature | Data Type | Description | Example | % Missing | Mean | Median | Std Dev | Interpretation |
|------|--------|----------|------------|--------|----------|------|--------|--------|----------------|
| patients | patient_nbr | integer | Unique identifier for each patient | 8222157 | — | — | — | — | — |
| patients | race | string | Patient race | Caucasian | — | — | — | — | — |
| patients | gender | string | Patient gender | Female | — | — | — | — | — |
| patients | age | string | Patient age range | [70-80) | — | — | — | — | — |
| patients | weight | string | Patient weight category | ? | — | — | — | — | — |
| encounters | encounter_id | integer | Unique identifier for each hospital encounter | 2278392 | — | — | — | — | — |
| encounters | patient_nbr | integer | Links to patients table | 8222157 | — | — | — | — | — |
| encounters | admission_type_id | integer | Encoded admission type | 6 | — | — | — | — | — |
| encounters | discharge_disposition_id | integer | Encoded discharge type | 25 | — | — | — | — | — |
| encounters | admission_source_id | integer | Encoded admission source | 1 | — | — | — | — | — |
| encounters | time_in_hospital | integer | Days spent in hospital | 1 | 0.0 | 4.40 | 4.0 | 2.99 | Slight right skew, most stays are short, but some longer stays increase uncertainty |
| encounters | payer_code | string | Encoded payer type | ? | — | — | — | — | — |
| encounters | medical_specialty | string | Physician specialty | ? | — | — | — | — | — |
| encounters | number_outpatient | integer | Prior outpatient visits | 0 | 0.0 | 0.36 | 0.0 | 1.08 | Strong right skew, most patients have none, but outliers increase uncertainty |
| encounters | number_emergency | integer | Prior emergency visits | 0 | 0.0 | 0.20 | 0.0 | 0.93 | Strong right skew, few high values drive variability and uncertainty |
| encounters | number_inpatient | integer | Prior inpatient visits | 0 | 0.0 | 0.64 | 0.0 | 1.26 | Right skew with high spread, small group drives uncertainty |
| clinical | encounter_id | integer | Links to encounters table | 2278392 | — | — | — | — | — |
| clinical | num_lab_procedures | integer | Number of lab procedures | 41 | 0.0 | 43.10 | 44.0 | 19.67 | Wide spread, differences in testing intensity increase uncertainty |
| clinical | num_procedures | integer | Number of procedures | 0 | 0.0 | 1.34 | 1.0 | 1.71 | Right skew, some patients have many procedures, increasing uncertainty |
| clinical | num_medications | integer | Number of medications | 1 | 0.0 | 16.02 | 15.0 | 8.13 | High variability, treatment complexity introduces uncertainty |
| clinical | number_diagnoses | integer | Number of diagnoses | 1 | 0.0 | 7.42 | 8.0 | 1.93 | Low variability, more consistent across patients, lower uncertainty |
| clinical | diag_1 | string | Primary diagnosis | 250.83 | — | — | — | — | — |
| clinical | diag_2 | string | Secondary diagnosis | ? | — | — | — | — | — |
| clinical | diag_3 | string | Tertiary diagnosis | ? | — | — | — | — | — |
| clinical | max_glu_serum | string | Glucose test category | None | — | — | — | — | — |
| clinical | A1Cresult | string | A1C test category | None | — | — | — | — | — |
| medications_outcomes | encounter_id | integer | Links to encounters table | 2278392 | — | — | — | — | — |
| medications_outcomes | insulin | string | Insulin status | No | — | — | — | — | — |
| medications_outcomes | diabetesMed | string | Diabetes medication given | Yes | — | — | — | — | — |
| medications_outcomes | readmitted | string | Readmission outcome | NO | — | — | — | — | — |
| medications_outcomes | diabetes_medications | string (multiple) | Indicators for specific diabetes medications (ex. metformin, insulin, etc.) | No | — | — | — | — | — |
| medications_outcomes | change | string | Whether medication was changed | No | — | — | — | — | — |


