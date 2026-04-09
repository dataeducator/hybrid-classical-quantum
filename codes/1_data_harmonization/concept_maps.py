"""OMOP concept-ID mappings used throughout the TNBC ETL pipelines.

In production these would be looked up from the ATHENA vocabulary
(https://athena.ohdsi.org/).  For synthetic/dev work we hard-code
commonly used concept IDs so every pipeline shares the same semantics.

Data-lineage: source vocabulary → OMOP standard concept_id.
"""

from __future__ import annotations

# ── Gender ────────────────────────────────────────────────────────────
GENDER = {
    "Female": 8532,
    "Male": 8507,
    "F": 8532,
    "M": 8507,
}

# ── Race / Ethnicity ─────────────────────────────────────────────────
RACE = {
    "White": 8527,
    "Black": 8516,
    "Black or African American": 8516,
    "Asian": 8515,
    "Hispanic": 8557,
    "Other": 8522,
    "Unknown": 0,
}

ETHNICITY = {
    "Hispanic": 38003563,
    "Not Hispanic": 38003564,
    "Unknown": 0,
}

# ── Condition concepts (SNOMED) ───────────────────────────────────────
CONDITION = {
    "malignant_neoplasm_breast": 4112853,       # SNOMED 254837009
    "tnbc": 37018726,                           # Triple-negative breast cancer
    "breast_cancer_icd10_C50": 4112853,         # mapped from ICD-10-CM C50
}

# ── Biomarker / receptor measurement concepts ────────────────────────
MEASUREMENT = {
    "er_status": 4218106,     # Estrogen receptor status
    "pr_status": 4133438,     # Progesterone receptor status
    "her2_status": 4072465,   # HER2 receptor status
    "ki67": 4170688,          # Ki-67 proliferation index
    "tumor_size_cm": 4139750, # Tumor size in cm
    "wbc": 4298431,           # White blood cell count
    "hemoglobin": 4302946,    # Hemoglobin
    "platelets": 4267147,     # Platelet count
    "creatinine": 4013964,    # Serum creatinine
    "albumin": 4016239,       # Serum albumin
    "alp": 4229543,           # Alkaline phosphatase
    "ldh": 4166400,           # Lactate dehydrogenase
    "bmi": 4245997,           # Body-mass index
    "systolic_bp": 4152194,   # Systolic blood pressure
    "diastolic_bp": 4154790,  # Diastolic blood pressure
    "blood_lead": 4199035,    # Blood lead level
    "blood_cadmium": 4197238, # Blood cadmium level
    "blood_mercury": 4198453, # Blood mercury level
    "urinary_bpa": 4198001,   # Urinary BPA
    "tmb": 36304253,              # Tumor mutational burden
    "fraction_genome_altered": 36304254,  # Fraction of genome altered (CNA)
    "heart_rate": 4239408,        # Heart rate
    "steps_daily": 4198147,       # Step count / day
    "sleep_hours": 4202832,       # Sleep duration
}

# ── Biomarker value concepts (positive / negative) ───────────────────
BIOMARKER_VALUE = {
    "Negative": 4132135,
    "Positive": 4181412,
}

# ── Drug concepts (RxNorm-mapped) ────────────────────────────────────
DRUG = {
    "doxorubicin": 1338512,
    "cyclophosphamide": 1338005,
    "paclitaxel": 1378382,
    "carboplatin": 1344905,
    "pembrolizumab": 45892628,
    "capecitabine": 1390051,
}

# ── Procedure concepts (SNOMED) ──────────────────────────────────────
PROCEDURE = {
    "mastectomy": 4301351,
    "lumpectomy": 4048106,
    "radiation_therapy": 4029715,
    "chemotherapy": 4273629,
    "sentinel_node_biopsy": 4058390,
}

# ── Observation concepts (surveys / exposures) ───────────────────────
OBSERVATION = {
    "smoking_status": 4275495,
    "income_level": 4076114,
    "education_level": 4171274,
    "insurance_type": 4058589,
    "poverty_income_ratio": 4076114,
    "total_calories": 4033249,
    "dietary_fiber": 4024570,
    "fruit_veg_servings": 4218684,
    "brca1_variant": 4040289,
    "brca2_variant": 4040290,
    "cancer_self_report": 4048809,
}

# ── AJCC stage mapping ───────────────────────────────────────────────
STAGE = {
    "I": 4127942,
    "IA": 4127943,
    "IB": 4127944,
    "II": 4127945,
    "IIA": 4127946,
    "IIB": 4127947,
    "III": 4127948,
    "IIIA": 4127949,
    "IIIB": 4127950,
    "IIIC": 4127951,
    "IV": 4127952,
}

# ── Type concepts (provenance) ───────────────────────────────────────
TYPE_CONCEPT = {
    "ehr": 32817,           # EHR record
    "registry": 32879,      # Tumor registry
    "survey": 32862,        # Survey / questionnaire
    "claim": 32810,         # Insurance claim
    "lab": 32856,           # Lab result
    "wearable": 32880,      # Device / wearable
}

# ── Data source tags ─────────────────────────────────────────────────
SOURCE_TAG = {
    "seer": "SEER",
    "mimic": "MIMIC-IV",
    "tcga": "TCGA-BRCA",
}
