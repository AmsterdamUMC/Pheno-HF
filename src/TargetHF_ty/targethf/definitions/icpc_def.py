 

dyspnea =       [r"K02",     # Dyspnea, cardiac attribution
                 r"R02"]     # Dyspnea, pulmonary attribution

edema =          r"K07"      # Ankle edema

heart_failure = [r"K77",     # Decompensatio cordis
                 r"K84.03"]  # Cardiomyopathie

atrial_fibrillation = [r"K78"]
valvular_heart_disease = [r"K83"]

outcomes_of_interest = {
        "HF" : heart_failure,
        "AF" : atrial_fibrillation,
        "VHD" : valvular_heart_disease
}

risk_factors = {"cvd_in_family":                 r"A29.01",
                "coronary_artery_disease":      [r"K74",     # Angina pectoris (stable + instable)
                                                 r"K75",     # Acute myocardial infarction
                                                 r"K76"],    # Other ischemic heart disease
                "atrial_fibrillation":           [r"K78",
                                                  r"K04",
                                                  r"K05",
                                                  r"K79",
                                                  r"K79.01",
                                                  r"K80",
                                                  r"K80.01"
                                                        ],             
                "heart_murmur":                  r"K81",
                "valvular_heart_disease":        [r"K83",
                                                  r"K83.01",
                                                  r"K83.02"
                                                        ],     # Non-reumatic valvular disease
                "hypertension":                 [r"K86",     # Essential hypertension
                                                 r"K87"],    # Essential hypertension (with organ damage/secondary)
                "stroke":                        r"K90",
                "copd":                         [r"R91",     # COPD/chronic bronchitis
                                                 r"R95"],    # COPD/emphysema
                "diabetes_mellitus":             r"T90",
                "chronic_kidney_disease":        r"U99.01",                
                "alcohol_abuse":                [r"P15",     # Acute
                                                 r"P16"],    # Chronic
                "tobacco_use":                   r"P17",
                "obesity":                      [r"T82",     # Adipositas
                                                 r"T83"],    # Obesitas
                "material_deprivation":          r"Z01"
}

risk_factors_old = {"coronary_artery_disease":      [r"K74",     # Angina pectoris (stable+instable)
                                                 r"K75",     # Acute myocardial infarction
                                                 r"K76"],    # Other ischemic heart disease
                "myocardial_infarction":        [r"K75",     # Acute myocardial infarction
                                                 r"K76.02"], # Prior myocardial infaction
                "hypertension":                 [r"K86",     # Essential hypertension
                                                 r"K87"],    # Essential hypertension (with organ damage/secondary)
                "diabetes_mellitus":             r"T90", 
                "atrial_fibrillation":           r"K78",            
                "valvular_heart_disease":        r"K83",     # Non-reumatic valvular disease
                "aortic_stenosis":               r"K83.01",
                "mitralis_insufficiency":        r"K83.02",
                "heart_murmur":                  r"K81",
                "chronic_kidney_disease":        r"U99.01",
                "aortic_aneurysm":               r"K99.01",
                "copd":                         [r"R91",     # COPD/chronic bronchitis
                                                 r"R95"],    # COPD/emphysema
                "thyroid_dysfunction":          [r"T85",     # Hyperthyroidism
                                                 r"T86"],    # Hypothyroidism
                "malignancy":                   [r"A79",     # Malignancy (unknown primary location)
                                                 r"B73",     # Leukemia
                                                 r"B74",     # Malignancy thyroid
                                                 r"D74",     # Malignancy stomach
                                                 r"D75",     # Malignancy colon/rectum
                                                 r"D76",     # Malignancy pancreas
                                                 r"D77",     # Malignancy digestive (other)
                                                 r"N74",     # Malignancy nervous system
                                                 r"R84",     # Malignancy lung/bronchi
                                                 r"R85",     # Malignancy respiratory (other)
                                                 r"S77.03",  # Malignancy skin (melanoma)
                                                 r"U75",     # Malignancy kidney
                                                 r"U76",     # Malignancy bladder
                                                 r"U77",     # Malignancy urinary tracts (other)
                                                 r"Y77",     # Malignancy prostate
                                                 r"X75",     # Malignancy cervix uteri
                                                 r"X76",     # Malignancy breast (female)
                                                 r"X77",     # Malignancy reproductive organ (female)
                                                 r"Y78"],    # Malignancy reproductive organ/breast (male)
                "chronic_alcohol_abuse":         r"P15",
                "alcoholism":                    r"P15.01",
                "delirium_tremens":              r"P15.02",
                "wernicke_korsakoff":            r"P15.03",
                "problematic_alcohol_use":       r"P15.05",
                "binge_drinking":                r"P15.06",
                "acute_alcohol_abuse":           r"P16",
                "substance_abuse":               r"P19",
                "tobacco_use":                   r"P17",
                "anemia":                       [r"B80",     # Iron deficiency anemia
                                                 r"B81",     # Pernicious anemia
                                                 r"B82",     # Anemia (other/NOS)
                                                 r"B78"],    # Hereditary hemolytic anemia
                "thalassemia":                   r"B78.01",
                "sickle_cell_disease":           r"B78.02",
                "obesity":                      [r"T82",     # Adipositas
                                                 r"T83"],    # Obesitas
                "depressive_disorder":          [r"P76",     # Depression
                                                 r"P77"],    # Suicide
                "anxiety_disorder":              r"P74",
                "psychiatric_disorder":         [r"P71",     # Organic/secondary psychosis (other)
                                                 r"P72",     # Schizophrenia
                                                 r"P73",     # Mood affective disorder
                                                 r"P79",     # Neurosis
                                                 r"P80",     # Personality disorder
                                                 r"P98",     # Psychosis (other)
                                                 r"P99"],    # Other psychological disorders
                "dementia":                      r"P70",
                "mental_retardation":            r"P85",
                "nutritional_deficiency":        r"T91",
                "inflammatory_bowel_disease":    r"D94",
                "rheumatic_arthritis":           r"L88",
                "cvd_in_family":                 r"A29.01"
}
