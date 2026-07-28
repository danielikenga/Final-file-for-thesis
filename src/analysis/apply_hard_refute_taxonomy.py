import pandas as pd

INPUT_PATH = (
    "results/analysis/cross_experiment_errors/"
    "case_analysis/persistent_hard_refutes_manual_review.csv"
)

OUTPUT_PATH = (
    "results/analysis/cross_experiment_errors/"
    "case_analysis/persistent_hard_refutes_reviewed.csv"
)

df = pd.read_csv(INPUT_PATH)

review_columns = [
    "primary_error_type",
    "secondary_error_type",
    "contradiction_explicitness",
    "requires_world_knowledge",
    "translation_issue_present",
    "annotation_suspected",
    "review_notes",
]

for column in review_columns:
    if column not in df.columns:
        df[column] = ""
    df[column] = df[column].fillna("").astype("object")

reviews = {
    "afrifact_data_culture_hausa_058": {
        "primary_error_type": "entity_attribute_mismatch",
        "secondary_error_type": "implicit_contradiction",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "no",
        "review_notes": "Claim identifies Dan Fodiyo as Hausa; evidence identifies Toronkawa Fulani origin."
    },

    "afrifact_data_culture_hausa_064": {
        "primary_error_type": "temporal_mismatch",
        "secondary_error_type": "role_relation_mismatch",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "uncertain",
        "review_notes": "Claim treats 1533 as accession year; evidence presents 1533 as birth/start-of-lifespan year."
    },

    "afrifact_data_culture_hausa_199": {
        "primary_error_type": "causal_mismatch",
        "secondary_error_type": "temporal_mismatch",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "no",
        "review_notes": "Claim attributes Islamisation to colonial arrival; evidence attributes it to Islam arriving in the 14th-15th centuries."
    },

    "afrifact_data_culture_hausa_392": {
        "primary_error_type": "evidence_insufficient",
        "secondary_error_type": "possible_annotation_issue",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence describes JIBWIS as a religious organisation combating innovations in religion but does not directly disprove cooking or peace meetings."
    },

    "afrifact_data_culture_hausa_463": {
        "primary_error_type": "implicit_contradiction",
        "secondary_error_type": "entity_attribute_mismatch",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "uncertain",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "uncertain",
        "review_notes": "Evidence lists eligible zakat recipient categories and omits wealthy people seeking investment capital; refutation depends on interpreting the list as exhaustive."
    },

    "afrifact_data_culture_hausa_472": {
        "primary_error_type": "boundary_condition",
        "secondary_error_type": "temporal_mismatch",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Evidence defines fasting as abstention from intercourse from dawn until sunset, directly opposing permission before sunset."
    },

    "afrifact_data_culture_hausa_523": {
        "primary_error_type": "evidence_insufficient",
        "secondary_error_type": "possible_annotation_issue",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence reports competing allegations about perpetrators and possible responsibility; it does not establish that the Nigerian government did not order the killing."
    },

    "afrifact_data_culture_hausa_799": {
        "primary_error_type": "entity_attribute_mismatch",
        "secondary_error_type": "role_relation_mismatch",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim identifies Sofia Sadek as a primary-school teacher; evidence identifies her as a Tunisian singer."
    },

    "afrifact_data_culture_igbo_151": {
        "primary_error_type": "negation",
        "secondary_error_type": "",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim says Ideato South people are not traders; evidence explicitly says they are traders."
    },

    "afrifact_data_culture_igbo_323": {
        "primary_error_type": "role_relation_mismatch",
        "secondary_error_type": "entity_attribute_mismatch",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "uncertain",
        "review_notes": "Claim describes Aghanya as a teacher who fled with students; evidence identifies him as a soldier and electrical engineer serving in the Biafran army."
    },

    "afrifact_data_culture_igbo_347": {
        "primary_error_type": "role_relation_mismatch",
        "secondary_error_type": "temporal_mismatch",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "no",
        "review_notes": "Claim assigns governorship of Anambra in the Second Republic; evidence assigns a ministerial agriculture role during 1959-1966/First Republic."
    },

    "afrifact_data_culture_igbo_625": {
        "primary_error_type": "evidence_insufficient",
        "secondary_error_type": "possible_annotation_issue",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence concerns health screening and early airport arrival; it does not directly address or contradict advice to dress warmly for cold weather."
    },

    "afrifact_data_culture_yoruba_077": {
        "primary_error_type": "temporal_mismatch",
        "secondary_error_type": "negation",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim says the rule would stop operating on 27 March 2020; evidence says the directive would begin on that date."
    },

    "afrifact_data_culture_yoruba_158": {
        "primary_error_type": "negation",
        "secondary_error_type": "role_relation_mismatch",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "no",
        "review_notes": "Claim denies any relationship between Yoruba and Kwa; evidence states a relationship."
    },

    "afrifact_data_culture_yoruba_179": {
        "primary_error_type": "entity_attribute_mismatch",
        "secondary_error_type": "role_relation_mismatch",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim attributes wealth, honour and success powers to Sagi; evidence attributes these powers to Olokun."
    },

    "afrifact_data_culture_yoruba_290": {
        "primary_error_type": "evidence_insufficient",
        "secondary_error_type": "possible_annotation_issue",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence discusses Igboho's authority to order people to leave and legal rights of residents but does not clearly resolve the kidnapping attribution in the claim."
    },

    "afrifact_data_culture_yoruba_377": {
        "primary_error_type": "negation",
        "secondary_error_type": "implicit_contradiction",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim says citizens refused to praise Osinbajo; evidence states Nigerians were applauding him for actions taken while acting for Buhari."
    },

    "afrifact_data_culture_yoruba_383": {
        "primary_error_type": "evidence_insufficient",
        "secondary_error_type": "possible_annotation_issue",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence states pilgrims must not touch the Black Stone and must distance, but does not directly disprove alleged videos of special guests secretly touching it."
    },

    "afrifact_data_culture_yoruba_431": {
        "primary_error_type": "evidence_insufficient",
        "secondary_error_type": "possible_annotation_issue",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence discusses legal challenge, detention and allegations of insubordination/failure to attend meetings; it does not directly resolve the claim of corruption or misappropriation as removal reason."
    },

    "afrifact_data_culture_yoruba_548": {
        "primary_error_type": "possible_annotation_issue",
        "secondary_error_type": "evidence_insufficient",
        "contradiction_explicitness": "ambiguous",
        "requires_world_knowledge": "uncertain",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "yes",
        "review_notes": "Evidence is malformed/incomplete and itself contains wording that appears to say this would be Obasanjo's third time as president; reliable refutation is difficult."
    },

    "afrifact_data_culture_yoruba_557": {
        "primary_error_type": "numerical_mismatch",
        "secondary_error_type": "implicit_contradiction",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "no",
        "review_notes": "Claim says only one type of garri exists; evidence states garri is divided into multiple varieties."
    },

    "afrifact_data_culture_yoruba_566": {
        "primary_error_type": "boundary_condition",
        "secondary_error_type": "negation",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim restricts sacrifice to bad Ifa outcomes; evidence says sacrifice is important whether the divination outcome is good or bad."
    },

    "afrifact_data_culture_yoruba_578": {
        "primary_error_type": "entity_attribute_mismatch",
        "secondary_error_type": "implicit_contradiction",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim lists calcium, carbohydrate and proteins; evidence lists proteins, vitamins and minerals such as calcium, iron and zinc, not carbohydrate."
    },

    "afrifact_data_culture_yoruba_635": {
        "primary_error_type": "role_relation_mismatch",
        "secondary_error_type": "entity_attribute_mismatch",
        "contradiction_explicitness": "implicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "uncertain",
        "annotation_suspected": "uncertain",
        "review_notes": "Claim describes Federal Government Girls' College as private and adds a location relation; evidence distinguishes a private university from government/Federal Government Girls' College, but wording is imperfect."
    },

    "afrifact_data_culture_yoruba_749": {
        "primary_error_type": "entity_attribute_mismatch",
        "secondary_error_type": "role_relation_mismatch",
        "contradiction_explicitness": "explicit",
        "requires_world_knowledge": "no",
        "translation_issue_present": "no",
        "annotation_suspected": "no",
        "review_notes": "Claim attributes the Oyo-origin view and illegitimate/slave statement to Delana; evidence explicitly attributes them to Johnson."
    },
}

for idx, row in df.iterrows():
    example_id = row["id"]

    if example_id not in reviews:
        raise ValueError(f"Missing manual review for: {example_id}")

    for column, value in reviews[example_id].items():
        df.at[idx, column] = value

df.to_csv(OUTPUT_PATH, index=False)

print("Saved reviewed taxonomy to:")
print(OUTPUT_PATH)
print("Rows:", len(df))

print("\nPRIMARY ERROR TYPE COUNTS")
print(df["primary_error_type"].value_counts().to_string())

print("\nCONTRADICTION EXPLICITNESS")
print(df["contradiction_explicitness"].value_counts().to_string())

print("\nANNOTATION SUSPECTED")
print(df["annotation_suspected"].value_counts().to_string())

print("\nBY LANGUAGE AND PRIMARY ERROR TYPE")
print(
    pd.crosstab(
        df["language"],
        df["primary_error_type"]
    ).to_string()
)