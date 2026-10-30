import mysql.connector
import pandas as pd
import os
import getpass

# Ask for password securely at runtime
db_password = getpass.getpass("Enter your MySQL root password: ")

# Connect to ChEMBL MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=db_password,
    database="chembl_37"
)

print("Connected to ChEMBL!")

# Create output folder
os.makedirs("data", exist_ok=True)

# ── 1. Extract drug molecules ──────────────────────────────────────────────
print("Extracting drugs...")
drug_query = """
SELECT
    md.molregno,
    md.pref_name,
    md.chembl_id,
    md.molecule_type,
    md.max_phase
FROM molecule_dictionary md
WHERE md.max_phase >= 2
  AND md.pref_name IS NOT NULL
LIMIT 10000;
"""
drugs_df = pd.read_sql(drug_query, conn)
drugs_df.to_csv("data/drugs.csv", index=False)
print(f"  Saved {len(drugs_df)} drugs -> data/drugs.csv")

# ── 2. Extract drug → target (enzyme/transporter) relationships ────────────
print("Extracting drug-target interactions...")
interaction_query = """
SELECT
    md.chembl_id        AS drug_chembl_id,
    md.pref_name        AS drug_name,
    td.chembl_id        AS target_chembl_id,
    td.pref_name        AS target_name,
    td.target_type,
    act.standard_type   AS activity_type,
    act.standard_value  AS activity_value,
    act.standard_units  AS activity_units
FROM activities act
JOIN assays a           ON act.assay_id      = a.assay_id
JOIN target_dictionary td ON a.tid           = td.tid
JOIN molecule_dictionary md ON act.molregno  = md.molregno
WHERE td.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX')
  AND act.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
  AND act.standard_value IS NOT NULL
  AND md.max_phase >= 2
LIMIT 500000;
"""
interactions_df = pd.read_sql(interaction_query, conn)
interactions_df.to_csv("data/drug_target_interactions.csv", index=False)
print(f"  Saved {len(interactions_df)} interactions -> data/drug_target_interactions.csv")

conn.close()
print("\nDone! Data extraction complete.")
