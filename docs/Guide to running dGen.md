**Single-State dGen Run — Researcher Guide**

# **How the system fits together**

The model never runs on your laptop. Here is the complete data flow:

YOU edit price CSVs in repo  
        ↓  
YOU run notebook → prices upload to Cloud SQL  
        ↓  
YOU edit state CSV → YOU upload to GCS  
        ↓  
YOU run build\_and\_submit.sh  
        ↓  (automatic from here)  
Docker image built & pushed to Google container registry  
Cloud Batch VM starts, pulls Docker image  
Downloads scenario Excel templates \+ state CSV from GCS  
Injects state name \+ end year into Excel files  
dgen\_model.py runs → reads prices from Cloud SQL  
Writes results back to Cloud SQL as new schemas  
        ↓  
YOU run export notebook → results land on your laptop as CSVs  
        ↓  
YOU run analysis notebook → outputs and charts

# **Linked filenames — know before you start**

Some filenames appear in multiple places and must stay in sync. Before you start, understand these connections:

| Filename | Where it appears | What breaks if mismatched |
| :---- | :---- | :---- |
| mid\_states\_test.csv | state\_input\_csvs/mid\_states\_test.csv (your repo) → uploaded to GCS → fetched by YAML | YAML downloads the wrong/nonexistent file and job fails |
| pv\_plus\_batt\_prices\_FY23\_{STATE}\_baseline.csv | Your repo → read by adjust\_state\_level\_prices.ipynb | Notebook fails to find the file |
| pv\_plus\_batt\_prices\_FY23\_{STATE}\_policy.csv | Your repo → read by adjust\_state\_level\_prices.ipynb | Notebook fails to find the file |
| YAML file referenced in submit\_one.sh | submit\_one.sh line 14 (\--config=...) and batch\_job\_yamls/ folder | Job submits with wrong machine size or wrong state list |
| Price table names | created by upload\_state\_price\_tables.py → named in your state's YAML env vars | Job silently runs on the shared national prices instead of your state's |

# **Step-by-step**

| Step | YOU do this | AUTOMATIC in background | File(s) you touch |
| ----- | ----- | ----- | ----- |
| **1\. Set your solar prices** | Edit system\_capex\_per\_kw\_res for years 2026–2040 in the policy CSV. Leave all other columns alone. If no file exists for your state yet, copy an existing state's files and rename them. | Nothing yet — file just sits in the repo. | `dgen_os/input_data/pv_plus_batt_prices/<br>pv_plus_batt_prices_FY23_{STATE}_baseline.csv<br>pv_plus_batt_prices_FY23_{STATE}_policy.csv` |
| **2\. Start Cloud SQL** | Go to GCP Console → Cloud SQL → Start the instance. Wait for green status. | Nothing. | — |
| **3\. Start Cloud SQL Proxy** | Run cloud-sql-proxy dgen-466702:us-east1:dgen-db in a dedicated terminal. Leave it running. | Opens a secure tunnel so your laptop talks to Cloud SQL on localhost:5432. | — |
| **4\. Upload prices to Cloud SQL** | From `dgen_os/python/`, run `python upload_state_price_tables.py --states XX` (add `--dry-run` first to check the numbers). | Reads your two price CSVs and writes them to **their own** tables — `pv_price_XX_baseline` / `_policy` and `pv_plus_batt_XX_baseline` / `_policy`. The shared national tables are left untouched, so state studies do not overwrite each other. | `dgen_os/python/upload_state_price_tables.py` |
| **5\. Set your target state** | Edit the CSV to contain exactly one line: XX,Full State Name. Then upload it to GCS: gsutil cp state\_input\_csvs/mid\_states\_test.csv gs://dgen-assets/mid\_states\_test.csv | Nothing yet. | `state_input_csvs/mid_states_test.csv` |
| **6\. Authenticate to GCP (one-time)** | Run three gcloud auth commands in Mac terminal. | GCP stores credentials locally so Docker and gcloud can push/submit. | — |
| **7\. Build & submit** | Run bash build\_and\_submit.sh from repo root. Walk away for 5–15 min while it builds. | (a) Packages all model code \+ conda env into a Docker image, pushes it to Google's registry. (b) Submits a Batch job. (c) GCP spins up a VM, pulls the image, downloads your scenario Excel templates \+ state CSV from GCS, injects state name \+ end year into the Excels, runs dgen\_model.py. Model reads prices from Cloud SQL, runs baseline then policy, writes results back to Cloud SQL. | `build_and_submit.sh → submit_one.sh → batch_job_yamls/dgen-batch-job-mid-states.yaml` |
| **8\. Monitor** | GCP Console → Batch → Jobs → find dgen-{state}-{timestamp} → check Logs. Confirm you see the correct state name. | Job runs 1–3 hours. Logs stream in real time. | — |
| **9\. Export results** | In the notebook: set OUT\_DIR to your local folder and RUN\_ID to something descriptive. Run all cells. | Notebook connects to Cloud SQL (via proxy), finds all diffusion\_results\_\* schemas from your run, exports them as CSVs to your laptop. | `Notebooks/loading_model_results.ipynb` |
| **10\. Analyze** | Open and run the analysis notebook. Point it at your exported folder. | Computes adoption, capacity, net load, savings metrics and generates charts. | `Notebooks/analysis_of_model_results.ipynb` |
| **11\. Stop Cloud SQL** | GCP Console → Cloud SQL → Stop. Don't delete. | — | — |

## **Step 1 — Create or edit your price CSVs**

**YOU DO:** 

If files already exist for your state (e.g., pv\_plus\_batt\_prices\_FY23\_MD\_baseline.csv), open them and edit system\_capex\_per\_kw\_res for years 2026–2040 only.

If no state-specific files exist yet, create them by copying the template files:

* Copy pv\_plus\_batt\_prices\_FY23\_mid.csv → save as pv\_plus\_batt\_prices\_FY23\_{STATE}\_baseline.csv

* Copy **the same** pv\_plus\_batt\_prices\_FY23\_mid.csv → save as pv\_plus\_batt\_prices\_FY23\_{STATE}\_policy.csv

> **Both arms come from FY23\_mid — not FY23\_dollar\_watt.** An earlier version of this step said
> to copy FY23\_dollar\_watt.csv for the policy arm. Don't: that template carries a different
> battery cost (about $708/kWh in 2030 against $366/kWh in mid, roughly 2x), and since you
> overwrite the PV column in both files anyway, battery cost would become the only other thing
> separating your two arms — showing up in results as if the policy made storage more expensive.
> Every existing state pair (IL, MA, MD, NC, NY, OH, PA, TX, VA) was in fact built from mid for
> both arms; audited across all nine, the only column that meaningfully differs between baseline
> and policy is system\_capex\_per\_kw\_res, everything else matching to ~1e-9. Keep it that way and
> the run isolates the price policy.

Then edit system\_capex\_per\_kw\_res for years 2026–2040 in each. Units are $/kW (so $2.50/W = 2500). Leave all other columns unchanged.

* The policy price, if the state exists, should be pulled from Row 78 of the OS Projections tab in the Price Projections google sheet [Permit Power Calculations v1.2.3.xlsx](https://docs.google.com/spreadsheets/d/1y9uM875xOK9upk1UI7icBb7LhFA_zs_j/edit?gid=1433908332#gid=1433908332). Copy over the prices from 2026 \- 2040 only. The units should be in $/kW, so multiply the values by 1,000.   
* The baseline price should come from NREL, as calculated by the adjust\_state\_level\_prices.ipynb notebook. Update the state, take the baseline price in 2024, and apply the % decrease per year as calculated in that sheet, but not using a value over 1%.   
  * If the state isn’t available in the NREL Tracking the Sun data, then either use the national average as the baseline, or use the baseline from the Permit Power Calculations sheet (if the national average is wildly different than the policy scenario costs). 

**AUTOMATIC:** Nothing yet — files just sit in the repo.

**Files:** dgen\_os/input\_data/pv\_plus\_batt\_prices/pv\_plus\_batt\_prices\_FY23\_{STATE}\_baseline.csv and ...\_policy.csv

## **Step 2 — Start Cloud SQL**

**YOU DO:** GCP Console → Cloud SQL → dgen-db → Start. Wait for the green status indicator.

**AUTOMATIC:** Nothing.

**Note:** The instance costs money while running. You will stop it at the end.

## **Step 3 — Start the Cloud SQL Proxy**

**YOU DO:** Open a dedicated Mac terminal window and run:

```shell
gcloud auth application-default login
cloud-sql-proxy dgen-466702:us-east1:dgen-db
```

Leave this running for the entire session. Do not close this window until you have finished exporting results.

**AUTOMATIC:** Opens a secure tunnel so your laptop can reach Cloud SQL at localhost:5432, which is how your Jupyter notebooks connect to the database.

## **Step 4 — Upload prices to Cloud SQL**

**YOU DO:** From `dgen_os/python/`, check the numbers first, then upload:

```
python upload_state_price_tables.py --states XX --dry-run
python upload_state_price_tables.py --states XX
```

You can pass several states at once (`--states PA OH`).

**AUTOMATIC:** Reads your two price CSVs and writes each to its **own** table in diffusion\_shared:

* pv\_price\_xx\_baseline and pv\_plus\_batt\_xx\_baseline ← from your baseline CSV

* pv\_price\_xx\_policy and pv\_plus\_batt\_xx\_policy ← from your policy CSV

The tables carry a state\_abbr column, and the job points at them through the env vars in its
YAML (Step 6). Nothing shared is overwritten.

> **Why this changed.** The old route was adjust\_state\_level\_prices.ipynb, which wrote your CSVs
> straight over pv\_plus\_batt\_baseline and pv\_plus\_batt\_dollar\_per\_watt — the shared tables. Those
> now hold the 49-state LBNL baseline and the $1/W policy trajectory that the national runs depend
> on, so overwriting them breaks the next national run and silently gives one state's prices to
> every state. Per-state tables let studies coexist. The old notebook is still useful for its LBNL
> price-decline calculation (cells 1–5), which is informational either way.

**File:** dgen\_os/python/upload\_state\_price\_tables.py

## **Step 5 — Set your target state and upload the state list to GCS**

**YOU DO:** Edit state\_input\_csvs/mid\_states\_test.csv so it contains exactly one line:

XX,Full State Name

No header. The abbreviation must match what you used in your price CSV filenames. The full name must be spelled exactly as dGen expects (e.g., Maryland not maryland).

Then upload it to GCS with the exact same filename (the YAML fetches it by this name). Run the below commands after navigating to your repo

`cd /Users/wael/Documents/repos/dgen`

gsutil cp state\_input\_csvs/mid\_states\_test.csv gs://dgen-assets/mid\_states\_test.csv

gsutil cp state\_input\_csvs/single\_state\_large.csv gs://dgen-assets/single\_state\_large.csv

**AUTOMATIC:** Nothing yet.

**Important:** If you ever want to rename mid\_states\_test.csv to something else, you must also update the fetch line in batch\_job\_yamls/dgen-batch-job-mid-states.yaml (line 44\) to match.

**File:** state\_input\_csvs/mid\_states\_test.csv

## **Step 6 — Check the YAML and submit\_one.sh for your state**

**YOU DO:** Open submit\_one.sh and check two things:

1. JOB\_NAME="dgen-md-${JOB\_TS}" — change md to your state abbreviation so you can identify the job in GCP Console

2. \--config="batch\_job\_yamls/dgen-batch-job-mid-states.yaml" — confirm this is the right size class for your state (MD is mid; CA/TX/NY/FL are large; check the README table for others)

Open the YAML and verify:

taskCount: "1"  
parallelism: "1"

The mid-states YAML already has these set to 1\. If you switch to a different YAML (e.g., large-states), confirm these are also set to 1 before submitting.

**AUTOMATIC:** Nothing yet.

**Files:** submit\_one.sh, batch\_job\_yamls/dgen-batch-job-mid-states.yaml

## **Step 7 — Authenticate to GCP (one-time, or when credentials expire)**

**YOU DO:** In a new Mac terminal:

gcloud auth login  
gcloud config set project dgen-466702  
gcloud auth configure-docker us-east1-docker.pkg.dev

**AUTOMATIC:** GCP stores credentials locally so Docker and gcloud can authenticate without prompting again.

## **Step 8 — Build and submit the job**

**YOU DO:** In a Mac terminal from the repo root:

```shell
cd /Users/wael/Documents/repos/dgen
bash build_and_submit.sh
```

**Note:** If you have not changed any Python model code since the last run, the Docker rebuild is optional — the existing image on Google’s registry would still work. build\_and\_submit.sh always rebuilds anyway, which is safe but takes 5–15 minutes.

**AUTOMATIC, in sequence:**

1. Docker builds an image containing all model code and the dg3n conda environment, pushes it to Google’s container registry

2. submit\_one.sh submits the Batch job to GCP

3. GCP spins up a VM and pulls the Docker image

4. The VM downloads the scenario Excel templates (baseline.xlsm, policy.xlsm) and your state CSV from GCS

5. prepare\_all\_scenarios.py opens the Excel templates and injects your state’s full name and end year (2040) into the right cells, saving them as baseline\_{STATE}\_2040.xlsm and policy\_{STATE}\_2040.xlsm

6. dgen\_model.py runs — it reads prices from Cloud SQL, runs the full model for baseline then policy, and writes results back to Cloud SQL as new schemas named diffusion\_results\_baseline\_{state}\_{timestamp} and diffusion\_results\_policy\_{state}\_{timestamp}

## **Step 9 — Monitor the job**

**YOU DO:** GCP Console → Batch → Jobs → find your job (named dgen-{state}-{timestamp}) → click Logs.

Confirm:

* You see \[Task 0\] Running for state=Your State (abbr=XX) — confirms the right state was picked from the CSV

* Agent count looks plausible for your state (too low or too high usually means the wrong state CSV was uploaded to GCS)

A single mid-state run takes 1–3 hours.

## **Step 10 — (Optional) Clean up old result schemas**

If you have results from previous runs sitting in Cloud SQL and want a clean export, drop them before exporting. In GCP Console, click Activate Cloud Shell (top right), then:

gcloud sql connect dgen-db \--user=postgres \--database=dgendb  
\# password: postgres

If the proxy from a previous connection is still running, run

```shell
pkill -f cloud-sql-proxy
```

To see all the schemas, including old ones, run the below.   
Then if you want to only pull those runs, add them to the `schemas_include` variable in dgen\_os/python/Notebooks/loading\_model\_results.ipynb, like so:

schemas\_include=\["diffusion\_results\_baseline\_tx\_2040\_20260429\_194512300061", "diffusion\_results\_policy\_tx\_2040\_20260429\_200138122066"\]

diffusion\_results\_baseline\_nj\_2040\_20260605\_122008109917  
diffusion\_results\_policy\_nj\_2040\_20260605\_121006322247

SELECT schema\_name FROM information\_schema.schemata WHERE schema\_name LIKE 'diffusion\_results%' ORDER BY schema\_name;

Then run:

DO $$  
DECLARE r record;  
BEGIN  
  PERFORM set\_config('lock\_timeout','5s', true);  
  FOR r IN SELECT nspname FROM pg\_namespace WHERE nspname LIKE 'diffusion\_results%' LOOP  
    BEGIN  
      EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE;', r.nspname);  
    EXCEPTION WHEN OTHERS THEN  
      RAISE NOTICE 'Skipped % due to %', r.nspname, SQLERRM;  
    END;  
  END LOOP;  
END$$;

## **Step 11 — Export results**

**YOU DO:** 

1. Confirm Cloud SQL Proxy from Step 3 is still running

2. Open dgen\_os/python/Notebooks/loading\_model\_results.ipynb

3. Update these two variables before running:

OUT\_DIR \= "/Users/yourname/wherever"  \# your local output folder  
RUN\_ID  \= "run\_md\_march2026"          \# descriptive label for this run

4. Run all cells

**AUTOMATIC:** Notebook connects to Cloud SQL via the proxy, finds all diffusion\_results\_\* schemas from your run (one baseline, one policy), and exports them as CSVs to your local folder.

**File:** dgen\_os/python/Notebooks/loading\_model\_results.ipynb

## **Step 12 — Run analysis**

**YOU DO:** Open dgen\_os/python/Notebooks/analysis\_of\_model\_results.ipynb, point it at your exported folder, and run it.

**AUTOMATIC:** Computes adoption, capacity, net load, and savings metrics across baseline and policy scenarios and generates comparison charts.

**File:** dgen\_os/python/Notebooks/analysis\_of\_model\_results.ipynb

## **Step 13 — Stop Cloud SQL**

**YOU DO:** GCP Console → Cloud SQL → Stop. Do not delete. It will restart in minutes next time.

# **Quick reference: the three things you change per state run**

| What | Where | Notes |
| :---- | :---- | :---- |
| Solar price trajectory | system\_capex\_per\_kw\_res col, years 2026–2040, in both baseline and policy CSVs | Units are $/kW |
| State abbreviation at upload | `--states XX` passed to upload\_state\_price\_tables.py | Must match the CSV filename abbreviation; sets the table names too |
| Price table names in YAML | PV\_PRICE\_TABLE\_\* / PV\_PLUS\_BATT\_TABLE\_\* in your state's YAML | Must match the tables the upload created, e.g. pv\_plus\_batt\_pa\_baseline. Unset = the shared national tables |
| State in list CSV | state\_input\_csvs/mid\_states\_test.csv → one line → upload to GCS | Filename must match what YAML fetches |
| Job name in submit script | JOB\_NAME="dgen-{state}-..." in submit\_one.sh | Cosmetic but important for tracking |
| YAML \+ machine size | \--config= line in submit\_one.sh | Only if switching state size class |

