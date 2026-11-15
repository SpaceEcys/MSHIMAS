from __future__ import annotations
"""MSHIMAS: offline synthetic-data research prototype; Python 3.10+.
Install once: python -m pip install -r requirements.txt
Run: python MSHIMAS.py (or VS Code Run Python File).
Standalone web application; the UI and all local modules are embedded below.
Model data and crew sessions share MSHIMAS_Database.xlsx.
Illustrative screening rules below are inherited research heuristics, not
NASA thresholds or validated clinical guidance. No confirmed diagnoses.
All model preprocessing and statistical references use training rows only.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, cast
try:
    import numpy as np
    import pandas as pd
    from pandas.util import hash_pandas_object
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as exc:
    raise SystemExit("Missing dependency: " + str(exc) + "\nInstall: python -m pip install numpy pandas scikit-learn openpyxl")

METRICS = [
    "Hemoglobin (g/dl)", "Hematocrit (%)", "Leukocytes (k/uL)",
    "Calcium (mg/dL)", "Vitamin D (ng/mL)", "alkaline phosphatase (U/L)",
    "Calcium excretion (mg/24hr)", "metabolites(ng/mL)", "Heart Rate (Bpm)",
    "HRV (ms)", "Oxygen saturation (%)", "SOS bone density (m/s)",
    "Muscle Mass (kg)", "Radiation per day(millirads)",
    "Cabin Pressure/atm (kPa)", "Space Suit Pressure (kPa)",
    "EVA profile | Role",
    "Retinal & Optic Nerve Thickening(um) minimum rim width;",
    "Sodium (mmol/L)", "Sex", "Age", "Aerobic Capacity ml⁄min/kg",
    "Work W", "Height",
]

CAT = ["EVA profile | Role", "Sex"]

NUM = [c for c in METRICS if c not in CAT]

MISSING = {"undefined", "unknown", "nan", "none", "null", "", "n/a", "na", "-"}

LEVELS = {"NORMAL": 0, "MONITOR": 1, "REVIEW": 2, "URGENT": 3}

def _as_float(value: Any, spo2=False):
    if isinstance(value, (list, dict, tuple)):
        return np.nan
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in MISSING:
            return np.nan
        if s.endswith("%"):
            if not spo2:
                return np.nan
            s = s[:-1].strip()
        s = s.replace(",", ".")
        try:
            value = float(s)
        except ValueError:
            return np.nan
    try:
        number = float(value)
        if not np.isfinite(number) or abs(number) > 1e12:
            return np.nan
        return number
    except (ValueError, TypeError, OverflowError):
        return np.nan

def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Strict schema, robust missing-value interpretation, safe numeric parsing."""
    missing = [c for c in METRICS if c not in frame.columns]
    extra = [c for c in frame.columns if c not in METRICS]
    if missing:
        raise ValueError(f"Missing required input metrics: {missing}")
    if extra:
        raise ValueError(f"Unexpected model input columns: {extra}. Use a separate metadata table.")
    out = pd.DataFrame(frame.loc[:, METRICS]).copy()
    for col in NUM:
        out[col] = cast(pd.Series, out[col]).map(lambda value: _as_float(value, spo2=(col == "Oxygen saturation (%)")))
    for col in CAT:
        out[col] = cast(pd.Series, out[col]).map(lambda v: "Unknown" if pd.isna(v) or str(v).strip().lower() in MISSING else str(v).strip())
        if col == "Sex":
            out[col] = cast(pd.Series, out[col]).map(lambda v: {"f": "F", "female": "F", "m": "M", "male": "M"}.get(str(v).lower(), str(v)))
        else:
            out[col] = cast(pd.Series, out[col]).map(lambda v: {"iva": "IVA", "eva": "EVA", "rover": "Rover", "base": "Base"}.get(str(v).lower(), str(v)))
    return out

def build_pipeline(contamination: float = 0.20, random_state: int = 26) -> Pipeline:
    forest_options: dict[str, Any] = dict(n_estimators=250, max_samples=256,
                                        contamination=contamination, random_state=random_state, n_jobs=1)
    return Pipeline(steps=[
        ("prepare", ColumnTransformer(transformers=[
            ("numbers", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=False)),
                ("scale", RobustScaler()),
            ]), NUM),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ], remainder="drop", sparse_threshold=0.0)),
        ("anomaly", IsolationForest(**forest_options)),
    ])

def train_unlabeled(cases: pd.DataFrame, *, contamination: float = 0.20) -> dict:
    X = normalize(cases)
    if len(X) < 50:
        raise ValueError("At least 50 unlabeled examples are required for this prototype.")
    columns = {c: cast(pd.Series, X[c]) for c in NUM}
    usable = [c for c in NUM if bool(columns[c].notna().any())]
    if not usable:
        raise ValueError("Training requires observed numeric measurements.")
    pipeline = build_pipeline(contamination=contamination)
    pipeline.named_steps["prepare"].transformers[0] = ("numbers", pipeline.named_steps["prepare"].transformers[0][1], usable)
    pipeline.fit(X)  # NO diagnostic/label column is used here
    references = {column: {
        "median": float(columns[column].median()) if bool(columns[column].notna().any()) else None,
        "q25": float(columns[column].quantile(.25)) if bool(columns[column].notna().any()) else None,
        "q75": float(columns[column].quantile(.75)) if bool(columns[column].notna().any()) else None,
    } for column in NUM}
    return {"pipeline": pipeline, "reference": references,
            "schema": METRICS[:], "model": "IsolationForest unsupervised",
            "n_train": len(X), "contamination": contamination,
            "clinical_certified": False}

def _flag(flags: list, metric: str, observed: Any, level: str, issue: str, action: str):
    flags.append({"metric": metric, "value": float(observed), "level": level,
                  "finding": issue, "action": action})

def screening_rules(values: dict) -> list[dict]:
    """Research-only conservative screening, NOT astronaut medical flight rules."""
    f = []

    def check(metric, moderate, urgent, concern, action):
        v = values.get(metric)
        if v is None or pd.isna(v):
            return
        if urgent(v):
            _flag(f, metric, v, "URGENT", concern, action)
        elif moderate(v):
            _flag(f, metric, v, "REVIEW", concern, action)

    oxygen_action = "Pause exertion, secure yourself as required, and check the onboard sensor setup. If the reading remains low or you feel unwell, follow the mission medical procedure and contact the crew medical officer."
    sodium_action = "Compare the value with mission lab ranges and your flight baseline. Do not adjust salt or fluids yourself; follow the assigned hydration plan and consult the crew medical officer."
    calcium_action = "Review the confirmed value against mission lab ranges and your flight baseline with crew medical staff. Follow the assigned bone-health countermeasure plan; do not change supplements without flight-medical direction."
    hemoglobin_action = "Review the onboard result with the crew medical officer against your flight baseline and fluid-shift context before changing iron intake or supplements; report any symptoms."
    leukocyte_action = "Have crew medical staff review the result and trend; a white-cell count alone cannot identify infection or an immune cause. Follow the mission medical plan if symptoms are present."
    heart_rate_action = "Stop exertion, secure yourself as required, rest for 5 minutes, and check the onboard monitor. Contact the crew medical officer if the reading stays outside your baseline or you have symptoms."
    vitamin_d_action = "Review the result with the flight surgeon against the onboard nutrition and bone-health plan. Use only prescribed supplements and do not start or increase a dose based on this screening result."
    calcium_excretion_action = "Check the timed urine collection start/end times and completeness using the onboard lab procedure; review the trend with crew medical staff alongside calcium, vitamin D, and your flight baseline."
    alkaline_phosphatase_action = "Review the result and trend with crew medical staff alongside calcium, vitamin D, your flight baseline, and the onboard lab method; do not change supplements or countermeasures based on this value alone."
    check("Oxygen saturation (%)", lambda x: x < 94, lambda x: x < 90,
          "Low measured oxygen saturation; check signal quality and clinical context.", oxygen_action)
    check("Sodium (mmol/L)", lambda x: x < 135 or x > 145, lambda x: x < 130 or x > 150,
          "Sodium is outside an illustrative reference interval; interpret it using mission laboratory ranges and crew medical context.", sodium_action)
    check("Calcium (mg/dL)", lambda x: x < 8.5 or x > 10.5, lambda x: x < 7.5 or x > 12,
          "Serum calcium is outside an illustrative reference interval; confirm it using mission laboratory ranges and review it in the crew member's bone-health context.", calcium_action)
    check("Hemoglobin (g/dl)", lambda x: x < 12 or x > 18, lambda x: x < 8 or x > 20,
          "Hemoglobin merits medical review; interpret against sex and individual baseline.", hemoglobin_action)
    check("Leukocytes (k/uL)", lambda x: x < 4 or x > 11, lambda x: x < 2 or x > 20,
          "White blood cell count requires verification; this alone does not establish infection.", leukocyte_action)
    check("Heart Rate (Bpm)", lambda x: x < 50 or x > 110, lambda x: x < 40 or x > 140,
          "Heart rate is outside illustrative resting values; activity level and context matter.", heart_rate_action)
    check("Vitamin D (ng/mL)", lambda x: x < 20 or x > 80, lambda x: x < 12 or x > 150,
          "Vitamin D requires review with the onboard nutrition and bone-health plan; this result is not a diagnosis or a supplement dose.", vitamin_d_action)
    check("Calcium excretion (mg/24hr)", lambda x: x > 300 or x < 40, lambda x: x > 500,
          "24-hour calcium excretion is outside illustrative values; verify the timed collection and review calcium balance and bone-health trends in the microgravity mission context.", calcium_excretion_action)
    check("alkaline phosphatase (U/L)", lambda x: x < 35 or x > 135, lambda x: x > 250,
          "Alkaline phosphatase merits review alongside calcium, vitamin D, the flight baseline, and the onboard laboratory method; this value alone does not determine a supplement or countermeasure change.", alkaline_phosphatase_action)
    # Environmental measurements require vehicle-specific protocols, not a clinical range.
    c = values.get("Cabin Pressure/atm (kPa)")
    if c is not None and pd.notna(c):
        d = abs(c - 101.34) / 101.34
        if d > .05:
            _flag(f, "Cabin Pressure/atm (kPa)", c, "URGENT" if d > .10 else "REVIEW",
                  "Cabin pressure differs from the example 101.34 kPa setpoint; vehicle configuration may differ.",
                  "Check the cabin instrumentation, verify the local pressure reference, and follow the vehicle's approved environmental emergency procedures.")
    role = str(values.get("EVA profile | Role", "")).strip().upper()
    suit = values.get("Space Suit Pressure (kPa)")
    if role == "EVA" and suit is not None and pd.notna(suit):
        deviation = abs(suit - 29.5) / 29.5
        if deviation > .05:
            _flag(f, "Space Suit Pressure (kPa)", suit,
                  "URGENT" if deviation > .10 else "REVIEW",
                  "Suit pressure deviates from the example 29.5 kPa EMU reference; this does not apply to every suit.",
                  "Verify the suit configuration, check the actual pressure sensor reading, and follow the approved EVA procedure for the specific suit system before any action.")
    rad = values.get("Radiation per day(millirads)")
    if rad is not None and pd.notna(rad) and rad > 350:
        _flag(f, "Radiation per day(millirads)", rad, "REVIEW",
              "Dose rate is higher than the example 175 mrad/day reference; this is not an official exposure threshold.",
              "Verify the dosimeter reading, review the mission exposure log, and follow spacecraft radiological safety, shelter, and flight-surgeon procedures.")
    # Explicit validation of impossible measurements for data quality.
    for col in ["Oxygen saturation (%)", "Heart Rate (Bpm)", "Sodium (mmol/L)", "Age", "Height"]:
        v = values.get(col)
        if v is None or pd.isna(v):
            continue
        impossible = (v < 0 or (col == "Oxygen saturation (%)" and v > 100)
                      or (col == "Age" and v > 120) or (col == "Height" and v > 250))
        if impossible:
            _flag(f, col, v, "URGENT", "Implausible input; likely instrument, unit, or transcription error.",
                  "Verify the onboard sensor or lab source, units, and data entry; do not rely on this result until it has been checked and accurately re-entered.")
    return sorted(f, key=lambda x: LEVELS[x["level"]], reverse=True)

def personalized_trends(history: pd.DataFrame | None, current: dict) -> list[dict]:
    """Compare recent reading with first 3-5 personal pre-change observations.

    Sequence is determined by row order (time/astronaut ID are metadata, never features).
    """
    if history is None or len(history) < 3:
        return []
    X = normalize(history)
    baseline = X.iloc[:min(5, len(X))]
    config = {
        "Hemoglobin (g/dl)": (.15, "decrease"),
        "Hematocrit (%)": (.15, "decrease"),
        "HRV (ms)": (.30, "decrease"),
        "SOS bone density (m/s)": (.08, "decrease"),
        "Muscle Mass (kg)": (.08, "decrease"),
        "Retinal & Optic Nerve Thickening(um) minimum rim width;": (.12, "both"),
        "Aerobic Capacity ml⁄min/kg": (.15, "decrease"),
        "Work W": (.20, "decrease"),
        "Radiation per day(millirads)": (1.0, "increase"),
    }
    findings = []
    for col, (cutoff, direction) in config.items():
        base = baseline[col].median()
        observed = current.get(col)
        if pd.isna(base) or observed is None or pd.isna(observed) or abs(base) < 1e-9:
            continue
        fraction = (observed - base) / abs(base)
        triggered = ((direction == "decrease" and fraction <= -cutoff) or
                     (direction == "increase" and fraction >= cutoff) or
                     (direction == "both" and abs(fraction) >= cutoff))
        if triggered:
            findings.append({
                "metric": col, "value": float(observed), "baseline": round(float(base), 2),
                "change_pct": round(100 * fraction, 1), "level": "REVIEW",
                "finding": "Important change relative to this person's early measurements; review trend and instrument consistency.",
                "action": "Check the sensor setup and mission measurement protocol; discuss the trend with flight medicine before changing the countermeasure plan.",
            })
    return findings

def robust_population_outliers(values: dict, reference: dict, multiplier: float = 1.5) -> list[dict]:
    """Unsupervised Tukey-fence novelty detector; 1.5*IQR beyond Q1/Q3.

    It complements Isolation Forest for isolated, strong univariate changes.
    This is statistical outlier detection, NOT a clinical normal range.
    """
    unusual = []
    for col in NUM:
        x = values.get(col)
        r = reference.get(col, {})
        if x is None or pd.isna(x) or r.get("q25") is None or r.get("q75") is None:
            continue
        q1, q3 = r["q25"], r["q75"]
        iqr = q3 - q1
        if iqr <= 1e-7:
            continue
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        if x < lo or x > hi:
            unusual.append({"metric": col, "value": float(x),
                            "population_reference_lower": round(float(lo), 3),
                            "population_reference_upper": round(float(hi), 3)})
    return unusual

def _driver_scores(values: dict, reference: dict, n=5) -> list[dict]:
    scored = []
    for col in NUM:
        x = values.get(col)
        r = reference.get(col, {})
        if x is None or pd.isna(x) or r.get("median") is None:
            continue
        width = (r.get("q75") or 0) - (r.get("q25") or 0)
        if width <= 1e-7:  # constant/undefined feature is not clinically interpretable
            continue
        score = abs(x - r["median"]) / width
        scored.append({"metric": col, "value": round(float(x), 3),
                       "reference_median": round(r["median"], 3),
                       "distance_IQR": round(float(score), 2)})
    return sorted(scored, key=lambda d: d["distance_IQR"], reverse=True)[:n]

def _patterns(values: dict, flagged: list) -> list[str]:
    def val(col):
        v = values.get(col)
        return float(v) if v is not None and pd.notna(v) else np.nan
    patterns = []
    if val("Hemoglobin (g/dl)") < 12 and val("Hematocrit (%)") < 36:
        patterns.append("Possible anemia-pattern change (not a diagnosis); confirm the onboard sample and compare with the crew member's flight baseline and fluid-shift context.")
    if val("Leukocytes (k/uL)") < 4 or val("Leukocytes (k/uL)") > 11:
        patterns.append("White-cell alteration; review the onboard result and trend with crew medical staff. Infection or immune cause cannot be inferred from this count alone.")
    if val("Oxygen saturation (%)") < 94 and val("Heart Rate (Bpm)") > 105:
        patterns.append("Combined cardiorespiratory warning: follow onboard medical procedures and review sensor quality, recent exercise, and mission context.")
    if val("Vitamin D (ng/mL)") < 20 and val("Calcium excretion (mg/24hr)") > 300:
        patterns.append("Possible microgravity bone-health pattern (not a diagnosis); review calcium excretion, vitamin D, onboard nutrition, and assigned countermeasures with crew medical staff.")
    if any(x["metric"] == "Radiation per day(millirads)" for x in flagged):
        patterns.append("Radiation-monitoring event; no radiation disease can be diagnosed from this daily reading.")
    if any(x["metric"] in ("Space Suit Pressure (kPa)", "Cabin Pressure/atm (kPa)") for x in flagged):
        patterns.append("Life-support environment deviation; evaluate using mission-specific operating limits.")
    return patterns

def legacy_assess(bundle: dict, record: dict | pd.Series,
           history: pd.DataFrame | None = None) -> dict:
    raw = pd.DataFrame([dict(record)])
    X = normalize(raw)
    values = X.iloc[0].to_dict()
    available = sum(pd.notna(values[c]) for c in NUM)
    prediction = int(bundle["pipeline"].predict(X)[0]) if available >= 3 else 1
    decision = float(bundle["pipeline"].decision_function(X)[0]) if available >= 3 else None
    flags = screening_rules(values)
    quality = quality_issues(raw.iloc[0].to_dict(), values)
    encoder = bundle["pipeline"].named_steps["prepare"].named_transformers_["categories"]
    for c, known in zip(CAT, encoder.categories_):
        if values[c] not in known and values[c] != "Unknown":
            quality.append(f"{c}: unseen category {values[c]!r}; model has no learned category reference.")
    trends = personalized_trends(history, values)
    for issue in trends:
        if not any(f["metric"] == issue["metric"] for f in flags):
            flags.append(issue)
    outliers = robust_population_outliers(values, bundle["reference"])
    # Two independently unsupervised novelty detectors; no case labels used.
    ml_anomaly = prediction == -1 or len(outliers) > 0
    missing_count = sum(pd.isna(values[c]) for c in NUM)
    # A statistical outlier is a prompt to investigate, not a medical diagnosis.
    level = max((LEVELS[x["level"]] for x in flags), default=0)
    if ml_anomaly and level < 1:
        level = 1
    if (missing_count >= 7 or quality) and level < 2:
        level = 2
    severity = next(label for label, i in LEVELS.items() if i == level)
    return {
        "status": severity,
        "reference_metrology": [process_observation(c, {"value": float(values[c])})
                                for c in NUM if c in BY_KEY and pd.notna(values[c])],
        "quality_issues": quality,
        "personal_trends": trends,
        "ml_anomaly": ml_anomaly,
        "isolation_forest_anomaly": prediction == -1,
        "iqr_outliers": outliers,
        "anomaly_margin": round(decision, 5) if decision is not None else None,
        "ml_withheld": available < 3,
        "missing_metrics": [c for c in METRICS if pd.isna(values[c]) or values[c] == "Unknown"],
        "nonmedical_score_explanation": "decision_function < 0 means statistical anomaly; not a disease probability",
        "flags": sorted(flags, key=lambda f: LEVELS[f["level"]], reverse=True),
        "statistical_drivers": _driver_scores(values, bundle["reference"]),
        "possible_patterns": _patterns(values, flags),
        "numeric_measurements_available": len(NUM) - missing_count,
        "numeric_measurements_total": len(NUM),
        "notice": "Educational research prototype. Neither the ML output nor these heuristics diagnose illness, prescribe treatment, or authorize EVA/vehicle actions. Mission-approved procedures and flight surgeons take precedence.",
    }

def evaluate_unseen(model: dict, cases: pd.DataFrame, scenario_labels: list[str]) -> dict:
    """External synthetic holdout evaluation. Labels are not model features."""
    X = normalize(cases)
    forest = model["pipeline"].predict(X) == -1
    iqr = np.array([bool(robust_population_outliers(row.to_dict(), model["reference"]))
                    for _, row in X.iterrows()], dtype=bool)
    predictions = forest | iqr
    actual = np.array([s != "Routine baseline" for s in scenario_labels], dtype=bool)
    tp = int(np.sum(predictions & actual))
    fp = int(np.sum(predictions & ~actual))
    fn = int(np.sum(~predictions & actual))
    tn = int(np.sum(~predictions & ~actual))
    forest_tp = int(np.sum(forest & actual))
    forest_fp = int(np.sum(forest & ~actual))
    return {"holdout": len(X), "isolation_forest_only_precision": round(forest_tp / max(forest_tp + forest_fp, 1), 3),
            "isolation_forest_only_recall": round(forest_tp / max(int(np.sum(actual)), 1), 3),
            "detector": "IsolationForest OR unsupervised 1.5 IQR population outlier",
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3),
            "warning": "Only synthetic scenarios were evaluated: NOT clinical validity or flight qualification."}

def _database_path():
    import os
    import shutil
    bundled = Path(__file__).resolve().with_name("MSHIMAS_Database.xlsx")
    target = Path(os.environ.get("MSHIMAS_DB_PATH", str(bundled)))
    if target != bundled and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, target)
    return target

DATABASE = _database_path()
NOTICE = "This is a research prototype, not a validated medical diagnostic system. Synthetic interpretations are not confirmed diagnoses."


def quality_issues(raw, values):
    issues = []
    for c in NUM:
        v = raw.get(c)
        absent = pd.isna(v) or str(v).strip().lower() in MISSING
        if not absent and pd.isna(values[c]):
            issues.append(f"{c}: invalid/nonfinite or excessive input {v!r}; unavailable for ML.")
        if pd.notna(values[c]) and (values[c] < 0 or
                (c in ("Oxygen saturation (%)", "Hematocrit (%)") and values[c] > 100) or
                (c == "Age" and values[c] > 120) or (c == "Height" and values[c] > 250)):
            issues.append(f"{c}: implausible value {values[c]}; check units and entry.")
    return issues


def load_database(path=DATABASE):
    """Only this workbook is read. Labels are kept outside the feature table."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    cases = sheets["Cases_24_metrics"]
    if cases.columns.tolist() != METRICS or len(METRICS) != 24:
        raise ValueError("Cases_24_metrics must contain exactly the original 24 metric names in order.")
    meta = sheets["Case_Metadata"]
    if len(meta) != len(cases) or meta.Case_ID.isna().any() or meta.Case_ID.duplicated().any():
        raise ValueError("Case metadata must provide one unique ID per measurement row.")
    if meta.Data_Row_in_Cases_24_metrics.tolist() != list(range(2, len(cases)+2)):
        raise ValueError("Case metadata row links are not aligned.")
    for name in ("Case_Diagnoses", "Case_Classifications"):
        labels = sheets[name]
        if labels.Case_ID.duplicated().any() or not set(labels.Case_ID).issubset(set(meta.Case_ID)):
            raise ValueError(f"Invalid case links in {name}.")
    sheets["X"] = normalize(cases)
    return sheets


def train_and_evaluate(db):
    # Eligibility is provenance-based; no scenario/diagnosis is inspected here.
    eligible = db["Case_Metadata"].index[db["Case_Metadata"].Training_Eligible.eq(True)].to_numpy()
    X = db["X"].iloc[eligible]
    # Keep identical feature rows in the same partition, preventing duplicate leakage.
    hash_rows = cast(Callable[..., pd.Series], hash_pandas_object)
    groups = hash_rows(X, index=False).to_numpy()
    if len(np.unique(groups)) < 60:
        raise ValueError("Insufficient independent synthetic cases for train/holdout split.")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=2026).split(X, groups=groups))
    bundle = train_unlabeled(X.iloc[train], contamination=.22)
    bundle["train_indices"] = eligible[train]
    bundle["test_indices"] = eligible[test]
    # Retrieve simulated labels only AFTER fitting. Never refit on the holdout.
    ids = db["Case_Metadata"].iloc[eligible[test]].Case_ID
    classification = db["Case_Classifications"].set_index("Case_ID")
    scenarios = classification.reindex(ids)["Scenario"]
    if scenarios.isna().any():
        raise ValueError("Missing synthetic validation classifications.")
    report = evaluate_unseen(bundle, X.iloc[test], scenarios.tolist())
    print("\nTRAINING AND SYNTHETIC HOLDOUT EVALUATION")
    print(f"Training rows: {len(train)} | Held out: {len(test)} | Original input metrics: 24")
    print("Isolation Forest and IQR use training-only imputers, encoders, scaler and quartiles.")
    print("Split is label-blind, grouped by identical measurements; no independent-person clinical validation.")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("All models remain in memory. No model or result files are saved.")
    return bundle


def previous_measurements(db, idx):
    meta = db["Case_Metadata"]
    row = meta.iloc[idx]
    person, stamp = row.Astronaut_ID, row.Timestamp_UTC
    if pd.isna(person) or pd.isna(stamp):
        return None
    times = pd.to_datetime(meta.Timestamp_UTC, utc=True, errors="coerce")
    t = pd.to_datetime(stamp, utc=True, errors="coerce")
    prior = meta.index[meta.Astronaut_ID.eq(person) & times.lt(t)]
    prior = sorted(prior, key=lambda i: times.iloc[i])
    # A trend needs >=3 distinct earlier timestamps and explicit baseline provenance.
    baseline = [i for i in prior if meta.iloc[i].Baseline_Eligible == True]
    if len(baseline) < 3 or times.iloc[baseline].nunique() < 3:
        return None
    # Only documented early reference observations, never current/future readings.
    return db["X"].iloc[baseline[:5]]


def print_assessment(result, label="", partition="", history_available=False):
    print("\nMSHIMAS | Legacy synthetic feature assessment")
    print("Monitoring System for Highly Important Metrics for Astronaut Survival")
    for measurement in result.get("reference_metrology", []):
        print(f"- {measurement['label']}: {measurement['value']:.4g} +/- "
              f"{measurement['expanded_uncertainty']:.4g} {measurement['unit']} "
              f"(k={measurement['coverage_k']}; {measurement['gate']}; automatic reference budget)")
    print("\nASTRONAUT HEALTH ASSESSMENT", label)
    status = {"REVIEW": "MEDICAL REVIEW REQUIRED", "URGENT": "MEDICAL REVIEW REQUIRED - PRIORITY SAFETY ALERT"}.get(result["status"], result["status"])
    print("\nOverall Status:", status)
    print("Machine Learning:", "WITHHELD - insufficient observations" if result["ml_withheld"] else "Anomaly Detected" if result["ml_anomaly"] else "No anomaly detected")
    print("Isolation Forest anomaly margin:", result["anomaly_margin"], "(negative = anomaly; not disease probability)")
    print("Isolation Forest:", result["isolation_forest_anomaly"], "| Complementary IQR:", bool(result["iqr_outliers"]))
    print("Numeric observations:", result["numeric_measurements_available"], "/", result["numeric_measurements_total"])
    if partition:
        print("Dataset partition:", partition)
    print("\nAbnormal Indicators:")
    for f in result["flags"]:
        print(f"- {f['metric']}: {f['value']} [{f['level']}] {f['finding']}")
    for f in result["iqr_outliers"]:
        print(f"- {f['metric']}: {f['value']} [statistical] training IQR fence {f['population_reference_lower']} to {f['population_reference_upper']}")
    if not result["flags"] and not result["iqr_outliers"]:
        print("- None under the defined demonstration detectors; illness is not ruled out.")
    print("\nMost unusual measurements (training median/IQR distances; not causal model attribution):")
    for f in result["statistical_drivers"]:
        print(f"- {f['metric']}: {f['value']}; median {f['reference_median']}; distance {f['distance_IQR']} IQR")
    print("\nPossible Health Concerns (simulated rule interpretations):")
    for p in result["possible_patterns"] or ["No specific pattern identified; this cannot establish or exclude disease."]:
        print("-", p)
    if any(f["metric"].startswith("Retinal") for f in result["iqr_outliers"]):
        print("- Ocular/fluid-shift related change may merit same-eye OCT and expert review; no SANS diagnosis.")
    if any(f["metric"] in ("SOS bone density (m/s)", "Muscle Mass (kg)", "Aerobic Capacity ml⁄min/kg") for f in result["iqr_outliers"]):
        print("- Musculoskeletal/fitness variation may merit review in altered gravity; protocol and baseline are required.")
    print("\nPersonal baseline:", "Earlier same-astronaut reference observations available." if history_available else "Unavailable: no verified earlier reference series; population comparison only.")
    for f in result["personal_trends"]:
        print(f"- {f['metric']}: {f['value']}, early median {f['baseline']}, change {f['change_pct']}%; possible deterioration signal.")
    if history_available and not result["personal_trends"]:
        print("No configured personal change threshold exceeded; stability is not clinically established.")
    print("\nData Quality:")
    print("Unavailable metrics:", ", ".join(result["missing_metrics"]) or "none")
    for issue in result["quality_issues"]:
        print("-", issue)
    print("Missing values are imputed with TRAINING medians for statistics only; imputed values are never called measurements.")
    print("\nRecommended Actions:")
    actions = list(dict.fromkeys(f["action"] for f in result["flags"]))
    for action in actions or ["Review unusual measurements, sensor quality, units, and vehicle-specific conditions with the crew medical team before changing care or countermeasures."]:
        print("-", action)
    print("Safety alerts are independent of ML. Terrestrial screening and cabin/suit setpoints are illustrative, not astronaut operating limits.")
    print("\nImportant:\n" + NOTICE)


def select_case(db):
    query = input("Case ID, Excel data row (2+), or scenario text to list matches [blank cancels]: ").strip()
    if not query:
        return None
    meta = db["Case_Metadata"]
    if query.isdigit():
        idx = int(query)-2
        if 0 <= idx < len(meta):
            return idx
    found = meta.index[meta.Case_ID.str.casefold().eq(query.casefold())]
    if len(found):
        return int(found[0])
    labels = db["Case_Classifications"]
    matches = labels[labels.Scenario.fillna("").str.contains(query, case=False, regex=False)]
    for _, row in matches.head(20).iterrows():
        print(row.Case_ID, "|", row.Scenario)
    print("Enter an exact Case ID using option 2. Shown labels are synthetic context only.")
    return None


def manual_entry(db):
    print("Enter values in the displayed units; blank/unknown = unavailable. :cancel aborts.")
    print("SpO2 uses percent units (97 or 97%); bare 0.97 means 0.97%, not 97%.")
    values = {}
    for c in METRICS:
        while True:
            v = input(c + ": ").strip()
            if v.lower() == ":cancel":
                return None
            if c in CAT or v.lower() in MISSING:
                values[c] = v
                break
            n = _as_float(v, spo2=c == "Oxygen saturation (%)")
            if pd.isna(n):
                print("Invalid or excessive number; re-enter or leave blank. No zero will be substituted.")
                continue
            values[c] = v
            break
    person = input("Astronaut ID for optional comparison [blank = no personal baseline]: ").strip()
    history = None
    if person:
        meta = db["Case_Metadata"]
        times = pd.to_datetime(meta.Timestamp_UTC, utc=True, errors="coerce")
        rows = meta.index[meta.Astronaut_ID.eq(person) & meta.Baseline_Eligible.eq(True) & times.lt(pd.Timestamp.now(tz="UTC"))]
        rows = sorted(rows, key=lambda i: times.iloc[i])
        if len(rows) >= 3 and times.iloc[rows].nunique() >= 3:
            history = db["X"].iloc[rows[:5]]
    return values, history


def show_trends(db, bundle):
    meta = db["Case_Metadata"]
    people = meta.Astronaut_ID.dropna().unique().tolist()
    print("Available longitudinal series:", ", ".join(people) or "none")
    person = input("Astronaut ID [blank cancels]: ").strip()
    if not person:
        return
    rows = meta.index[meta.Astronaut_ID.eq(person)]
    times = pd.to_datetime(meta.Timestamp_UTC, utc=True, errors="coerce")
    rows = [i for i in rows if pd.notna(times.iloc[i])]
    rows.sort(key=lambda i: times.iloc[i])
    if not rows:
        print("No dated history for this astronaut; no personal trend inferred.")
        return
    print("Trends use the first 3-5 explicitly documented reference readings, not a clinically validated healthy baseline.")
    for idx in rows:
        history = previous_measurements(db, idx)
        findings = personalized_trends(history, db["X"].iloc[idx].to_dict()) if history is not None else []
        print(str(meta.iloc[idx].Timestamp_UTC), "|", meta.iloc[idx].Case_ID, "|", "; ".join(f"{f['metric']}: {f['change_pct']}% vs reference" for f in findings) or ("No configured change" if history is not None else "Insufficient earlier reference observations"))
    idx = rows[-1]
    history = previous_measurements(db, idx)
    print_assessment(legacy_assess(bundle, db["Cases_24_metrics"].iloc[idx], history), meta.iloc[idx].Case_ID, history_available=history is not None)


def legacy_main():
    configure_output = getattr(sys.stdout, "reconfigure", None)
    if callable(configure_output):
        configure_output(errors="replace")
    try:
        db = load_database()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print("Cannot load MSHIMAS_Database.xlsx beside MSHIMAS.py:", exc)
        return 1
    bundle = None
    print(NOTICE)
    while True:
        print("\n========================================\n MSHIMAS - ASTRONAUT SURVIVAL METRICS\n Legacy Synthetic Feature Assessment\n========================================\n\n[1] Train and evaluate the ML model\n[2] Analyze an astronaut from the database\n[3] Enter new astronaut health measurements\n[4] Review available health metrics\n[5] Analyze astronaut health trends\n[6] Exit\n")
        try:
            option = input("Select an option: ").strip()
            if option == "6":
                print("MSHIMAS closed. No persistent outputs were created.")
                return 0
            if option == "4":
                for _, row in db["Dictionary_24_metrics"].iterrows():
                    print(f"{row.iloc[0]} | {row.iloc[1]} | {row.iloc[2]} | {row.iloc[4]}")
                print("metabolites: source zero placeholders remain in raw sheets; canonical values are unavailable. No analyte is specified.")
            elif option in ("1", "2", "3", "5"):
                if bundle is None or option == "1":
                    print("Training unsupervised model in memory...")
                    bundle = train_and_evaluate(db)
                if option == "2":
                    idx = select_case(db)
                    if idx is not None:
                        h = previous_measurements(db, idx)
                        partition = "training (not an unbiased evaluation)" if idx in bundle["train_indices"] else "held-out synthetic case" if idx in bundle["test_indices"] else "excluded from training and holdout"
                        print_assessment(legacy_assess(bundle, db["Cases_24_metrics"].iloc[idx], h), db["Case_Metadata"].iloc[idx].Case_ID, partition, h is not None)
                elif option == "3":
                    entered = manual_entry(db)
                    if entered is not None:
                        values, h = entered
                        print_assessment(legacy_assess(bundle, values, h), "manual entry (session only)", history_available=h is not None)
                elif option == "5":
                    show_trends(db, bundle)
            else:
                print("Invalid option. Choose 1-6.")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended; no data written.")
            return 0
        except (ValueError, KeyError, TypeError, OSError) as exc:
            print("Unable to complete this operation:", exc)



from types import SimpleNamespace
astra = SimpleNamespace(METRICS=METRICS, CAT=CAT, NUM=NUM, LEVELS=LEVELS, load_database=load_database, train_and_evaluate=train_and_evaluate, screening_rules=screening_rules, assess=legacy_assess)

# --- Instrument metrology and reference calculations ---
"""Instrument-aware research acquisition. No hardware or clinical certification.

Policy derived from the supplied engineering manual, pp. 4-42. Numerical
uncertainties in that manual are examples, never calibration certificates.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np

SYSTEM_NAME = "Monitoring System for Highly Important Metrics for Astronaut Survival"
SYSTEM_ID = "MSHIMAS"
POLICY_VERSION = "metrology-1.0"
SOURCE = "User-supplied engineering manual, 48 pages, sections 2-8"
SOURCE_SHA256 = "592d8bc196e7b4e3b41468464e68fcd360d859d6eca32fe5f254e7c403d63d7a"


def finite(value, name="value", *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError(f"{name} must be finite and at most 1e12 in magnitude")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def utc(value, name="timestamp_utc"):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp with timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"Invalid {name}") from None
    if result.tzinfo is None:
        raise ValueError(f"{name} requires an explicit timezone")
    return result.astimezone(timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def profile(key, label, unit, group, method, nature, span, page, rules=None, note="", model=False) -> dict[str, Any]:
    return dict(key=key, label=label, unit=unit, group=group, method_id=method,
                nature=nature, minimum=span[0], maximum=span[1], resolution=span[2],
                document_pages=page, quality_rules=rules or {}, note=note,
                legacy_model_feature=model, calibration_claim="Operator-declared; certificate not independently verified")


# The 24 historical feature names remain unchanged. New measurands stay outside
# that feature space; RNFL, MRW, QUS, DXA and muscle thickness are not aliases.
CATALOG = [
    profile("Hemoglobin (g/dl)", "Hemoglobin", "g/dL", "Laboratory", "hb_photometry", "direct", (0,30,.1), "4-6", {"sample_qc_pass": ["eq", True]}),
    profile("Hematocrit (%)", "Hematocrit", "%", "Laboratory", "hct_conductometry", "direct", (10,75,1), "4-6", {"sample_qc_pass": ["eq", True], "bubble_free": ["eq", True]}),
    profile("Leukocytes (k/uL)", "Leukocytes", "k/uL", "Laboratory", "laboratory_wbc", "direct", (0,100,.1), "Not specified", {"sample_qc_pass": ["eq", True]}, "Method and uncertainty must come from the laboratory report."),
    profile("Calcium (mg/dL)", "Total serum calcium", "mg/dL", "Laboratory", "total_calcium_lab", "direct", (0,30,.01), "8-10", {"sample_qc_pass": ["eq", True]}, "Total calcium. Ionized calcium is a separate measurand."),
    profile("Vitamin D (ng/mL)", "25-hydroxyvitamin D", "ng/mL", "Laboratory", "vitd_lcms", "direct", (0,160,.1), "6-8", {"hemolysis_mg_dl": ["max",50], "cold_chain_intact": ["eq",True]}),
    profile("alkaline phosphatase (U/L)", "Alkaline phosphatase", "U/L", "Laboratory", "alp_lab", "direct", (20,1000,1), "6-8", {"hemolysis_mg_dl": ["max",50], "cold_chain_intact": ["eq",True]}),
    profile("Calcium excretion (mg/24hr)", "24-hour calcium excretion", "mg/24h", "Laboratory", "urine_calcium_24h", "calculated", (0,2000,.1), "8-10", {"collection_complete": ["eq",True]}),
    profile("metabolites(ng/mL)", "Metabolites / legacy field", "ng/mL", "Laboratory", "unspecified_analyte", "unspecified", (0,500,.1), "Not specified", note="Unspecified analyte. Excluded from instrument-qualified processing."),
    profile("Sodium (mmol/L)", "Sodium", "mmol/L", "Laboratory", "sodium_ise", "direct", (0,250,.1), "4-6", {"sample_qc_pass": ["eq",True]}),
    profile("Heart Rate (Bpm)", "Heart rate", "bpm", "Cardiopulmonary", "ecg_hr", "calculated", (0,250,1), "12-15", {"signal_qc_pass": ["eq",True]}),
    profile("HRV (ms)", "HRV / RMSSD", "ms", "Cardiopulmonary", "ecg_rmssd", "calculated", (0,500,.01), "12-15", {"duration_s": ["min",240], "artifact_pct": ["lt",5], "stationary": ["eq",True]}),
    profile("Oxygen saturation (%)", "Peripheral oxygen saturation", "%", "Cardiopulmonary", "pulse_oximetry", "indirect", (70,100,.1), "15-16", {"perfusion_index_pct": ["min",.3], "valid_fraction_pct": ["min",80], "motion_free": ["eq",True]}, "Arms accuracy is not automatically an expanded uncertainty."),
    profile("SOS bone density (m/s)", "Bone SOS / legacy method", "m/s", "Musculoskeletal", "legacy_bone_sos", "unspecified", (0,5000,1), "17-18", note="Historical synthetic SOS values use an undocumented site/method. Not calcaneal QUS or DXA."),
    profile("Muscle Mass (kg)", "Muscle mass estimate", "kg", "Musculoskeletal", "muscle_mass_model", "indirect", (0,150,.1), "19-21", {"volume_model_documented": ["eq",True]}, "2D thickness alone cannot establish whole-body muscle mass."),
    profile("Aerobic Capacity ml⁄min/kg", "Submaximal aerobic capacity", "mL/kg/min", "Musculoskeletal", "submax_vo2", "indirect", (0,150,.01), "31-32", {"gas_calibration_pass": ["eq",True], "stationary": ["eq",True], "stages": ["min",3], "correlation_r": ["min",.97]}),
    profile("Work W", "Mechanical power", "W", "Operations", "ergometer_power", "direct", (0,1500,1), "27-28", {"zero_verified": ["eq",True]}),
    profile("Radiation per day(millirads)", "Daily absorbed dose", "mrad/day", "Environment", "absorbed_dose", "direct", (0,5000,.1), "21-22", {"completeness_pct": ["min",95], "saturated": ["eq",False]}, "Absorbed dose, not equivalent or effective dose."),
    profile("Cabin Pressure/atm (kPa)", "Cabin absolute pressure", "kPa", "Environment", "cabin_mems", "direct", (0,115,.001), "23-24", {"sensor_spread_kpa": ["max",.2], "sensor_count": ["min",3]}),
    profile("Space Suit Pressure (kPa)", "Suit differential pressure", "kPa", "Environment", "suit_pressure", "direct", (0,115,.001), "23-26", {"zero_verified": ["eq",True]}),
    profile("Retinal & Optic Nerve Thickening(um) minimum rim width;", "Optic nerve minimum rim width", "um", "Neuro-ocular", "oct_mrw", "direct", (0,1000,.1), "29-30", {"oct_quality_db": ["min",20], "same_location": ["eq",True]}, "Minimum rim width is not RNFL thickness."),
    profile("Age", "Age", "years", "Context", "age_record", "context", (0,120,1), "Context"),
    profile("Height", "Height", "cm", "Context", "stadiometry", "direct", (0,250,.1), "Context", {"zero_verified": ["eq",True]}),
    profile("ionized_calcium", "Ionized calcium", "mmol/L", "Laboratory", "calcium_ise", "direct", (.25,2.5,.01), "8-10", {"anaerobic_sample": ["eq",True]}),
    profile("hs_crp", "High-sensitivity C-reactive protein", "mg/L", "Laboratory", "crp_lab", "direct", (.1,20,.01), "6-8", {"hemolysis_mg_dl": ["max",50], "cold_chain_intact": ["eq",True]}),
    profile("urine_specific_gravity", "Urine specific gravity", "1", "Hydration", "refractometry", "direct", (1,1.06,.001), "11-12", {"temperature_c": ["range",[10,40]], "duplicate_difference": ["max",.002], "zero_verified": ["eq",True]}),
    profile("calcaneal_sos", "Calcaneal speed of sound", "m/s", "Musculoskeletal", "calcaneal_qus", "direct", (1450,1650,.1), "17-18", {"phantom_pass": ["eq",True], "replicates": ["min",3], "cv_pct": ["max",.5]}, "Acoustic speed; not bone mineral density."),
    profile("calcaneal_bua", "Calcaneal broadband attenuation", "dB/MHz", "Musculoskeletal", "calcaneal_bua", "direct", (20,160,.1), "17-18", {"phantom_pass": ["eq",True], "replicates": ["min",3]}),
    profile("muscle_thickness", "Rectus femoris thickness", "mm", "Musculoskeletal", "bmode_thickness", "direct", (8,55,.1), "19-21", {"gel_mm": ["min",2], "cv_pct": ["lt",2], "replicates": ["min",3], "operator_review_pass": ["eq",True]}),
    profile("rnfl_od", "RNFL thickness / right eye", "um", "Neuro-ocular", "oct_rnfl_od", "direct", (0,250,.1), "29-30", {"oct_quality_db": ["min",20], "same_location": ["eq",True]}),
    profile("rnfl_oi", "RNFL thickness / left eye", "um", "Neuro-ocular", "oct_rnfl_oi", "direct", (0,250,.1), "29-30", {"oct_quality_db": ["min",20], "same_location": ["eq",True]}),
    profile("sleep_tst", "Total sleep time", "min", "Cognitive", "actigraphy_tst", "indirect", (0,1440,1), "35-36", {"off_wrist_min": ["max",60], "wear_verified": ["eq",True]}),
    profile("pvt_speed", "PVT-B reciprocal response speed", "1/s", "Cognitive", "pvt_b", "calculated", (0,10,.01), "37-38", {"valid_stimuli": ["min",25], "false_starts": ["max",5]}),
    profile("functional_torque", "Functional isometric torque", "Nm", "Musculoskeletal", "dynamometry", "calculated", (0,1500,.1), "33-34", {"zero_verified": ["eq",True], "replicates": ["min",3], "cv_pct": ["lt",5]}),
    profile("ppco2", "Cabin carbon dioxide partial pressure", "mmHg", "Environment", "co2_ndir", "direct", (0,20,.01), "23-24", {"zero_verified": ["eq",True]}),
    profile("eva_energy", "EVA energy expenditure", "kcal", "Operations", "eva_lcvg", "indirect", (0,6000,1), "25-26", {"thermal_consistency": ["eq",True]}),
    profile("postural_sway", "Postural sway / sensorimotor", "mm", "Cognitive", "posturography", "direct", (0,500,.1), "45", {"signal_qc_pass": ["eq",True]}, "Device-specific uncertainty is not specified by the document."),
]
BY_KEY = {p["key"]: p for p in CATALOG}

# Embedded reference budgets: (absolute standard component, relative standard
# component, scope). These describe assumptions, not the user's real instrument.
# Where the manual only gives an accuracy bound, rectangular distribution is an
# explicit engineering assumption. Resolution-only budgets remain labelled partial.
REFERENCE_BUDGETS = {
    "hb_photometry":(.3/math.sqrt(3),0,"accuracy-bound assumption"),
    "hct_conductometry":(.75,0,"document example"),
    "total_calcium_lab":(0,.03/math.sqrt(3),"accuracy-bound assumption"),
    "vitd_lcms":(0,.06,"document example"), "alp_lab":(0,.05,"document example"),
    "crp_lab":(0,.05,"document example"), "calcium_ise":(.01,0,"document example"),
    "urine_calcium_24h":(0,math.sqrt(.015**2+.005**2+.015**2),"document example"),
    "refractometry":(.001/math.sqrt(3),0,"accuracy bound plus duplicate repeatability"),
    "ecg_rmssd":(.408,0,"document timing-only approximation; partial budget"),
    "pulse_oximetry":(2,0,"Arms proxy assumed as standard component; not certified GUM uncertainty"),
    "calcaneal_qus":(3.1,0,"document example"), "calcaneal_bua":(1.65,0,"document example"),
    "bmode_thickness":(.4,0,"document example"),
    "absorbed_dose":(0,.05/math.sqrt(3),"detector calibration bound only; partial budget"),
    "cabin_mems":(.025,0,"document example"), "suit_pressure":(.025,0,"reference MEMS assumption"),
    "submax_vo2":(0,.039,"document extrapolation example"),
    "oct_rnfl_od":(2,0,"document longitudinal example"), "oct_rnfl_oi":(2,0,"document longitudinal example"),
    "oct_mrw":(2,0,"reference OCT assumption; MRW-specific budget unavailable"),
    "actigraphy_tst":(14,0,"document classification-error example"),
    "pvt_b":(.06,0,"document response-speed example"),
    "dynamometry":(0,.0105,"document torque example"),
    "eva_lcvg":(0,.06,"document thermal-model example"),
}


def reference_budget(p, value, replicates=None) -> dict[str, Any]:
    absolute,relative,scope=REFERENCE_BUDGETS.get(p["method_id"],(0,0,"resolution-only; incomplete instrument budget"))
    resolution=p["resolution"]/math.sqrt(12)
    # Document budgets already aggregate instrument contributions; quantization
    # is added only where no instrument budget is available, avoiding double count.
    u_b=math.hypot(absolute,relative*abs(value)) if absolute or relative else resolution
    u_a=0.0
    if replicates is not None:
        if not isinstance(replicates,list) or len(replicates)<2:raise ValueError("replicates requires at least two numeric readings")
        arr=np.array([finite(x,"replicate") for x in replicates])
        u_a=float(np.std(arr,ddof=1)/math.sqrt(len(arr)))
    if p["method_id"]=="refractometry" and replicates is None:u_a=.0005
    return dict(standard_uncertainty=math.hypot(u_a,u_b),type_a=u_a,type_b=u_b,
                coverage_k=2,source=f"Embedded reference budget; supplied manual pp. {p['document_pages']}",
                scope=scope,calibration_verified=False)


for _p in CATALOG:
    _p["reference_budget"]=reference_budget(_p,(_p["minimum"]+_p["maximum"])/2)
    _p["range_scope"]="Document method span" if _p["method_id"] in {
        "hct_conductometry","alp_lab","pulse_oximetry","calcium_ise","crp_lab",
        "refractometry","calcaneal_qus","calcaneal_bua","cabin_mems","bmode_thickness"
    } else "Input span; device-specific analytical limits not established"


def convert(value, unit, canonical, metric=None):
    """Only explicit, dimensionally compatible conversions. Never Ca total <-> iCa."""
    if unit == canonical:
        return value, 1.0
    factors = {
               ("mmHg","kPa"): .133322387415, ("Gy/day","mrad/day"):1e5,
               ("uGy/day","mrad/day"): .1, ("m","cm"):100,
               ("g/L","g/dL"):.1, ("fraction","%"):100}
    if metric=="Vitamin D (ng/mL)":factors[("nmol/L","ng/mL")]=1/2.496
    if metric=="Calcium excretion (mg/24hr)":factors[("mmol/day","mg/24h")]=40.078
    if metric=="Calcium (mg/dL)":factors[("mmol/L","mg/dL")]=4.0078
    factor = factors.get((unit, canonical))
    if factor is None:
        raise ValueError(f"Unsupported conversion from {unit!r} to {canonical!r}")
    return value * factor, factor


def baseline_change(value, u, baseline, method_id, unit, k: float=2) -> dict[str, Any] | None:
    if not baseline:
        return None
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be an object")
    if baseline.get("method_id") != method_id or baseline.get("unit") != unit:
        return {"available":False, "reason":"Baseline method/unit differs from this observation"}
    if finite(baseline.get("n"), "baseline n", minimum=0) < 3:
        return {"available":False, "reason":"At least three baseline observations are required"}
    mean=finite(baseline.get("mean"),"baseline mean")
    sd=finite(baseline.get("sd"),"baseline sd",minimum=0)
    u0=finite(baseline.get("standard_uncertainty"),"baseline standard uncertainty",minimum=0)
    denom=math.sqrt(sd*sd+u*u+u0*u0)
    delta=value-mean
    U_delta=k*math.hypot(u,u0)
    z=delta/denom if denom else None
    level="NO_CHANGE"
    if z is not None and abs(delta)>U_delta:
        if abs(z)>=2.5: level="REVIEW"
        elif abs(z)>=1.5: level="MONITOR"
    pct=100*delta/mean if mean else None
    U_pct=k*math.hypot(100*u/mean,100*value*u0/(mean*mean)) if mean else None
    return dict(available=True, delta=delta, expanded_uncertainty_delta=U_delta,
                change_pct=pct, expanded_uncertainty_change_pct=U_pct,
                z_individual=z, level=level, assumption="Independent measurement uncertainties; operator-supplied baseline")


def process_observation(key, raw, now=None):
    if key not in BY_KEY or not isinstance(raw,dict):
        raise ValueError(f"Unknown metric or invalid observation: {key}")
    p=BY_KEY[key]; now=now or datetime.now(timezone.utc)
    reasons=[]; missing=[]
    value=finite(raw.get("value"),key)
    unit=raw.get("unit",p["unit"])
    value,factor=convert(value,unit,p["unit"],key)
    mode=raw.get("acquisition_mode","manual")
    if mode not in ("manual","instrument","ground_lab","experimental","synthetic"):
        raise ValueError("Unsupported acquisition_mode")
    method=raw.get("method_id",p["method_id"])
    if method != p["method_id"]: reasons.append("METHOD_MISMATCH")
    if p["nature"]=="unspecified": reasons.append("UNSPECIFIED_MEASURAND_OR_METHOD")
    if not p["minimum"]<=value<=p["maximum"]: reasons.append("OUTSIDE_METHOD_RANGE")
    timestamp=raw.get("timestamp_utc",now.isoformat())
    if timestamp:
        t=utc(timestamp)
        if (t-now).total_seconds()>300: reasons.append("FUTURE_ACQUISITION_TIME")
    else: missing.append("timestamp_utc"); t=None
    if not isinstance(raw.get("sensor_id"),str) or not raw["sensor_id"].strip(): missing.append("physical sensor identity")
    if mode=="manual": missing.append("physical instrument acquisition evidence")
    calibration=raw.get("calibration",{})
    if not isinstance(calibration,dict): raise ValueError("calibration must be an object")
    if calibration.get("status")=="fail": reasons.append("CALIBRATION_FAILED")
    elif calibration.get("status")!="pass": missing.append("calibration.status")
    if calibration.get("status")=="pass":
        if not calibration.get("certificate_id"): missing.append("calibration.certificate_id")
        if not calibration.get("date_utc") or not calibration.get("due_utc"):
            missing.append("calibration dates")
        else:
            date=utc(calibration["date_utc"],"calibration date")
            due=utc(calibration["due_utc"],"calibration due date")
            if due<date: reasons.append("INVALID_CALIBRATION_INTERVAL")
            if t and (t<date or t>due): reasons.append("CALIBRATION_NOT_VALID_AT_ACQUISITION")
    quality=raw.get("quality",{})
    if not isinstance(quality,dict): raise ValueError("quality must be an object")
    for name,(operator,bound) in p["quality_rules"].items():
        if name not in quality: missing.append("quality."+name); continue
        observed=quality[name]
        if operator=="eq":
            passed=type(observed) is type(bound) and observed==bound
        else:
            observed=finite(observed,"quality."+name)
            passed={"min":lambda:observed>=bound,"max":lambda:observed<=bound,
                    "lt":lambda:observed<bound,"range":lambda:bound[0]<=observed<=bound[1]}[operator]()
        if not passed: reasons.append("QUALITY_"+name.upper())
    replicates=raw.get("replicates")
    if replicates is not None:
        if not isinstance(replicates,list):raise ValueError("replicates must be an array")
        replicates=[finite(x,"replicate")*factor for x in replicates]
    budget=reference_budget(p,value,replicates)
    u=None; k=finite(raw.get("coverage_k",2),"coverage_k",minimum=1,maximum=10)
    if raw.get("standard_uncertainty") is not None:
        u=finite(raw["standard_uncertainty"],"standard_uncertainty",minimum=0)*abs(factor)
    elif raw.get("expanded_uncertainty") is not None:
        u=finite(raw["expanded_uncertainty"],"expanded_uncertainty",minimum=0)*abs(factor)/k
    else: u=budget["standard_uncertainty"]; k=2
    reference_used=raw.get("standard_uncertainty") is None and raw.get("expanded_uncertainty") is None
    gate="REJECTED" if reasons else "SIMULATED" if mode=="synthetic" else "REFERENCE" if missing or reference_used else "ACCEPTED"
    result: dict[str, Any]=dict(metric=key,label=p["label"],value=value,unit=p["unit"],nature=p["nature"],
                method_id=method,sensor_id=raw.get("sensor_id"),timestamp_utc=timestamp,
                timestamp_origin="supplied" if raw.get("timestamp_utc") else "local entry time; sensor acquisition time unknown",
                acquisition_mode=mode,gate=gate,rejection_codes=reasons,missing_evidence=missing,
                standard_uncertainty=u,expanded_uncertainty=(k*u if u is not None else None),
                coverage_k=k,uncertainty_source=budget["source"] if reference_used else raw.get("uncertainty_source","Imported uncertainty"),
                uncertainty_budget=budget if reference_used else None,
                usable_for_research=gate in ("ACCEPTED","SIMULATED","REFERENCE"),
                document_pages=p["document_pages"],raw_sha256=digest(raw),
                calibration=calibration,quality=quality,
                calibration_verification="Declaration only; no certificate authenticity validation")
    result["baseline_change"]=baseline_change(value,u,raw.get("baseline"),method,p["unit"],k) if u is not None and result["usable_for_research"] else None
    return result


def acquire(observations, context=None, now=None):
    if not isinstance(observations,dict) or len(observations)>len(CATALOG):
        raise ValueError("observations must be an object keyed by registered metric")
    processed=[process_observation(key,raw,now) for key,raw in observations.items()]
    context=context or {}
    if not isinstance(context,dict): raise ValueError("context must be an object")
    counts={state:sum(p["gate"]==state for p in processed) for state in ("ACCEPTED","SIMULATED","REFERENCE","REJECTED")}
    flight_day=None
    if context.get("launch_utc") and context.get("timestamp_utc"):
        delta=(utc(context["timestamp_utc"])-utc(context["launch_utc"],"launch_utc")).total_seconds()
        flight_day=math.floor(delta/86400)+(1 if delta>=0 else 0)
    return dict(policy_version=POLICY_VERSION,processed=processed,counts=counts,
                flight_day=flight_day,raw_sha256=digest(observations),
                scope="Manual/JSON adapter. No physical sensor connection or time synchronization is asserted.")


def derive_indicators(processed):
    """Automatically derive available indicators without replacing measured values."""
    valid={p["metric"]:p for p in processed if p["usable_for_research"]}
    results=[]
    def add(name,value,U,unit,inputs,equation,nature="calculated",scope="Independent input uncertainties; embedded reference assumptions"):
        results.append(dict(name=name,value=value,expanded_uncertainty=U,coverage_k=2,unit=unit,
                            inputs=inputs,equation=equation,nature=nature,scope=scope,
                            state="SIMULATED" if any(valid[k]["gate"]=="SIMULATED" for k in inputs) else "REFERENCE"))
    hct=valid.get("Hematocrit (%)")
    if hct:
        value=.34*hct["value"]
        U=2*math.hypot(.34*hct["standard_uncertainty"],hct["value"]*.012)
        add("Hemoglobin estimate from hematocrit",value,U,"g/dL",[hct["metric"]],"Hb_est = 0.34 Hct; empirical u(k_MCHC) = 0.012", "indirect")
    sos,bua=valid.get("calcaneal_sos"),valid.get("calcaneal_bua")
    if sos and bua:
        value=.67*bua["value"]+.28*sos["value"]-420
        U=2*math.hypot(.67*bua["standard_uncertainty"],.28*sos["standard_uncertainty"])
        add("Calcaneal stiffness index",value,U,"1",[sos["metric"],bua["metric"]],"SI = 0.67 BUA + 0.28 SOS - 420","indirect","Peripheral acoustic estimate, not bone mineral density; independent input uncertainties assumed")
    vitd=valid.get("Vitamin D (ng/mL)")
    if vitd:add("25(OH)D in SI units",vitd["value"]*2.496,vitd["expanded_uncertainty"]*2.496,"nmol/L",[vitd["metric"]],"C_nmol/L = 2.496 C_ng/mL")
    usg=valid.get("urine_specific_gravity")
    if usg:add("Urine osmolality estimate",33333*(usg["value"]-1),33333*usg["expanded_uncertainty"],"mOsm/kg",[usg["metric"]],"Uosm_est = 33333 (USG - 1)","indirect","Input uncertainty only; regression residual and solute effects are not quantified")
    return results


def vector(data, name, minimum=None, maximum=None, at_least=1):
    seq=data.get(name)
    if not isinstance(seq,list) or not at_least<=len(seq)<=300000:
        raise ValueError(f"{name} requires {at_least} to 300000 samples")
    return np.array([finite(x,name,minimum=minimum,maximum=maximum) for x in seq],dtype=float)


def calculate(kind, data):
    """Auditable pure calculations. Acceptance here is processing QC only, not calibration."""
    if not isinstance(data,dict): raise ValueError("inputs must be an object")
    n=lambda key,low=None,high=None: finite(data.get(key),key,minimum=low,maximum=high)
    v=lambda key,low=None,high=None,count=1:vector(data,key,low,high,count)
    out={"processor":kind,"version":POLICY_VERSION,"accepted":True,"nature":"calculated",
         "calibration_verified":False,"source":SOURCE,
         "uncertainty_note":"Embedded method-reference components are used automatically unless imported components are present. No physical calibration or clinical coverage is certified."}
    def reject(reason): return {**out,"accepted":False,"reason":reason}
    if kind=="hb_hct":
        h=n("hct_pct",10,75); u=finite(data.get("u_hct",.75),"u_hct",minimum=0); ku=finite(data.get("u_mchc",.012),"u_mchc",minimum=0)
        hb=.34*h; uh=math.hypot(.34*u,h*ku)
        out.update(hemoglobin_estimate_g_dl=hb,expanded_uncertainty_hb=2*uh,coverage_k=2,
                   nature="indirect",equation="Hb = 0.34 Hct; uc = hypot(0.34 uHct, Hct uk)")
        if data.get("baseline_hb") is not None:
            h0=n("baseline_hct",0,99.99);hb0=n("baseline_hb",.000001)
            out["delta_plasma_volume_pct"]=100*hb0/hb*(1-h/100)/(1-h0/100)-100
    elif kind=="urine_calcium":
        ca=n("calcium_mmol_l",0);li=n("lithium_umol_l",.000001);mass=n("tracer_umol",.000001)
        cr=n("creatinine_mmol_l",.000001);body=n("body_mass_kg",.000001)
        volume=mass/li;excretion=ca*volume
        complete=.11*body<=cr*volume<=.25*body
        relative=math.sqrt(sum(finite(data.get(key,default),key,minimum=0)**2 for key,default in [("u_relative_calcium",.015),("u_relative_tracer",.005),("u_relative_lithium",.015)]))
        out.update(volume_l=volume,calcium_creatinine_ratio=ca/cr,collection_complete=complete,
                   calcium_mmol_day=excretion if complete else None,
                   expanded_uncertainty=2*relative*excretion if complete else None,coverage_k=2)
    elif kind=="hydration":
        a=n("usg_1",1,1.06);b=n("usg_2",1,1.06);temp=n("temperature_c")
        if not 10<=temp<=40 or abs(a-b)>.002+1e-12: return reject("USG duplicate disagreement or temperature outside 10-40 C")
        mean=(a+b)/2;out.update(usg=mean,osmolality_estimate_mosm_kg=33333*(mean-1),nature="indirect")
    elif kind=="spo2_window":
        spo=v("spo2_pct"); pi=v("perfusion_index_pct",0);motion=data.get("motion")
        if not isinstance(motion,list) or not len(spo)==len(pi)==len(motion) or any(type(x) is not bool for x in motion): raise ValueError("SpO2, perfusion and boolean motion arrays must align")
        mask=(pi>=.3)&~np.array(motion)&(spo>=70)&(spo<=100)
        valid=float(mask.mean()*100)
        if valid<80: return reject(f"Only {valid:.1f}% valid SpO2 samples; at least 80% required")
        out.update(spo2_median_pct=float(np.median(spo[mask])),spo2_p05_pct=float(np.percentile(spo[mask],5)),valid_fraction_pct=valid,nature="indirect")
    elif kind=="hrv_nn":
        from scipy.interpolate import CubicSpline
        from scipy.signal import welch, detrend
        rr=v("rr_ms",.000001,count=3)
        if rr.sum()/1000<240:return reject("At least 240 seconds of stationary RR data required")
        if data.get("stationary") is not True:return reject("Stationarity must be confirmed")
        med=np.array([np.median(rr[max(0,i-2):i+3]) for i in range(len(rr))])
        bad=(rr<300)|(rr>2000)|(np.abs(rr-med)>.2*med)
        pct=float(bad.mean()*100)
        if pct>=5:return reject("At least 5% edited/ectopic intervals; repeat acquisition")
        good=np.flatnonzero(~bad)
        if len(good)<3:return reject("Too few valid intervals")
        clean=rr.copy()
        if bad.any():clean[bad]=CubicSpline(good,rr[good],extrapolate=False)(np.flatnonzero(bad))
        if not np.isfinite(clean).all() or (clean<=0).any():return reject("Edge artifacts cannot be safely interpolated")
        out.update(rmssd_ms=float(np.sqrt(np.mean(np.diff(clean)**2))),sdnn_ms=float(np.std(clean,ddof=1)),
                   heart_rate_bpm=float(60000/clean.mean()),artifact_pct=pct,spectral_available=False)
        if rr.sum()/1000>=300:
            times=np.cumsum(clean)/1000;grid=np.arange(times[0],times[-1],.25)
            uniform=CubicSpline(times,clean)(grid)
            freqs,psd=welch(detrend(uniform),fs=4,nperseg=min(256,len(uniform)))
            band=lambda lo,hi:float(np.trapezoid(psd[(freqs>=lo)&(freqs<hi)],freqs[(freqs>=lo)&(freqs<hi)]))
            lf,hf=band(.04,.15),band(.15,.4)
            out.update(lf_ms2=lf,hf_ms2=hf,hf_normalized_pct=100*hf/(lf+hf) if lf+hf else None,spectral_available=True)
        out["scope"]="Imported RR intervals; R-peak detection from raw ECG is not performed. RMSSD uncertainty requires its own budget."
    elif kind=="qus":
        sos=v("sos_m_s",1450,1650,3);bua=v("bua_db_mhz",20,160,3)
        if len(sos)!=len(bua):raise ValueError("QUS replicate arrays must align")
        cv=float(np.std(sos,ddof=1)/np.mean(sos)*100)
        if data.get("phantom_pass") is not True or cv>.5:return reject("QUS phantom failure or SOS CV > 0.5%")
        si=.67*float(bua.mean())+.28*float(sos.mean())-420
        U=2*math.hypot(.67*finite(data.get("u_bua",1.65),"u_bua",minimum=0),.28*finite(data.get("u_sos",3.1),"u_sos",minimum=0))
        out.update(stiffness_index=si,expanded_uncertainty=U,coverage_k=2,sos_mean_m_s=float(sos.mean()),cv_pct=cv,nature="indirect",scope="Peripheral acoustic indicator; not DXA bone mineral density")
    elif kind=="muscle_change":
        current=n("current_mm",.000001);base=n("baseline_mm",.000001)
        u=finite(data.get("u_current_mm",.4),"u_current_mm",minimum=0);u0=finite(data.get("u_baseline_mm",.4),"u_baseline_mm",minimum=0)
        delta=100*(current-base)/base;U=2*math.hypot(100*u/base,100*current*u0/base**2)
        out.update(change_pct=delta,expanded_uncertainty_change_pct=U,coverage_k=2,exceeds_instrument_uncertainty=abs(delta)>U)
    elif kind=="pressure":
        sensors=v("sensors_kpa",0,115,3);series=v("history_kpa",0,115,2);fs=n("sample_rate_hz",.01,10000)
        spread=float(np.ptp(sensors))
        if spread>.2+1e-12:return reject("Redundant pressure sensor spread > 0.20 kPa")
        if len(series)/fs<10:return reject("At least ten seconds of pressure history required")
        y=series[-int(10*fs):];t=np.arange(len(y))/fs
        slope=float(60*np.polyfit(t,y,1)[0])
        out.update(median_kpa=float(np.median(sensors)),slope_kpa_min=slope,
                   decreasing_faster_than_document_example=slope<-.05,
                   scope="Descriptive pressure trend; vehicle-specific operational limits must be supplied separately")
    elif kind=="radiation":
        dose=v("dose_rates_ugy_h",0);let=v("let_kev_um",.000001)
        if len(dose)!=len(let):raise ValueError("Dose and LET bins must align")
        q=np.where(let<10,1,np.where(let<=100,.32*let-2.2,300/np.sqrt(let)))
        duration=n("duration_h",0)
        out.update(absorbed_dose_ugy=float(dose.sum()*duration),equivalent_dose_msv=float(np.sum(dose*q)*duration/1000),
                   nature="indirect",scope="Q(L)-weighted point dose estimate. Effective dose and career-limit fractions are not computed.")
    elif kind=="pvt":
        rt=v("reaction_times_ms",0,30000);false=int(np.sum(rt<100));valid=rt[rt>=100]
        if false>5 or len(valid)<25:return reject("PVT requires at least 25 valid stimuli and at most five false starts")
        out.update(reciprocal_speed_s_inv=float(np.mean(1000/valid)),median_rt_ms=float(np.median(valid)),lapses_gt355ms=int(np.sum(valid>355)),false_starts=false,valid_stimuli=len(valid))
    elif kind=="workload":
        hr=v("heart_rate_bpm",0,250);rest=n("resting_hr",0,249);maximum=n("maximum_hr",rest+.01,250)
        dt=n("sample_interval_min",.000001);sex=data.get("sex")
        if sex not in ("F","M"):raise ValueError("TRIMP coefficient set requires F or M")
        a,b=(.64,1.92) if sex=="M" else (.86,1.67)
        reserve=np.clip((hr-rest)/(maximum-rest),0,1)
        load=v("loads_kg",0);reps=v("repetitions",0)
        if len(load)!=len(reps) or (reps!=np.floor(reps)).any():raise ValueError("Loads and integer repetitions must align")
        daily=data.get("daily_trimp_28d",[]);acwr=None
        if daily:
            hist=v("daily_trimp_28d",0)
            if len(hist)>=28 and hist[-28:].sum()>0:acwr=float(hist[-7:].sum()/(hist[-28:].mean()*7))
        out.update(trimp=float(np.sum(dt*reserve*a*np.exp(b*reserve))),ared_tonnage_kg=float(np.sum(load*reps)),acwr=acwr,scope="ACWR describes relative load progression; no injury probability or spaceflight threshold")
    elif kind=="vo2_submax":
        hr=v("stage_hr",0,250,3);vo=v("stage_vo2",0,200,3)
        if len(hr)!=len(vo) or np.ptp(hr)==0 or np.ptp(vo)==0:raise ValueError("Distinct, aligned HR and VO2 stages required")
        if data.get("gas_calibration_pass") is not True or data.get("stationary") is not True:return reject("Gas calibration and steady state must pass")
        slope,intercept=np.polyfit(hr,vo,1);r=float(np.corrcoef(hr,vo)[0,1])
        if r<.97 or slope<=0:return reject("Positive HR/VO2 relation with r >= 0.97 required")
        out.update(vo2max_estimate_ml_kg_min=float(slope*n("preflight_max_hr",max(hr),250)+intercept),correlation_r=r,nature="indirect")
    elif kind=="sleep_epochs":
        scores=data.get("sleep_epochs")
        if not isinstance(scores,list) or not 1<=len(scores)<=100000 or any(type(x) is not bool for x in scores):raise ValueError("sleep_epochs requires boolean sleep/wake classifications")
        if n("off_wrist_min",0)>60:return reject("Off-wrist period > 60 minutes")
        minutes=n("epoch_seconds",1,60)/60
        out.update(total_sleep_min=sum(scores)*minutes,time_in_bed_min=len(scores)*minutes,sleep_efficiency_pct=100*sum(scores)/len(scores),nature="indirect",scope="Summary of imported classified epochs. Cole-Kripke device coefficients are not assumed.")
    else:raise ValueError(f"Unknown processor: {kind}")
    automatic={
        "hydration":("usg", "urine_specific_gravity"),
        "spo2_window":("spo2_median_pct", "Oxygen saturation (%)"),
        "hrv_nn":("rmssd_ms", "HRV (ms)"),
        "pressure":("median_kpa", "Cabin Pressure/atm (kPa)"),
        "pvt":("reciprocal_speed_s_inv", "pvt_speed"),
        "vo2_submax":("vo2max_estimate_ml_kg_min", "Aerobic Capacity ml⁄min/kg"),
        "sleep_epochs":("total_sleep_min", "sleep_tst"),
    }
    if kind in automatic:
        field,key=automatic[kind]
        budget=reference_budget(BY_KEY[key],out[field])
        out.update(expanded_uncertainty=2*budget["standard_uncertainty"],coverage_k=2,uncertainty_budget=budget)
    if kind=="radiation":
        out.update(expanded_uncertainty_equivalent_dose_msv=.112*out["equivalent_dose_msv"],coverage_k=2,
                   uncertainty_scope="Manual's 11.2% expanded reference assumption; not a device calibration")
    out["input_sha256"]=digest(data)
    return out


PROCESSOR_EXAMPLES = {
    "hb_hct":{"hct_pct":45,"u_hct":.75,"u_mchc":.012},
    "urine_calcium":{"calcium_mmol_l":3.7,"lithium_umol_l":500,"tracer_umol":1000,"creatinine_mmol_l":6,"body_mass_kg":70,"u_relative_calcium":.015,"u_relative_tracer":.005,"u_relative_lithium":.015},
    "hydration":{"usg_1":1.024,"usg_2":1.025,"temperature_c":22},
    "spo2_window":{"spo2_pct":[98,98,97,98,98],"perfusion_index_pct":[1,1,1,1,1],"motion":[False]*5},
    "hrv_nn":{"rr_ms":[800,810,790,805,795]*80,"stationary":True},
    "qus":{"sos_m_s":[1550,1551,1549],"bua_db_mhz":[90,91,89],"phantom_pass":True,"u_sos":3.1,"u_bua":1.65},
    "muscle_change":{"current_mm":23,"baseline_mm":25,"u_current_mm":.4,"u_baseline_mm":.4},
    "pressure":{"sensors_kpa":[101.3,101.31,101.29],"history_kpa":[101.3-i*.0001 for i in range(100)],"sample_rate_hz":10},
    "radiation":{"dose_rates_ugy_h":[2,1,.5],"let_kev_um":[5,50,150],"duration_h":1},
    "pvt":{"reaction_times_ms":[250,260,270,280,390]*6},
    "workload":{"heart_rate_bpm":[100,105,110],"resting_hr":60,"maximum_hr":180,"sample_interval_min":1,"sex":"F","loads_kg":[50,60],"repetitions":[10,8],"daily_trimp_28d":[30]*28},
    "vo2_submax":{"stage_hr":[100,120,140],"stage_vo2":[15,20,25],"preflight_max_hr":180,"gas_calibration_pass":True,"stationary":True},
    "sleep_epochs":{"sleep_epochs":[True]*800+[False]*100,"epoch_seconds":30,"off_wrist_min":0},
}

metrology = SimpleNamespace(**{key: value for key, value in globals().copy().items() if not key.startswith("__")})

# --- Local persistence ---

DEMO_NAMES = {"Simulation Test Alpha", "Simulation Test Beta"}
"""Excel-backed crew and session storage; source sheets are preserved."""
import unicodedata
from uuid import uuid4
from threading import RLock
import os
import tempfile
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment

_excel_lock = RLock()
_excel_cache: dict[str, tuple[Any, Any]] = {}
_SESSION_HEADERS = ['Record ID','Created UTC','SHA256','Part','Payload']
_CREW_HEADERS = ['ID','Name','Name key','Created UTC']
_MEASUREMENT_HEADERS = ['Record ID','Created UTC','Crew ID','Crew name','Sex','Mission role','Synthetic','Metric','Value','Unit','Method','Status']

def _excel_book(path):
    path=Path(path)
    signature=(path.stat().st_mtime_ns,path.stat().st_size) if path.exists() else None
    cached=_excel_cache.get(str(path.resolve()))
    if cached and cached[0]==signature:return cached[1]
    if path.exists():book=load_workbook(path)
    else:
        book=Workbook();book.remove(book.worksheets[0])
    for name,headers in [('Crew_Registry',_CREW_HEADERS),('Session_Records',_SESSION_HEADERS),('Session_Measurements',_MEASUREMENT_HEADERS)]:
        if name not in book.sheetnames:
            ws=book.create_sheet(name);ws.append(headers);ws.freeze_panes='A2'
            for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='303030')
            for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=24
    _excel_cache[str(path.resolve())]=(signature,book)
    return book

def _commit_excel(path,book):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle,temporary=tempfile.mkstemp(prefix='.mshimas-',suffix='.xlsx',dir=path.parent)
    os.close(handle)
    try:
        book.save(temporary)
        os.replace(temporary,path)
    except PermissionError as exc:
        _excel_cache.pop(str(path.resolve()),None)
        raise ValueError('Close the Excel workbook before saving this session, then press Save again.') from exc
    except Exception:
        _excel_cache.pop(str(path.resolve()),None)
        raise
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    _excel_cache[str(path.resolve())]=((path.stat().st_mtime_ns,path.stat().st_size),book)

def _text_row(ws,values):
    ws.append(values)
    for cell in ws[ws.max_row]:
        if isinstance(cell.value,str):cell.data_type='s'

def _crew_dict(row):
    return dict(id=row[0],name=row[1],created_utc=row[3],is_demo=row[1] in DEMO_NAMES)

def register_crew(path: Path,name):
    if not isinstance(name,str):raise ValueError('Crew name must be text')
    name=unicodedata.normalize('NFC',' '.join(name.split()))
    if not 1<=len(name)<=100 or any(ord(c)<32 for c in name):raise ValueError('Use a crew name with 1 to 100 characters')
    with _excel_lock:
        book=_excel_book(path);ws=book['Crew_Registry']
        for row in ws.iter_rows(min_row=2,values_only=True):
            if row[2]==name.casefold():return _crew_dict(row)
        row=(str(uuid4()),name,name.casefold(),datetime.now(timezone.utc).isoformat())
        _text_row(ws,row);_commit_excel(path,book)
        return _crew_dict(row)

def list_crew(path: Path):
    if not path.exists():return []
    with _excel_lock:
        rows=list(_excel_book(path)['Crew_Registry'].iter_rows(min_row=2,values_only=True))
        return [_crew_dict(r) for r in sorted(rows,key=lambda r:str(r[2])) if r[0]]

def _records(path):
    if not path.exists():return []
    grouped={}
    with _excel_lock:
        for record_id,created,checksum,part,payload in _excel_book(path)['Session_Records'].iter_rows(min_row=2,values_only=True):
            if not record_id:continue
            entry=grouped.setdefault(record_id,dict(id=record_id,created=created,checksum=checksum,chunks={}))
            if checksum!=entry['checksum'] or created!=entry['created'] or part in entry['chunks']:raise ValueError('Audit record integrity check failed')
            entry['chunks'][part]=payload or ''
    result=[]
    for entry in grouped.values():
        chunks=entry['chunks']
        if sorted(chunks)!=list(range(len(chunks))):raise ValueError('Audit record is incomplete')
        result.append((entry['id'],entry['created'],''.join(chunks[i] for i in range(len(chunks))),entry['checksum']))
    return sorted(result,key=lambda r:r[1])

def _append_excel_record(book,record_id,created,payload,checksum):
    text=json.dumps(payload,ensure_ascii=False,allow_nan=False)
    ws=book['Session_Records']
    for part,start in enumerate(range(0,len(text),16000)):_text_row(ws,[record_id,created,checksum,part,text[start:start+16000]])
    context=payload.get('layer_0',{}).get('context',{})
    for p in payload.get('layer_2_processed',{}).get('processed',[]):
        _text_row(book['Session_Measurements'],[record_id,created,context.get('astronaut_id'),context.get('crew_name'),context.get('sex'),context.get('mission_role'),context.get('simulation',False),p.get('metric'),p.get('value'),p.get('unit'),p.get('method_id'),p.get('gate')])

def save(path: Path,payload: dict) -> dict:
    record_id=str(uuid4());created=datetime.now(timezone.utc).isoformat();checksum=digest(payload)
    with _excel_lock:
        book=_excel_book(path);_append_excel_record(book,record_id,created,payload,checksum);_commit_excel(path,book)
    return dict(record_id=record_id,created_utc=created,sha256=checksum,storage='Excel workbook')

def read(path: Path,record_id: str) -> dict | None:
    for rid,created,text,checksum in _records(path):
        if rid!=record_id:continue
        payload=json.loads(text)
        if digest(payload)!=checksum:raise ValueError('Audit record integrity check failed')
        return dict(record_id=rid,created_utc=created,sha256=checksum,layers=payload,integrity_verified=True)
    return None

def crew_session(path: Path,crew_id) -> dict[str,Any] | None:
    person=next((p for p in list_crew(path) if p['id']==crew_id),None)
    if person is None:return None
    real=[];simulated=[]
    for rid,created,text,checksum in reversed(_records(path)):
        data=json.loads(text)
        if digest(data)!=checksum:continue
        context=data.get('layer_0',{}).get('context',{})
        if context.get('astronaut_id')!=crew_id:continue
        synthetic=context.get('simulation') is True or any(p.get('gate')=='SIMULATED' for p in data.get('layer_2_processed',{}).get('processed',[]))
        (simulated if synthetic else real).append(rid)
    selected=simulated if person['is_demo'] and simulated else real if real else simulated
    return dict(crew=person,latest=read(path,selected[0]) if selected else None,saved_sessions=len(real)+len(simulated),real_sessions=len(real),simulated_sessions=len(simulated))

def baseline_candidates(path: Path,astronaut_id: str):
    groups={};seen=set()
    for _,_,text,checksum in _records(path)[:10000]:
        item=json.loads(text)
        if digest(item)!=checksum:continue
        if item.get('layer_0',{}).get('context',{}).get('astronaut_id')!=astronaut_id:continue
        for p in item.get('layer_2_processed',{}).get('processed',[]):
            if p.get('gate') not in ('ACCEPTED','REFERENCE'):continue
            identity=(p['metric'],p.get('timestamp_utc'),p.get('method_id'))
            if identity in seen:continue
            seen.add(identity);groups.setdefault(p['metric'],[]).append(p)
    return groups

record_store = SimpleNamespace(save=save,read=read,baseline_candidates=baseline_candidates,list_crew=list_crew,register_crew=register_crew,crew_session=crew_session)


# --- Health report ---

def reference_summary(processed):
    intervals={"Hemoglobin (g/dl)":(12,18),"Sodium (mmol/L)":(135,145),"Heart Rate (Bpm)":(50,110),"Oxygen saturation (%)":(94,100),"Calcium (mg/dL)":(8.5,10.5),"Leukocytes (k/uL)":(4,11),"Vitamin D (ng/mL)":(20,80),"alkaline phosphatase (U/L)":(35,135),"Calcium excretion (mg/24hr)":(40,300)}
    readings=[]
    for p in processed:
        if p["metric"] not in intervals or not p["usable_for_research"]:continue
        low,high=intervals[p["metric"]]
        readings.append(dict(label=p["label"],value=p["value"],unit=p["unit"],within_reference=low<=p["value"]<=high))
    within=sum(p["within_reference"] for p in readings)
    return dict(evaluated=len(readings),within_reference=within,outside_reference=len(readings)-within,readings=readings)
"""Provisional, rule-based health interpretations. No calibrated diagnosis model."""


def provisional_title(metric, value):
    if metric=="Oxygen saturation (%)":return "Possible low blood oxygen (hypoxemia)"
    if metric=="Hemoglobin (g/dl)":return "Possible anemia / low hemoglobin" if value<12 else "Raised hemoglobin concentration"
    if metric=="Sodium (mmol/L)":return "Possible low blood sodium (hyponatremia)" if value<135 else "Possible high blood sodium (hypernatremia)"
    if metric=="Calcium (mg/dL)":return "Low total serum calcium; confirmation needed" if value<8.5 else "High total serum calcium; confirmation needed"
    if metric=="Heart Rate (Bpm)":return "Low heart-rate reading; context needed" if value<50 else "Elevated heart-rate reading; context needed"
    if metric=="Leukocytes (k/uL)":return "White-cell count change; cause undetermined"
    if metric=="Vitamin D (ng/mL)":return "Possible low vitamin D status" if value<20 else "High vitamin D result; confirmation needed"
    if metric=="Calcium excretion (mg/24hr)":return "Altered urinary calcium excretion"
    if metric=="alkaline phosphatase (U/L)":return "Alkaline phosphatase change; cause undetermined"
    if metric=="Radiation per day(millirads)":return "Elevated absorbed-radiation reading"
    if metric in ("rnfl_od","rnfl_oi","Retinal & Optic Nerve Thickening(um) minimum rim width;"):return "Ocular measurement change; specialist review needed"
    if metric=="muscle_thickness":return "Possible local muscle-thickness change"
    if metric=="HRV (ms)":return "Recovery-related HRV change; cause undetermined"
    return "Change from your recorded reference"


def reliability(processed):
    """Transparent passed-check coverage; never a calibrated clinical probability."""
    totals=dict(passed=0,failed=0,unknown=0)
    per_metric={}
    for p in processed:
        failed=len(p["rejection_codes"])
        # Three automatic checks: finite value, method identity and method span.
        passed=3-sum(code in p["rejection_codes"] for code in ("METHOD_MISMATCH","OUTSIDE_METHOD_RANGE","UNSPECIFIED_MEASURAND_OR_METHOD"))
        passed+=sum(name in p.get("quality",{}) and "QUALITY_"+name.upper() not in p["rejection_codes"] for name in BY_KEY[p["metric"]]["quality_rules"])
        unknown=sum(name not in p.get("quality",{}) for name in BY_KEY[p["metric"]]["quality_rules"]) if p["acquisition_mode"]=="synthetic" else len(p["missing_evidence"])+(1 if p.get("uncertainty_budget") else 0)
        if p["gate"]=="ACCEPTED":passed+=3
        denominator=passed+failed+unknown
        pct=round(100*passed/denominator,1) if denominator else 0
        per_metric[p["metric"]]=dict(percentage=pct,passed=passed,failed=failed,unknown=unknown)
        totals["passed"]+=passed;totals["failed"]+=failed;totals["unknown"]+=unknown
    total=sum(totals.values())
    return dict(percentage=round(100*totals["passed"]/total,1) if total else 0,checks=totals,per_metric=per_metric,
                label="Data checks",definition="Supported numerical, method and signal-quality checks. Physical calibration is not applicable to synthetic generation. Passing these checks does not establish diagnostic accuracy or a probability of being healthy.",
                calibrated=False,simulated=any(p["gate"]=="SIMULATED" for p in processed))


def build(assessment):
    processed=assessment["metrology"]["processed"]
    by_key={p["metric"]:p for p in processed}
    conditions=[];steps=[]
    def action(text):
        if text and text not in steps:steps.append(text)
    for flag in assessment["flags"]:
        p=by_key.get(flag["metric"])
        if not p or not p["usable_for_research"]:continue
        conditions.append(dict(title=provisional_title(flag["metric"],flag["value"]),
                               status="PROVISIONAL",confirmed=False,diagnostic_confidence_pct=None,
                               urgency=flag["level"],metric=p["metric"],value=p["value"],unit=p["unit"],
                               expanded_uncertainty=p["expanded_uncertainty"],coverage_k=p["coverage_k"],
                               evidence=flag["finding"],recommended_action=flag["action"],
                               uncertainty_basis=p["gate"],
                               requires="Clinical context, confirmation of the measurement and review by the mission medical team"))
        action(flag["action"])
    rejected=[p for p in processed if p["gate"]=="REJECTED"]
    if rejected:
        action("Repeat the readings marked for verification using the approved onboard measurement procedure. A rejected reading cannot establish a diagnosis.")
    if not conditions:
        action("Continue the assigned monitoring and countermeasure plan. Report new or worsening symptoms to the crew medical officer even if no rule is triggered.")
    conclusion="Clinical review is needed for the possible conditions below." if conditions else "No monitored abnormality detected in the assessed reference intervals; clinical health status is not established by this prototype."
    if assessment["status"]=="URGENT":
        conclusion="A reading needs urgent confirmation and contact with the crew medical officer."
    return dict(conclusion=conclusion,possible_conditions=conditions,next_steps=steps,
                reliability=reliability(processed),reference_summary=reference_summary(processed),
                diagnostic_confidence_pct=None,diagnostic_confidence_status="NOT_CALIBRATED",
                confidence_explanation="A reliable diagnostic percentage cannot be calculated from synthetic training data and unvalidated clinical rules. Clinical confirmation is required.",
                confirmed_diagnosis=False,simulated=any(p["gate"]=="SIMULATED" for p in processed),
                scope="Provisional decision support. Measurement uncertainty is not diagnostic probability.")

health_report = SimpleNamespace(build=build)

# --- Embedded astronaut interface ---
HTML_PAGE = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>MSHIMAS | Crew report</title>\n<style>\n:root{color-scheme:light;--ink:#171717;--muted:#777;--line:#e6e6e6;--alert:#a32922}*{box-sizing:border-box}html{width:100%;min-height:100%;background:#fff}body{width:100%;min-height:100vh;margin:0;background:#fff;color:var(--ink);font:14px/1.5 Arial,Helvetica,sans-serif}button,input,select{font:inherit;border-radius:0}button,input,select{border:1px solid #cfcfcf;background:#fff;color:var(--ink);padding:7px 9px;min-width:0}button{cursor:pointer}button:hover{background:#f4f4f4}button:disabled{opacity:.4;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #555;outline-offset:2px}header,main{max-width:1440px;margin:auto;padding-inline:28px}header{padding-top:22px}.header-line{display:flex;justify-content:space-between;align-items:center;gap:20px;padding-bottom:18px;border-bottom:1px solid var(--line)}.brand{font-size:23px;font-weight:700;letter-spacing:-.8px}.brand small{font-size:10px;letter-spacing:0;font-weight:normal;color:var(--muted);margin-left:10px}.crew-controls{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.crew-controls label{font-size:11px;color:var(--muted)}#crewName{width:170px}#crewList{width:180px}.controls{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:12px 0}.simulation-controls{display:flex;align-items:center;gap:7px}.simulation-controls button,.simulation-controls select{font-size:12px;padding:6px 9px}.session-line{font-size:11px;color:var(--muted);display:flex;gap:14px;align-items:center}.session-line time{font-variant-numeric:tabular-nums}#message{font-size:11px;color:var(--alert);min-height:18px;margin:0}.report{display:grid;grid-template-columns:minmax(260px,.8fr) minmax(0,1.8fr);gap:55px;padding:24px 0 32px;border-top:1px solid var(--line)}.assessment{position:sticky;top:20px;align-self:start}.status{font-size:26px;font-weight:normal;line-height:1.25;margin:0 0 6px}.status.urgent{color:var(--alert)}.person{font-size:12px;color:var(--muted);margin:0 0 25px}.score{font:33px/1.1 Consolas,monospace;margin:0 0 3px}.score-label{font-size:10px;color:var(--muted);margin:0 0 28px}h2{font-size:12px;font-weight:600;margin:0 0 16px}.recommendation{padding-bottom:18px}.recommendation h3{font-size:15px;font-weight:normal;line-height:1.4;margin:0 0 7px}.recommendation ol{padding-left:18px;margin:0;font-size:13px;line-height:1.65}.recommendation li{margin-bottom:5px}.recommendation p{font-size:13px;line-height:1.7;margin:0}.recommendation.urgent h3{color:var(--alert)}.measure-title{display:flex;justify-content:space-between;align-items:center;gap:15px}.measure-title h2{margin:0}.measure-title span{font-size:10px;color:var(--muted)}.reading-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:28px;margin-top:12px}.reading{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid #eeeeec;min-width:0}.reading-name{font-size:11px;line-height:1.4}.reading-value{font:13px Consolas,monospace;white-space:nowrap;flex-shrink:0}.reading.urgent .reading-value{color:var(--alert);font-weight:bold}.unit{font:10px Arial,sans-serif;color:var(--muted);margin-left:4px}.reading.rejected .reading-value{color:var(--alert)}.empty{font-size:12px;color:var(--muted)}\n@media(max-width:1000px){.report{grid-template-columns:280px minmax(0,1fr);gap:30px}.reading-grid{grid-template-columns:1fr}.crew-controls{justify-content:flex-end}.header-line{align-items:start}}\n@media(max-width:650px){header,main{padding-inline:16px}header{padding-top:16px}.header-line{flex-direction:column;gap:12px}.brand{font-size:22px}.crew-controls{justify-content:flex-start;width:100%;gap:6px}#crewName{width:135px}#crewList{width:145px}.crew-controls button{font-size:12px}.crew-controls label{display:none}.controls{align-items:start;gap:8px;flex-direction:column}.session-line{width:100%;justify-content:space-between}.report{grid-template-columns:1fr;gap:25px;padding-top:20px}.assessment{position:static}.status{font-size:23px}.person{margin-bottom:17px}.score{font-size:28px}.score-label{margin-bottom:18px}.reading-grid{grid-template-columns:1fr}.recommendation{padding-bottom:12px}.reading-name{font-size:12px}.reading-value{font-size:14px}}\n#conditions{display:flex;flex-wrap:wrap;gap:7px 19px;padding:3px 0 13px}#conditions label{display:flex;align-items:center;gap:6px;font-size:11px;cursor:pointer;white-space:nowrap}#conditions input{appearance:auto;accent-color:#303030;width:13px;height:13px;margin:0;padding:0} @media(max-width:650px){#conditions{gap:9px 15px}#conditions label{font-size:12px}}\n.condition-reading{font:13px Consolas,monospace;white-space:nowrap}\n</style>\n</head>\n<body>\n<header>\n <div class="header-line"><div class="brand" title="Monitoring System for Highly Important Metrics for Astronaut Survival">MSHIMAS<small>Simulation</small></div><div class="crew-controls"><label for="crewName">Crew</label><input id="crewName" list="crewNames" maxlength="100" placeholder="Crew name" autocomplete="off"><datalist id="crewNames"></datalist><select id="crewList" aria-label="Registered crew"><option value="">Registered crew</option></select><button id="saveCrew">Save</button></div></div>\n <div class="controls"><div class="simulation-controls"><button id="pause">Pause</button><button id="restart">Restart</button></div><div class="session-line"><span id="phase">Connecting</span><span id="elapsed">00:00</span><time id="updated">--:--:--</time></div></div>\n <div id="conditions" role="group" aria-label="Simulation conditions"><label><input type="checkbox" value="auto" checked>Automatic changes</label><label><input type="checkbox" value="low_oxygen">Low oxygen</label><label><input type="checkbox" value="fast_pulse" data-conflict="pulse">Fast pulse</label><label><input type="checkbox" value="slow_pulse" data-conflict="pulse">Slow pulse</label><label><input type="checkbox" value="low_hemoglobin">Low hemoglobin</label><label><input type="checkbox" value="high_sodium" data-conflict="sodium">High sodium</label><label><input type="checkbox" value="low_sodium" data-conflict="sodium">Low sodium</label><label><input type="checkbox" value="low_vitamin_d">Low vitamin D</label><label><input type="checkbox" value="high_calcium" data-conflict="calcium">High calcium</label><label><input type="checkbox" value="low_calcium" data-conflict="calcium">Low calcium</label><label><input type="checkbox" value="high_leukocytes">High white-cell count</label><label><input type="checkbox" value="high_radiation">High radiation</label><label><input type="checkbox" value="sensor_fault">Sensor fault</label></div>\n <p id="message" role="alert"></p>\n</header>\n<main>\n <article class="report">\n  <div class="assessment"><h1 class="status" id="status">Awaiting readings</h1><p class="person" id="person"></p><div class="score" id="precision">--%</div><p class="score-label">Model precision · synthetic validation</p><h2>Recommendations</h2><div id="recommendations"><p class="empty">Starting simulation.</p></div></div>\n  <div class="measurements"><div class="measure-title"><h2>Measurements</h2><span id="readingCount"></span></div><div class="reading-grid" id="readings"></div></div>\n </article>\n</main>\n<script>\n\'use strict\';\nconst $=id=>document.getElementById(id);\nconst el=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n};\nconst state={profiles:[],crew:[],active:null,name:\'\',sex:\'F\',role:\'IVA\',observations:{},values:{},running:false,elapsedMs:0,startedAt:0,lastTick:0,timer:null,busy:false,generation:0,saveNext:false,pendingTick:false,result:null,recommendationKey:\'\',rows:new Map()};\nconst simulationProfile={\n \'Hemoglobin (g/dl)\':[14.2,.06,12.5,16.5], \'Hematocrit (%)\':[42,.2,38,47], \'Sodium (mmol/L)\':[140,.15,137,143],\n \'Heart Rate (Bpm)\':[64,2.4,56,78], \'HRV (ms)\':[54,2.8,40,68], \'Oxygen saturation (%)\':[98,.25,96.5,99.5],\n \'Calcium (mg/dL)\':[9.5,.03,9,10], \'Leukocytes (k/uL)\':[6.2,.06,5,8], \'Vitamin D (ng/mL)\':[32,.03,28,40],\n \'alkaline phosphatase (U/L)\':[76,.25,65,90], \'Calcium excretion (mg/24hr)\':[160,.4,120,220],\n \'Cabin Pressure/atm (kPa)\':[101.3,.008,101.2,101.4], \'muscle_thickness\':[25,.01,24.8,25.2],\n \'calcaneal_sos\':[1550,.2,1545,1555], \'calcaneal_bua\':[90,.08,88,92], \'rnfl_od\':[98,.02,97.5,98.5],\n \'sleep_tst\':[430,.2,420,440], \'pvt_speed\':[4.4,.015,4.2,4.6],\n \'Muscle Mass (kg)\':[28.8,.01,28,30], \'Aerobic Capacity ml⁄min/kg\':[33.3,.02,30,36],\n \'Work W\':[0,.03,0,.2], \'Radiation per day(millirads)\':[175,.5,170,180],\n \'Space Suit Pressure (kPa)\':[29.5,.01,29,30], \'Retinal & Optic Nerve Thickening(um) minimum rim width;\':[363,.1,360,366],\n \'Age\':[34,0,34,34], \'Height\':[170,0,170,170], \'ionized_calcium\':[1.2,.003,1.15,1.25],\n \'hs_crp\':[1.1,.01,.8,1.3], \'urine_specific_gravity\':[1.015,.00005,1.01,1.02],\n \'rnfl_oi\':[99,.02,98.5,99.5], \'functional_torque\':[140,.1,135,145], \'ppco2\':[3.1,.02,3,3.2],\n \'eva_energy\':[0,0,0,0], \'postural_sway\':[5,.05,4.5,5.5]\n};\nconst format=n=>Number(n).toLocaleString(\'en-US\',{maximumFractionDigits:3});\nfunction elapsedSeconds(){return (state.elapsedMs+(state.running?Date.now()-state.startedAt:0))/1000}\nfunction elapsedLabel(s){return String(Math.floor(s/60)).padStart(2,\'0\')+\':\'+String(Math.floor(s%60)).padStart(2,\'0\')}\nfunction automaticTargets(seconds){\n const t=seconds%180,mix=(a,b,f)=>a+(b-a)*Math.max(0,Math.min(1,f));\n if(t<20)return {phase:\'Rest\',hr:64,oxygen:98,hrv:54};\n if(t<60){const f=(t-20)/40;return {phase:\'Exertion\',hr:mix(64,115,f),oxygen:mix(98,96,f),hrv:mix(54,32,f)}}\n if(t<105){const f=(t-60)/45;return {phase:\'Deterioration\',hr:mix(115,155,f),oxygen:mix(96,87,f),hrv:mix(32,18,f)}}\n if(t<120)return {phase:\'Alert\',hr:155,oxygen:87,hrv:18};\n const f=(t-120)/60;return {phase:\'Recovery\',hr:mix(155,64,f),oxygen:mix(87,98,f),hrv:mix(18,54,f)};\n}\nasync function api(url,body){\n const retry=body===undefined||body.preview===true;\n for(let attempt=0;attempt<(retry?4:1);attempt++){\n  try{const response=await fetch(url,body===undefined?{cache:\'no-store\'}:{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||\'Request failed\');return data;}\n  catch(error){if(error instanceof TypeError&&retry&&attempt<3){$(\'message\').textContent=\'Reconnecting\';await new Promise(r=>setTimeout(r,1000));continue}throw new Error(error instanceof TypeError?\'Connection unavailable. Your profile is preserved.\':error.message)}\n }\n}\nfunction buildReadings(){\n const fragment=document.createDocumentFragment();state.rows.clear();\n state.profiles.filter(p=>Object.hasOwn(simulationProfile,p.key)).forEach(p=>{const row=el(\'div\',undefined,\'reading\');row.dataset.metric=p.key;const value=el(\'span\',\'--\',\'reading-value\'),number=el(\'span\',\'--\'),unit=el(\'span\',p.unit,\'unit\');value.replaceChildren(number,unit);row.append(el(\'span\',p.label,\'reading-name\'),value);fragment.appendChild(row);state.rows.set(p.key,{row,number})});\n $(\'readings\').replaceChildren(fragment);$(\'readingCount\').textContent=state.rows.size+\' readings\';\n}\nfunction quality(p){const q={};for(const [key,[op,b]] of Object.entries(p.quality_rules)){q[key]=op===\'eq\'?b:op===\'range\'?(b[0]+b[1])/2:op===\'min\'?b:op===\'max\'?0:Math.max(0,b/2)}return q;}\nfunction actionsFor(c){\n if(c.metric===\'Oxygen saturation (%)\')return [\'Stop exertion and secure yourself.\',\'Check sensor position and repeat the reading with a stable signal.\',\'Contact the crew medical officer now. Follow the approved oxygen procedure.\'];\n if(c.metric===\'Heart Rate (Bpm)\')return [\'Stop exertion, secure yourself and rest.\',\'Verify the pulse with the monitor or an alternate approved device.\',\'Contact the crew medical officer if it stays outside your baseline or you feel unwell.\'];\n if(c.metric===\'Hemoglobin (g/dl)\')return [\'Pause strenuous work pending medical review.\',\'Confirm the laboratory result and compare it with your flight baseline.\',\'Contact the crew medical officer. Use only the prescribed nutrition and supplement plan.\'];\n if(c.metric===\'Sodium (mmol/L)\')return [\'Confirm the result with the onboard laboratory procedure.\',\'Follow your assigned hydration plan. Do not change fluid or salt intake to correct this result yourself.\',\'Contact the crew medical officer\'+(c.urgency===\'URGENT\'?\' now.\':\'.\')];\n return [c.recommended_action||\'Repeat the measurement and contact the crew medical officer.\'];\n}\nfunction renderResult(data){\n state.result=data;$(\'message\').textContent=\'\';\n const names={NORMAL:\'No alert\',MONITOR:\'Monitor\',REVIEW:\'Review needed\',URGENT:\'Urgent review\'};\n $(\'status\').textContent=names[data.status]||data.status;$(\'status\').className=\'status \'+(data.status===\'URGENT\'?\'urgent\':\'\');\n $(\'person\').textContent=(data.crew_name||state.name||\'Anonymous\')+\' / \'+({F:\'Female\',M:\'Male\'}[state.sex]||\'Unspecified\')+\' / \'+state.role;\n $(\'precision\').textContent=data.model_validation?format(100*data.model_validation.precision)+\'%\':\'--%\';\n $(\'updated\').textContent=new Date(data.audit.created_utc).toLocaleTimeString();\n const flags=new Map(data.flags.map(f=>[f.metric,f]));\n data.metrology.processed.forEach(p=>{const entry=state.rows.get(p.metric);if(!entry)return;const text=p.gate===\'REJECTED\'?\'Check\':format(p.value);if(entry.number.textContent!==text)entry.number.textContent=text;entry.row.classList.toggle(\'urgent\',flags.get(p.metric)?.level===\'URGENT\');entry.row.classList.toggle(\'rejected\',p.gate===\'REJECTED\')});\n const conditions=data.health_assessment.possible_conditions;\n const rejected=data.metrology.processed.filter(p=>p.gate===\'REJECTED\');\n const key=JSON.stringify([conditions.map(c=>[c.title,c.urgency,c.recommended_action]),rejected.map(p=>p.metric)]);\n if(key!==state.recommendationKey){\n  state.recommendationKey=key;const fragment=document.createDocumentFragment();\n  if(!conditions.length&&!rejected.length)fragment.appendChild(el(\'p\',\'Continue scheduled monitoring.\'));\n  conditions.forEach(c=>{const block=el(\'div\',undefined,\'recommendation \'+(c.urgency===\'URGENT\'?\'urgent\':\'\')),steps=el(\'ol\');actionsFor(c).forEach(a=>steps.appendChild(el(\'li\',a)));const heading=el(\'h3\',c.title),reading=el(\'span\',undefined,\'condition-reading\');reading.dataset.readingMetric=c.metric;heading.appendChild(reading);block.append(heading,steps);fragment.appendChild(block)});\n  rejected.forEach(p=>{const block=el(\'div\',undefined,\'recommendation\'),steps=el(\'ol\');[\'Check sensor placement and signal quality.\',\'Repeat the measurement or use an approved backup device.\',\'Contact the crew medical officer if symptoms are present.\'].forEach(a=>steps.appendChild(el(\'li\',a)));const heading=el(\'h3\',p.label+\': repeat measurement\'),reading=el(\'span\',undefined,\'condition-reading\');reading.dataset.readingMetric=p.metric;heading.appendChild(reading);block.append(heading,steps);fragment.appendChild(block)});\n  $(\'recommendations\').replaceChildren(fragment);\n }\n document.querySelectorAll(\'#recommendations [data-reading-metric]\').forEach(node=>{\n  const p=data.metrology.processed.find(p=>p.metric===node.dataset.readingMetric);\n  node.textContent=p?\' / \'+(p.gate===\'REJECTED\'?\'Unverified: \':\'\')+(Number.isFinite(p.value)?format(p.value):\'--\')+\' \'+p.unit:\'\';\n });\n\n}\nasync function assessFrame(persist=false){\n if(state.busy){if(persist)state.saveNext=true;return}\n state.busy=true;const generation=state.generation;\n try{const data=await api(\'/api/assess\',{preview:!persist,observations:JSON.parse(JSON.stringify(state.observations)),context:{crew_name:state.name||null,astronaut_id:state.active?.id||null,sex:state.sex,mission_role:state.role,simulation:true,notes:\'Synthetic simulation.\',timestamp_utc:new Date().toISOString()}});if(generation!==state.generation)return;renderResult(data);if(persist){await refreshCrew(state.name);$(\'message\').textContent=\'Saved\';}}\n catch(error){if(generation===state.generation){$(\'message\').textContent=error.message;$(\'status\').textContent=\'Connection lost\';$(\'precision\').textContent=\'--%\'}}\n finally{state.busy=false;if(state.pendingTick){state.pendingTick=false;queueMicrotask(tickSimulation)}else if(state.saveNext){state.saveNext=false;queueMicrotask(()=>assessFrame(true))}}\n}\nfunction tickSimulation(){\n if(!state.running||state.busy)return;\n const now=Date.now(),dt=state.lastTick?Math.min(5,Math.max(.2,(now-state.lastTick)/1000)):1;state.lastTick=now;\n const seconds=elapsedSeconds(),cycle=automaticTargets(seconds),selected=new Set(Array.from(document.querySelectorAll(\'#conditions input:checked\'),n=>n.value)),alpha=1-Math.exp(-dt/4);\n const phase=Array.from(state.name).reduce((sum,c)=>sum+c.charCodeAt(0),0)/17;\n for(const [key,[center,amplitude]] of Object.entries(simulationProfile)){\n  let target=center;\n  if(selected.has(\'auto\')){if(key===\'Heart Rate (Bpm)\')target=cycle.hr;if(key===\'Oxygen saturation (%)\')target=cycle.oxygen;if(key===\'HRV (ms)\')target=cycle.hrv}\n  if(selected.has(\'low_oxygen\')){if(key===\'Oxygen saturation (%)\')target=87;if(key===\'Heart Rate (Bpm)\')target=135}\n  if(selected.has(\'low_hemoglobin\')&&key===\'Hemoglobin (g/dl)\')target=9.4;\n  if(selected.has(\'high_sodium\')&&key===\'Sodium (mmol/L)\')target=151;\n  if(selected.has(\'fast_pulse\')&&key===\'Heart Rate (Bpm)\')target=148;\n  if(selected.has(\'slow_pulse\')&&key===\'Heart Rate (Bpm)\')target=43;\n  if(selected.has(\'low_sodium\')&&key===\'Sodium (mmol/L)\')target=128;\n  if(selected.has(\'low_vitamin_d\')&&key===\'Vitamin D (ng/mL)\')target=10;\n  if(selected.has(\'high_calcium\')&&key===\'Calcium (mg/dL)\')target=12.5;\n  if(selected.has(\'low_calcium\')&&key===\'Calcium (mg/dL)\')target=7.2;\n  if(selected.has(\'high_leukocytes\')&&key===\'Leukocytes (k/uL)\')target=15;\n  if(selected.has(\'high_radiation\')&&key===\'Radiation per day(millirads)\')target=420;\n  const p=state.profiles.find(p=>p.key===key),previous=state.values[key]??center;\n  const noise=.25*amplitude*Math.sin(seconds/8+phase)+.12*amplitude*(Math.random()-.5);\n  const next=Math.max(p.minimum,Math.min(p.maximum,previous+(target+noise-previous)*alpha));state.values[key]=next;\n  const value=Number((Math.round(next/p.resolution)*p.resolution).toFixed(6));\n  state.observations[key]={value,unit:p.unit,method_id:p.method_id,acquisition_mode:\'synthetic\',quality:quality(p),timestamp_utc:new Date().toISOString()};\n  if(selected.has(\'sensor_fault\')&&key===\'Oxygen saturation (%)\')state.observations[key].quality.perfusion_index_pct=.1;\n }\n $(\'phase\').textContent=selected.has(\'auto\')?cycle.phase:\'Running\';$(\'elapsed\').textContent=elapsedLabel(seconds);assessFrame(state.saveNext);state.saveNext=false;\n}\nfunction start(){if(state.running)return;state.running=true;state.startedAt=Date.now();state.lastTick=Date.now();$(\'pause\').textContent=\'Pause\';tickSimulation();state.timer=setInterval(tickSimulation,1000);}\nfunction pause(){if(!state.running)return;state.elapsedMs+=Date.now()-state.startedAt;state.running=false;clearInterval(state.timer);state.timer=null;$(\'pause\').textContent=\'Resume\';$(\'phase\').textContent=\'Paused\';$(\'elapsed\').textContent=elapsedLabel(elapsedSeconds());}\nfunction restart(){const wasRunning=state.running;pause();state.elapsedMs=0;state.lastTick=0;state.values={};state.generation++;if(wasRunning)start();else{$(\'elapsed\').textContent=\'00:00\';}}\n$(\'pause\').addEventListener(\'click\',()=>state.running?pause():start());$(\'restart\').addEventListener(\'click\',restart);$(\'conditions\').addEventListener(\'change\',event=>{\n const input=event.target;if(input.checked&&input.dataset.conflict)document.querySelectorAll(\'#conditions input\').forEach(other=>{if(other!==input&&other.dataset.conflict===input.dataset.conflict)other.checked=false});\n localStorage.setItem(\'mshimasConditions\',JSON.stringify(Array.from(document.querySelectorAll(\'#conditions input:checked\'),n=>n.value)));\n state.generation++;if(state.busy)state.pendingTick=true;else if(state.running)tickSimulation();\n});\nasync function refreshCrew(selectedName=null){\n const data=await api(\'/api/crew\');state.crew=data.crew;$(\'crewNames\').replaceChildren();$(\'crewList\').replaceChildren(el(\'option\',\'Registered crew\'));$(\'crewList\').firstChild.value=\'\';\n state.crew.forEach(p=>{const option=el(\'option\',p.name);option.value=p.id;$(\'crewList\').appendChild(option);const suggestion=el(\'option\');suggestion.value=p.name;$(\'crewNames\').appendChild(suggestion)});\n const selected=state.crew.find(p=>selectedName?p.name.toLowerCase()===selectedName.toLowerCase():p.id===state.active?.id);if(selected){state.active=selected;$(\'crewList\').value=selected.id}\n}\nasync function loadCrew(id){\n pause();const generation=++state.generation;$(\'message\').textContent=\'\';\n try{const data=await api(\'/api/crew/\'+encodeURIComponent(id));if(generation!==state.generation)return;state.active=data.crew;state.name=data.crew.name;$(\'crewName\').value=state.name;$(\'crewList\').value=id;localStorage.setItem(\'mshimasCrew\',id);\n  const context=data.latest?.layers?.layer_0?.context||{};state.sex=context.sex&&context.sex!==\'Unknown\'?context.sex:state.name.endsWith(\'Beta\')?\'M\':\'F\';state.role=context.mission_role&&context.mission_role!==\'Unknown\'?context.mission_role:\'IVA\';\n  state.values={};state.observations={};state.elapsedMs=0;state.lastTick=0;state.result=null;state.recommendationKey=\'\';const raw=data.latest?.layers?.layer_1_raw||{};for(const key of Object.keys(simulationProfile)){if(Number.isFinite(raw[key]?.value))state.values[key]=raw[key].value}start();\n }catch(error){$(\'message\').textContent=error.message}\n}\n$(\'crewList\').addEventListener(\'change\',()=>{if($(\'crewList\').value)loadCrew($(\'crewList\').value)});\n$(\'crewName\').addEventListener(\'change\',()=>{const name=$(\'crewName\').value.trim(),match=state.crew.find(p=>p.name.toLowerCase()===name.toLowerCase());if(match&&match.id!==state.active?.id){loadCrew(match.id)}else if(name!==state.name){state.name=name;state.active=null;state.generation++;$(\'crewList\').value=\'\';}});\n$(\'saveCrew\').addEventListener(\'click\',async()=>{const name=$(\'crewName\').value.trim();if(!name){$(\'message\').textContent=\'Enter a crew name.\';return}try{const data=await api(\'/api/crew\',{name});state.active=data.crew;state.name=data.crew.name;await refreshCrew(state.name);assessFrame(true)}catch(error){$(\'message\').textContent=error.message}});\nwindow.addEventListener(\'beforeunload\',()=>{if(state.timer)clearInterval(state.timer)});\nasync function initialize(){\n try{const saved=JSON.parse(localStorage.getItem(\'mshimasConditions\')||\'null\');if(Array.isArray(saved))document.querySelectorAll(\'#conditions input\').forEach(input=>input.checked=saved.includes(input.value))}catch{}\n try{const catalog=await api(\'/api/catalog\');state.profiles=catalog.profiles;buildReadings();await refreshCrew();const saved=localStorage.getItem(\'mshimasCrew\'),person=state.crew.find(p=>p.id===saved)||state.crew.find(p=>p.is_demo)||state.crew[0];if(person)await loadCrew(person.id);else start();}\n catch(error){$(\'message\').textContent=error.message}\n}\ninitialize();\n</script>\n</body>\n</html>\n'

# --- Web API ---
"""Local server to connect the HTML interface with the ASTRA engine."""

from pathlib import Path
from threading import Lock
import math
from flask import Flask, jsonify, request, Response

BASE_DIR = Path(__file__).resolve().parent
DATABASE = _database_path()
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["AUDIT_DATABASE"] = DATABASE

_db: Any = None
_bundle: dict[str, Any] | None = None
_initialization_lock = Lock()


def initialize():
    """Load the database and train the detector in memory a single time."""
    with _initialization_lock:
        _initialize_locked()


def _initialize_locked():
    global _db, _bundle
    if _bundle is not None:
        return
    if not DATABASE.exists():
        raise FileNotFoundError(f"Could not find {DATABASE.name} next to MSHIMAS.py")
    _db = astra.load_database(DATABASE)
    # train_and_evaluate prints evaluation metrics on the synthetic holdout.
    # The evaluation does not equate to clinical validation.
    _bundle = astra.train_and_evaluate(_db)


@app.get("/")
def home():
    return Response(HTML_PAGE, mimetype="text/html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/status")
def status():
    return jsonify({
        "ready": True,
        "build": "standalone-20261115",
        "legacy_model_ready": _bundle is not None,
        "database_found": DATABASE.exists(),
        "engine": "MSHIMAS reference metrology and deterministic screening",
        "system_id": metrology.SYSTEM_ID,
        "system_name": metrology.SYSTEM_NAME,
        "metrology_policy": metrology.POLICY_VERSION,
        "sensor_connection": "Manual / JSON import; no hardware connected",
        "clinical_validation": False,
    })


@app.get("/api/metrics")
def metrics():
    try:
        initialize()
        dictionary = _db["Dictionary_24_metrics"]
        rows = []
        for _, row in dictionary.iterrows():
            rows.append({
                "name": str(row.iloc[0]),
                "type": str(row.iloc[1]),
                "unit": str(row.iloc[2]),
                "caution": str(row.iloc[4]),
            })
        return jsonify(rows)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/catalog")
def catalog():
    return jsonify({"system_id":metrology.SYSTEM_ID,"system_name":metrology.SYSTEM_NAME,
                    "profiles":metrology.CATALOG,"processors":metrology.PROCESSOR_EXAMPLES,
                    "source":metrology.SOURCE,"source_sha256":metrology.SOURCE_SHA256,"policy_version":metrology.POLICY_VERSION})


@app.route("/api/crew",methods=["GET","POST"])
def crew():
    path=Path(app.config["AUDIT_DATABASE"])
    if request.method=="GET":return jsonify(crew=record_store.list_crew(path))
    body=request.get_json(silent=True)
    try:
        if not isinstance(body,dict):raise ValueError("A crew name is required")
        return jsonify(crew=record_store.register_crew(path,body.get("name")))
    except ValueError as exc:return jsonify(error=str(exc)),400


@app.get("/api/crew/<crew_id>")
def crew_details(crew_id):
    result=record_store.crew_session(Path(app.config["AUDIT_DATABASE"]),crew_id)
    return jsonify(result) if result else (jsonify(error="Crew member not found"),404)


@app.post("/api/process")
def process():
    body=request.get_json(silent=True)
    if not isinstance(body,dict):return jsonify(error="A JSON object is required"),400
    try:
        result=metrology.calculate(body.get("processor"),body.get("inputs"))
        audit=record_store.save(Path(app.config["AUDIT_DATABASE"]),{
            "layer_0":{"policy":metrology.POLICY_VERSION,"source_sha256":metrology.SOURCE_SHA256},
            "layer_1_raw":body,"layer_2_processed":result,
            "layer_3_inference":{"state":"NOT_PERFORMED"}})
        return jsonify(result=result,audit=audit)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return jsonify(error=str(exc)),400


@app.get("/api/records/<record_id>")
def record(record_id):
    try:
        result=record_store.read(Path(app.config["AUDIT_DATABASE"]),record_id)
        if result is None:return jsonify(error="Record not found"),404
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)),409


def assess_observations(body):
    context=body.get("context") or {}
    if not isinstance(context,dict):raise ValueError("context must be an object")
    if context.get("crew_name"):
        person_record=record_store.register_crew(Path(app.config["AUDIT_DATABASE"]),context["crew_name"])
        context={**context,"astronaut_id":person_record["id"],"crew_name":person_record["name"]}
        body={**body,"context":context}
    acquired=metrology.acquire(body["observations"],body.get("context"))
    context=body.get("context") or {}
    person=context.get("astronaut_id")
    if person is not None and (not isinstance(person,str) or len(person)>100):raise ValueError("astronaut_id must be a string of at most 100 characters")
    # Anonymous sessions never share a subject baseline. No extra user form is
    # required: longitudinal comparison becomes available if an identifier exists.
    earlier=record_store.baseline_candidates(Path(app.config["AUDIT_DATABASE"]),person) if person else {}
    for p in acquired["processed"]:
        if p.get("baseline_change") or not p["usable_for_research"] or p["gate"]=="SIMULATED":continue
        series=[x for x in earlier.get(p["metric"],[]) if x["method_id"]==p["method_id"] and x["unit"]==p["unit"] and
                metrology.utc(x["timestamp_utc"])<metrology.utc(p["timestamp_utc"])]
        if len(series)>=3:
            base=series[:5];values=[x["value"] for x in base]
            baseline=dict(mean=float(metrology.np.mean(values)),sd=float(metrology.np.std(values,ddof=1)),
                          standard_uncertainty=max(x["standard_uncertainty"] for x in base),
                          n=len(base),method_id=p["method_id"],unit=p["unit"])
            p["baseline_change"]=metrology.baseline_change(p["value"],p["standard_uncertainty"],baseline,p["method_id"],p["unit"],p["coverage_k"])
            p["baseline_change"]["source"]="First 3-5 earlier local observations for this identifier; session reference, not a certified preflight baseline"
    eligible={p["metric"]:p["value"] for p in acquired["processed"] if p["usable_for_research"]}
    # Preserve safety screening as research-only rules, separately from ML. The
    # legacy forest has no validated mapping for the new document's measurands.
    # Vehicle-specific cabin/suit setpoints are not inferred from an ISS example.
    original={key:eligible.get(key) for key in astra.METRICS}
    original["Sex"]=context.get("sex","Unknown")
    original["EVA profile | Role"]=context.get("mission_role","Unknown")
    original["Cabin Pressure/atm (kPa)"]=None
    original["Space Suit Pressure (kPa)"]=None
    flags=astra.screening_rules(original)
    for p in acquired["processed"]:
        change=p.get("baseline_change")
        if change and change.get("level") in ("MONITOR","REVIEW"):
            flags.append(dict(metric=p["metric"],value=p["value"],level=change["level"],
                              finding=f"Change exceeds the combined measurement uncertainty; individual Z = {change['z_individual']:.2f}.",
                              action="Confirm acquisition and baseline comparability; request review by the mission medical team."))
    counts=acquired["counts"]
    acquired["derived_indicators"]=metrology.derive_indicators(acquired["processed"])
    quality=[f"{p['label']}: {p['gate']} / "+", ".join(p["rejection_codes"]+p["missing_evidence"])
             for p in acquired["processed"] if not p["usable_for_research"]]
    level=max([astra.LEVELS[f["level"]] for f in flags]+[2 if quality or not eligible else 0])
    status=next(k for k,v in astra.LEVELS.items() if v==level)
    inference={"state":"WITHHELD","reason":"The legacy synthetic model has no validated instrument-specific mapping or subject-level longitudinal validation for this schema.",
               "prediction_horizons":[],"clinical_validation":False}
    result=dict(system_id=metrology.SYSTEM_ID,status=status,metrology=acquired,
                flags=flags,quality_issues=quality,inference=inference,
                ml_withheld=True,ml_anomaly=None,anomaly_margin=None,
                numeric_measurements_available=len(eligible),numeric_measurements_total=len(metrology.CATALOG),
                notice="Research prototype. Instrument records are operator declarations. No diagnosis, treatment prescription or mission clearance is produced.")
    result["health_assessment"]=health_report.build(result)
    result["model_validation"]=get_model_validation()
    layers={"layer_0":{"policy":metrology.POLICY_VERSION,"source_sha256":metrology.SOURCE_SHA256,"context":body.get("context",{})},
            "layer_1_raw":body["observations"],"layer_2_processed":acquired,
            "layer_3_inference":{**inference,"health_assessment":result["health_assessment"],"model_validation":result["model_validation"]}}
    preview=body.get("preview",False)
    if not isinstance(preview,bool):raise ValueError("preview must be boolean")
    if preview:
        if not acquired["processed"] or any(p["acquisition_mode"]!="synthetic" for p in acquired["processed"]):raise ValueError("Preview is available only for synthetic simulation")
        result["audit"]=dict(record_id=None,created_utc=metrology.datetime.now(metrology.timezone.utc).isoformat(),persisted=False)
    else:
        result["audit"]=record_store.save(Path(app.config["AUDIT_DATABASE"]),layers)
        result["audit"]["persisted"]=True
    result["crew_name"]=context.get("crew_name")
    result["crew_context"]={"sex":context.get("sex","Unknown"),"mission_role":context.get("mission_role","Unknown")}
    return jsonify(result)


@app.post("/api/assess")
def assess():
    try:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "Send a JSON object with measurements."}), 400
        if "observations" in body:
            if "measurements" in body:raise ValueError("Do not mix instrument observations and legacy model measurements")
            return assess_observations(body)
        initialize()
        submitted = body.get("measurements", {})
        if not isinstance(submitted, dict):
            return jsonify({"error": "The measurements field must be a JSON object."}), 400
        extra = set(submitted) - set(astra.METRICS)
        if extra:
            return jsonify({"error": f"Unknown measurement names: {', '.join(sorted(extra))}"}), 400

        # The engine requires exactly the original names of the 24 variables.
        record = {}
        for name in astra.METRICS:
            value = submitted.get(name)
            if name in astra.CAT:
                if value is not None and not isinstance(value, str):
                    return jsonify({"error": f"Invalid category for: {name}"}), 400
                record[name] = str(value).strip() if value not in (None, "") else "Unknown"
            elif value in (None, ""):
                record[name] = None
            else:
                try:
                    if isinstance(value, (bool, list, dict)):
                        raise ValueError("Not a numeric scalar")
                    record[name] = float(str(value).replace(",", "."))
                    if not math.isfinite(record[name]) or abs(record[name]) > 1e12:
                        raise ValueError("Nonfinite or excessive number")
                except (TypeError, ValueError):
                    return jsonify({"error": f"Invalid numeric value for: {name}"}), 400

        result = astra.assess(_bundle, record, history=None)
        result["input_provenance"]="Legacy feature-only request; instrument qualification was not performed"
        return jsonify(result)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Error in ASTRA")
        return jsonify({"error": f"Unable to complete the analysis: {exc}"}), 500




_model_validation_cache: dict[str, Any] | None = None

def get_model_validation() -> dict[str, Any]:
    global _model_validation_cache
    if _model_validation_cache is not None:return _model_validation_cache
    initialize()
    if _bundle is None:raise RuntimeError("Legacy model initialization did not complete")
    indices=_bundle["test_indices"]
    ids=_db["Case_Metadata"].iloc[indices].Case_ID
    scenarios=_db["Case_Classifications"].set_index("Case_ID").reindex(ids)["Scenario"].tolist()
    report=evaluate_unseen(_bundle,pd.DataFrame(_db["X"].iloc[indices]),scenarios)
    _model_validation_cache={**report,"database_records":len(_db["Cases_24_metrics"]),"scope":"Legacy 24-feature statistical detector; synthetic holdout only, not clinical diagnostic calibration"}
    return _model_validation_cache

def ensure_demo_crews():
    """Complete synthetic profiles, never mistaken for real crew observations."""
    values={"Hemoglobin (g/dl)":14.2,"Hematocrit (%)":42,"Sodium (mmol/L)":140,
            "Heart Rate (Bpm)":64,"HRV (ms)":54,"Oxygen saturation (%)":98,
            "Calcium (mg/dL)":9.5,"Leukocytes (k/uL)":6.2,"Vitamin D (ng/mL)":32,
            "alkaline phosphatase (U/L)":76,"Calcium excretion (mg/24hr)":160,
            "Cabin Pressure/atm (kPa)":101.3,"muscle_thickness":25,
            "calcaneal_sos":1550,"calcaneal_bua":90,"rnfl_od":98,"sleep_tst":430,"pvt_speed":4.4}
    values.update({"Muscle Mass (kg)":28.8,"Aerobic Capacity ml⁄min/kg":33.3,"Work W":0,
                   "Radiation per day(millirads)":175,"Space Suit Pressure (kPa)":29.5,
                   "Retinal & Optic Nerve Thickening(um) minimum rim width;":363,"Age":34,"Height":170,
                   "ionized_calcium":1.2,"hs_crp":1.1,"urine_specific_gravity":1.015,"rnfl_oi":99,
                   "functional_torque":140,"ppco2":3.1,"eva_energy":0,"postural_sway":5})
    path=Path(app.config["AUDIT_DATABASE"])
    for name in sorted(DEMO_NAMES):
        person=register_crew(path,name)
        existing=crew_session(path,person["id"])
        raw=existing["latest"]["layers"]["layer_1_raw"] if existing and existing["latest"] else {}
        if len(raw)>=len(values) and all(raw.get(k,{}).get("acquisition_mode")=="synthetic" and set(BY_KEY[k]["quality_rules"]).issubset(raw.get(k,{}).get("quality",{})) for k in values):continue
        def generated_quality(p):
            result={}
            for field,(operator,bound) in p["quality_rules"].items():
                result[field]=bound if operator in ("eq","min") else (bound[0]+bound[1])/2 if operator=="range" else 0 if operator=="max" else max(0,bound/2)
            return result
        observations={key:dict(value=value,unit=BY_KEY[key]["unit"],method_id=BY_KEY[key]["method_id"],acquisition_mode="synthetic",quality=generated_quality(BY_KEY[key]),timestamp_utc=datetime.now(timezone.utc).isoformat()) for key,value in values.items()}
        with app.test_request_context():
            assess_observations({"observations":observations,"context":{"crew_name":name,"simulation":True,"sex":"M" if name.endswith("Beta") else "F","mission_role":"IVA","notes":"Built-in complete synthetic resting profile. Selecting this demo starts live simulation automatically."}})


def main():
    import os
    if "--terminal" in sys.argv:return legacy_main()
    port=int(os.environ.get("PORT", os.environ.get("MSHIMAS_PORT","5010")))
    ensure_demo_crews()
    if port==5010:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from threading import Thread
        class RedirectOldTestPort(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(307)
                self.send_header("Location","http://127.0.0.1:5010"+self.path)
                self.end_headers()
            def log_message(self, format: str, *args: Any):pass
        try:
            for old_port in (5000,5001):
                redirect=HTTPServer(("127.0.0.1",old_port),RedirectOldTestPort)
                Thread(target=redirect.serve_forever,daemon=True).start()
        except OSError:pass
    print(SYSTEM_ID+" - "+SYSTEM_NAME)
    print(f"Open http://127.0.0.1:{port}. Demo profiles start live simulation automatically.")
    app.run(host="0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1",port=port,debug=False,use_reloader=False)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
