"""Healthcare, operations only: beds, staff, admissions, waiting times, appointment capacity.

Silt does not make clinical decisions: nothing here estimates a diagnosis, a treatment effect or a patient outcome, and a
question that asks for one is classified unsupported. Patient-level columns that identify a person are never echoed back;
results are counts and averages."""
from __future__ import annotations

from .base import HYPOTHETICAL, Concept, Domain, InputSpec, Template

CONCEPTS = (
    Concept("patient", "Patient", frozenset({"patient", "patients", "mrn", "encounter"}), "id", 1.5),
    Concept("admission_date", "Admission date", frozenset({"admission", "admitted", "admit", "arrival", "visit"}), "datetime", 1.0),
    Concept("length_of_stay", "Length of stay", frozenset({"los", "stay", "length_of_stay"}), "numeric", 1.5),
    Concept("wait_time", "Waiting time", frozenset({"wait", "waiting", "wait_time"}), "numeric", 1.5),
    Concept("ward", "Ward / department", frozenset({"ward", "department", "unit", "clinic", "specialty"}), "category", 0.5),
    Concept("diagnosis", "Diagnosis", frozenset({"diagnosis", "icd", "condition", "treatment", "disease"}), "category", 1.0),
    Concept("doctor", "Doctor / staff", frozenset({"doctor", "physician", "nurse", "clinician", "staff"}), "category", 1.0),
    Concept("appointment", "Appointment", frozenset({"appointment", "appointments", "booking", "slot"}), "datetime", 1.0),
)

OCCUPANCY = {
    "inputs": {"admissions_per_day": {"label": "Admissions per day", "unit": "patients/day", "min": 0, "good": "down"},
               "avg_length_of_stay": {"label": "Average length of stay", "unit": "days", "min": 0, "good": "down"},
               "beds": {"label": "Beds available", "unit": "beds", "min": 0, "good": "up"}},
    "outputs": {"occupied_beds": {"label": "Beds needed (average)", "unit": "beds", "formula": "admissions_per_day * avg_length_of_stay"},
                "occupancy_pct": {"label": "Bed occupancy %", "unit": "percent", "formula": "admissions_per_day * avg_length_of_stay / beds * 100"},
                "bed_gap": {"label": "Beds short (negative = spare)", "unit": "beds", "formula": "admissions_per_day * avg_length_of_stay - beds"}},
    "primary_output": "occupancy_pct"}

BASE = (InputSpec("admissions_per_day", "Admissions per day", "patients/day", None, 0, None, from_data="rows_per_day:admission_date", role="baseline"),
        InputSpec("avg_length_of_stay", "Average length of stay (days)", "days", None, 0, None, from_data="mean:length_of_stay", role="baseline"),
        InputSpec("beds", "Beds available", "beds", None, 0, None, role="baseline", help="Not in the data; you must supply it"))

SURGE = Template(
    id="hc_admissions_surge", title="Admissions rise", question="What if admissions increase by {admissions_change}%?",
    method="monte_carlo", operation="formula_model", model=OCCUPANCY, manual=True, evidence=HYPOTHETICAL,
    inputs=(*BASE, InputSpec("admissions_change", "Change in admissions", "%", None, -100, 1000, role="change", target="admissions_per_day", change_type="pct")),
    params={"uncertain": ["admissions_per_day"]},
    reason="Admission demand is uncertain, so it can be run as a Monte Carlo simulation with a spread you state.",
    tags=frozenset({"admissions", "demand", "surge", "emergency", "double", "occupancy", "beds"}),
    patterns=(r"admissions?\s+(?:increase|rise|grow|double|surge)", r"(?:emergency|ed|er)\s+(?:department\s+)?demand", r"patient\s+(?:admissions|volume|load)"),
    limitation="Bed need = admissions per day × average stay (a steady-state average). It ignores day-of-week patterns and discharge delays.")

BEDS = Template(
    id="hc_add_beds", title="Add or remove beds", question="What if we add {extra_beds} beds?",
    method="deterministic", operation="formula_model", model=OCCUPANCY, manual=True,
    inputs=(*BASE, InputSpec("extra_beds", "Beds added (negative = removed)", "beds", None, None, None, role="change", target="beds", change_type="abs")),
    reason="Bed capacity is a simple balance between demand and beds; you provide the bed count.",
    tags=frozenset({"beds", "bed", "capacity", "occupancy", "ward"}),
    patterns=(r"(?:add|remove|open|close)\s+\d*\s*(?:more\s+)?(?:hospital\s+)?beds?", r"\bbed\s+capacity"),
    limitation="Steady-state average occupancy; peaks are not shown.")

STAFF_MODEL = {
    "inputs": {"appointments_per_day": {"label": "Appointments per day", "unit": "appointments/day", "min": 0},
               "doctors": {"label": "Doctors available", "unit": "doctors", "min": 0, "good": "up"},
               "slots_per_doctor": {"label": "Appointments one doctor can see per day", "unit": "appointments/doctors", "min": 0, "good": "up"}},
    "outputs": {"capacity": {"label": "Daily appointment capacity", "unit": "appointments", "formula": "doctors * slots_per_doctor"},
                "utilisation_pct": {"label": "Capacity used %", "unit": "percent", "formula": "appointments_per_day / (doctors * slots_per_doctor) * 100"},
                "unmet": {"label": "Appointments that do not fit", "unit": "appointments", "formula": "max(appointments_per_day - doctors * slots_per_doctor, 0)"}},
    "primary_output": "utilisation_pct"}

STAFF = Template(
    id="hc_staff_unavailable", title="Doctors unavailable", question="What if {doctor_change} doctors are unavailable?",
    method="deterministic", operation="formula_model", model=STAFF_MODEL, manual=True,
    inputs=(InputSpec("appointments_per_day", "Appointments per day", "appointments/day", None, 0, None, from_data="rows_per_day:appointment", role="baseline"),
            InputSpec("doctors", "Doctors available", "doctors", None, 0, None, role="baseline"),
            InputSpec("slots_per_doctor", "Appointments per doctor per day", "appointments/doctor", None, 0, None, role="baseline"),
            InputSpec("doctor_change", "Change in doctors (negative = fewer)", "doctors", None, None, None, role="change", target="doctors", change_type="abs")),
    reason="Appointment capacity is doctors × how many each can see; the doctor numbers are yours to supply.",
    tags=frozenset({"doctors", "staff", "unavailable", "appointments", "capacity", "leave"}),
    patterns=(r"(?:doctors?|nurses?|staff)\s+(?:\w+\s+)?(?:unavailable|absent|on leave|leave)", r"appointment\s+capacity"),
    limitation="A capacity balance; it does not model queueing or clinical priority.")

WAIT = Template(
    id="hc_wait_target", title="Change the waiting-time target", question="What if the waiting-time target changes from {current_threshold} to {new_threshold} minutes?",
    method="deterministic", operation="threshold", requires=("wait_time", "patient"),
    inputs=(InputSpec("current_threshold", "Current target (minutes)", "minutes", None, 0, None), InputSpec("new_threshold", "New target (minutes)", "minutes", None, 0, None)),
    params={"concept": "wait_time", "noun": "patients", "rule": "the waiting-time target", "direction": "le"},
    reason="Your data holds waiting times, so the share seen within each target is an exact count.",
    tags=frozenset({"wait", "waiting", "target", "minutes"}),
    patterns=(r"wait(?:ing)?[\s-]*time\s+target", r"wait(?:ing)?\s+(?:time|target)"),
    limitation="Counts past waits against the new target; it does not predict whether the new target is achievable.")

CLINICAL = Template(
    id="hc_treatment", title="Change a treatment", question="What if we change the treatment protocol?",
    method="deterministic", operation="none", unsupported="Silt does not estimate clinical outcomes or recommend treatments. It supports operational capacity questions only.",
    reason="Listed so it is clear what is out of scope.", tags=frozenset({"treatment", "protocol", "diagnosis", "drug", "therapy", "outcome"}),
    patterns=(r"treatment", r"diagnos", r"\btherapy\b", r"clinical"), limitation="Out of scope.")

HEALTHCARE = Domain(
    key="healthcare", title="Healthcare", description="Operational capacity: beds, staff, admissions, waiting times and appointments. Not clinical decision-making.",
    entities=("patient", "ward", "doctor"), concepts=CONCEPTS, templates=(SURGE, BEDS, STAFF, WAIT, CLINICAL),
    metrics=("bed occupancy", "waiting time", "appointment capacity"),
    keywords=frozenset({"hospital", "clinic", "medical", "icd"}),
    sensitive_tokens=frozenset({"name", "dob", "birth", "address", "phone", "email", "ssn", "mrn", "aadhaar", "insurance"}),
    notes=("Operational planning only: no clinical, diagnostic or treatment conclusion is drawn.",
           "Patient-level values are never shown; results are counts and averages."))
