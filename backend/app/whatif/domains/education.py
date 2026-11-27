"""Education: student performance, attendance eligibility, pass/fail, capacity and enrolment."""
from __future__ import annotations

from .base import ESTIMATE, EXACT, FORECAST, HYPOTHETICAL, Concept, Domain, InputSpec, Template

CONCEPTS = (
    Concept("student", "Student", frozenset({"student", "students", "roll", "learner", "pupil", "pupils", "enrolment", "enrollment", "admission"}), "id", 1.5),
    Concept("attendance", "Attendance", frozenset({"attendance", "attended", "present", "presence"}), "numeric", 1.5),
    Concept("marks", "Marks / scores", frozenset({"marks", "mark", "maths", "math", "mathematics", "science", "physics", "chemistry", "biology",
                                                  "english", "history", "exam", "exams", "grade", "grades", "gpa", "cgpa", "percentage", "result"}),
            "numeric", 1.0, multi=True, excludes=frozenset({"attendance", "attended", "presence", "completion", "assignment", "assignments", "study"})),
    Concept("study_hours", "Study hours", frozenset({"study", "studying", "homework", "revision"}), "numeric", 1.0),
    Concept("faculty", "Faculty", frozenset({"faculty", "teacher", "teachers", "lecturer", "instructor", "professor"}), "category", 1.0),
    Concept("classroom", "Class / section", frozenset({"class", "section", "classroom", "semester", "course", "subject", "department"}), "category", 0.5),
    Concept("date", "Enrolment / exam date", frozenset({"date", "enrolled", "admitted", "admission", "semester_start", "month"}), "datetime", 0.3),
)

PASS = Template(
    id="edu_pass_marks", title="Change the passing marks", question="What if the passing marks change from {current_threshold} to {new_threshold}?",
    method="deterministic", operation="threshold", requires=("marks", "student"),
    inputs=(InputSpec("current_threshold", "Current passing mark", "marks", None, 0, None, help="The mark at or above which a student passes today"),
            InputSpec("new_threshold", "New passing mark", "marks", None, 0, None)),
    params={"concept": "marks", "noun": "students", "rule": "the pass mark"},
    reason="Your data holds each student's marks, so the new pass/fail split can be counted exactly.",
    tags=frozenset({"pass", "passing", "fail", "marks", "threshold", "cutoff", "exam", "results"}),
    patterns=(r"pass(?:ing)?\s+(?:marks?|score|grade|percentage|threshold)", r"\bpass\s*mark", r"minimum\s+(?:passing\s+)?marks?"),
    limitation="Counts students against the new cut-off using their existing marks; it does not predict how students would perform if the cut-off were different.")

ATTENDANCE_RULE = Template(
    id="edu_attendance_requirement", title="Change the attendance requirement", question="What if the attendance requirement changes from {current_threshold}% to {new_threshold}%?",
    method="deterministic", operation="threshold", requires=("attendance", "student"),
    inputs=(InputSpec("current_threshold", "Current required attendance", "%", None, 0, 100),
            InputSpec("new_threshold", "New required attendance", "%", None, 0, 100)),
    params={"concept": "attendance", "noun": "students", "rule": "the attendance requirement", "percent_scale": True},
    reason="Your data holds each student's attendance, so the eligible and ineligible groups can be counted exactly.",
    tags=frozenset({"attendance", "eligibility", "eligible", "ineligible", "requirement", "detained"}),
    patterns=(r"attendance\s+(?:requirements?|criteria|threshold|cut-?off|rule|minimum)", r"(?:minimum|required)\s+attendance", r"attendance\s+(?:is\s+)?(?:required|mandatory)"),
    limitation="Uses attendance as recorded. Students may change their attendance if the rule changes; this does not model that.")

ATTENDANCE_UP = Template(
    id="edu_attendance_increase", title="Raise attendance", question="What if attendance increases by {change}%?",
    method="deterministic", operation="shift_metric", requires=("attendance", "student"),
    inputs=(InputSpec("change", "Change in every student's attendance", "%", None, -100, 100),
            InputSpec("threshold", "Attendance requirement to check against", "%", None, 0, 100, optional=True)),
    params={"concept": "attendance", "noun": "students", "mode": "pct", "cap": "percent_scale", "percent_scale": True, "unit": "%"},
    reason="Attendance is in your data; the new average and, with a requirement, the eligible share can be computed directly.",
    tags=frozenset({"attendance", "increase", "improve", "raise"}),
    patterns=(r"attendance\s+(?:increase|improve|rise|goes up|up)", r"(?:increase|improve|raise|boost)\w*\s+(?:the\s+)?attendance"),
    limitation="Applies the same percentage to every student. It does not say how attendance could actually be raised.")

STUDY = Template(
    id="edu_study_hours", title="Study more hours", question="What if students study {change} more hours a day?",
    method="regression", operation="association", requires=("study_hours", "marks"), evidence=ESTIMATE,
    inputs=(InputSpec("change", "Extra study hours per day", "hours", None, -24, 24),),
    params={"x": "study_hours", "y": "marks", "mode": "abs", "x_unit": "hours", "y_unit": "marks"},
    reason="Your data holds study hours and marks, so the historical relationship can be checked. If it is too weak, no effect is claimed.",
    tags=frozenset({"study", "hours", "marks", "improve", "performance", "homework"}),
    patterns=(r"stud(?:y|ies|ying)\s+(?:for\s+)?(?:\w+\s+)?(?:more|additional|extra)", r"(?:more|additional|extra)\s+(?:\w+\s+)?(?:hours?\s+)?(?:of\s+)?stud", r"study\s+hours"),
    limitation="An association in past data, not proof that studying more causes higher marks.")

ENROLMENT_MODEL = {
    "description": "Student numbers against faculty and seats",
    "inputs": {"students": {"label": "Students", "unit": "students", "min": 0, "good": "up"},
               "faculty": {"label": "Faculty members", "unit": "faculty", "min": 0, "good": "up"},
               "seats": {"label": "Seats available", "unit": "seats", "min": 0, "good": "up"}},
    "outputs": {"students_per_faculty": {"label": "Students per faculty member", "unit": "students/faculty", "formula": "students / faculty"},
                "seat_utilisation_pct": {"label": "Seat utilisation %", "unit": "percent", "formula": "students / seats * 100"},
                "seats_short": {"label": "Students without a seat", "unit": "students", "formula": "max(students - seats, 0)"}},
    "primary_output": "students_per_faculty"}

ENROL_PLUS = Template(
    id="edu_enrolment_capacity", title="Enrol more students", question="What if {extra_students} additional students enrol?",
    method="deterministic", operation="formula_model", model=ENROLMENT_MODEL, manual=True,
    inputs=(InputSpec("students", "Current students", "students", None, 0, None, from_data="entity_count", role="baseline"),
            InputSpec("faculty", "Faculty members available", "faculty", None, 0, None, role="baseline", help="Not in the data; you must supply it"),
            InputSpec("seats", "Seats available", "seats", None, 0, None, role="baseline", help="Not in the data; you must supply it"),
            InputSpec("extra_students", "Additional students", "students", None, 0, None, role="change", target="students", change_type="abs")),
    reason="Capacity depends on faculty and seats, which a student table does not hold. You provide them; the ratios are then exact.",
    tags=frozenset({"enrol", "enroll", "enrolment", "enrollment", "students", "capacity", "class", "size", "faculty", "seats", "classroom"}),
    patterns=(r"(?:additional|more|extra|new)\s+\d*\s*students?\s+(?:enrol|enroll|join|admit)", r"\d+\s+(?:additional|more|extra|new)\s+students", r"enrol(?:l)?ment\s+(?:increase|grow|rise)", r"class\s+size"),
    limitation="Compares totals with the capacity you state. It does not judge teaching quality or timetable feasibility.")

FACULTY_CHANGE = Template(
    id="edu_faculty_change", title="Change faculty numbers", question="What if the faculty headcount changes by {faculty_change}?",
    method="deterministic", operation="formula_model", model=ENROLMENT_MODEL, manual=True,
    inputs=(InputSpec("students", "Current students", "students", None, 0, None, from_data="entity_count", role="baseline"),
            InputSpec("faculty", "Current faculty members", "faculty", None, 0, None, role="baseline"),
            InputSpec("seats", "Seats available", "seats", None, 0, None, role="baseline"),
            InputSpec("faculty_change", "Change in faculty (negative = fewer)", "faculty", None, None, None, role="change", target="faculty", change_type="abs")),
    reason="Faculty capacity is a student-to-faculty ratio; the headcount is yours to supply.",
    tags=frozenset({"faculty", "teachers", "staff", "unavailable", "hire", "ratio"}),
    patterns=(r"(?:faculty|teachers?|lecturers?)\s+(?:\w+\s+)?(?:leave|unavailable|reduce|increase|hire|added)", r"(?:hire|lose|add)\s+\d*\s*(?:faculty|teachers?|lecturers?)"),
    limitation="A ratio calculation only.")

FORECAST = Template(
    id="edu_enrolment_forecast", title="Forecast enrolment", question="What if the enrolment trend continues for {periods_ahead} more period(s)?",
    method="forecast", operation="forecast_trend", requires=("date",), evidence=FORECAST,
    inputs=(InputSpec("periods_ahead", "Periods ahead", "months", 1, 1, 24),),
    params={"date": "date", "noun": "Records (enrolments) per month", "freq": "M"},
    reason="Your data has dates, so records per month can be projected, if there is enough history.",
    tags=frozenset({"enrolment", "enrollment", "next", "semester", "forecast", "future", "trend"}),
    patterns=(r"enrol(?:l)?ment\s+next", r"next\s+(?:semester|year|term)", r"forecast\s+enrol"),
    limitation="A straight-line trend; it ignores seasons, policy changes and capacity limits.")

EXTRA_EXAM = Template(
    id="edu_extra_exam", title="Introduce an additional examination", question="What if we introduce an additional examination?",
    method="deterministic", operation="none", external=("evidence on how an extra examination changes marks or dropout",),
    reason="A common planning question, but no dataset of past marks can say how students would respond to a new examination.",
    tags=frozenset({"examination", "exam", "additional", "introduce", "test"}),
    patterns=(r"additional\s+exam", r"extra\s+exam", r"new\s+exam"),
    limitation="Needs outside evidence, for example results from a school that already did this.")

EDUCATION = Domain(
    key="education", title="Education", description="Student performance, attendance rules, pass/fail thresholds, class capacity and enrolment.",
    entities=("student", "class", "faculty"), concepts=CONCEPTS,
    templates=(PASS, ATTENDANCE_RULE, ATTENDANCE_UP, STUDY, ENROL_PLUS, FACULTY_CHANGE, FORECAST, EXTRA_EXAM),
    metrics=("pass rate", "eligible students", "average marks", "students per faculty", "seat utilisation"),
    keywords=frozenset({"school", "college", "university", "academic", "gpa"}),
    notes=("Student-level data is personal data: results show counts and averages, not individual students.",))
