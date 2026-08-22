"""
Input schema for the Employee Salary predictor.

Only the 7 fields in SELECTED_FEATURES are shown to the user (see
grouped_fields() below) — these are the ones that actually reach the
estimator inside the pipeline. Every other field still exists in FIELDS
and still gets sent to the pipeline on every prediction, but as a fixed
`default` value rather than something the user can edit, since the
pipeline's preprocessing steps were fit on the full original schema and
need every column present to run.
"""

# features to show to the user for input; the rest are hidden but still sent to the pipeline
# SELECTED_FEATURES = [
#     "experience_years",
#     "country",
#     "job_role",
#     "employee_satisfaction",
#     "experience_level",
#     "work_mode",
#     "company_funding_billion"
# ]

COUNTRY_OPTIONS = ['Australia', 'Brazil', 'Canada', 'France', 'Germany', 'India', 'Japan', 'Netherlands', 'Singapore', 'UAE', 'UK', 'USA']
JOB_ROLE_OPTIONS = ['AI Engineer', 'Computer Vision Engineer', 'Data Analyst', 'Data Scientist', 'Machine Learning Engineer', 'NLP Engineer', 'Research Scientist', 'Software Engineer AI']
AI_SPECIALIZATION_OPTIONS = ['Analytics', 'Computer Vision', 'Forecasting', 'Generative AI', 'LLM', 'MLOps', 'NLP', 'Reinforcement Learning']
EXPERIENCE_LEVEL_OPTIONS = ['Entry', 'Lead', 'Mid', 'Senior']
EDUCATION_REQUIRED_OPTIONS = ['Bachelor', 'Bootcamp', 'Diploma', 'Master', 'PhD']
INDUSTRY_OPTIONS = ['Automotive', 'Consulting', 'Education', 'Energy', 'Finance', 'Gaming', 'Healthcare', 'Retail', 'Tech', 'Telecom']
COMPANY_SIZE_OPTIONS = ['Enterprise', 'Large', 'Medium', 'Small', 'Startup']
WORK_MODE_OPTIONS = ['Hybrid', 'Onsite', 'Remote']

# Each field: name, label, kind ("select"|"number"), group, and either
# options+default (select) or min/max/step/default (number).
FIELDS = [
    # --- Role & Experience ---
    dict(name="country", label="Country", kind="select", group="Role & Experience",
         options=COUNTRY_OPTIONS, default=COUNTRY_OPTIONS[0]),
    dict(name="job_role", label="Job role", kind="select", group="Role & Experience",
         options=JOB_ROLE_OPTIONS, default=JOB_ROLE_OPTIONS[0]),
    dict(name="ai_specialization", label="AI specialization", kind="select", group="Role & Experience",
         options=AI_SPECIALIZATION_OPTIONS, default=AI_SPECIALIZATION_OPTIONS[0]),
    dict(name="experience_level", label="Experience level", kind="select", group="Role & Experience",
         options=EXPERIENCE_LEVEL_OPTIONS, default="Mid"),
    dict(name="experience_years", label="Years of experience", kind="number", group="Role & Experience",
         min=0, max=25, step=1, default=5),
    dict(name="education_required", label="Education", kind="select", group="Role & Experience",
         options=EDUCATION_REQUIRED_OPTIONS, default="Bachelor"),
    dict(name="work_mode", label="Work mode", kind="select", group="Role & Experience",
         options=WORK_MODE_OPTIONS, default="Hybrid"),
    dict(name="year", label="Year", kind="number", group="Role & Experience",
         min=2020, max=2026, step=1, default=2025),

    # --- Compensation & Company ---
    dict(name="bonus_usd", label="Annual bonus (USD)", kind="number", group="Compensation & Company",
         min=0, max=100000, step=100, default=12000),
    dict(name="industry", label="Industry", kind="select", group="Compensation & Company",
         options=INDUSTRY_OPTIONS, default="Tech"),
    dict(name="company_size", label="Company size", kind="select", group="Compensation & Company",
         options=COMPANY_SIZE_OPTIONS, default="Medium"),
    dict(name="company_rating", label="Company rating (1-5)", kind="number", group="Compensation & Company",
         min=1, max=5, step=0.1, default=4.0),
    dict(name="company_funding_billion", label="Company funding ($B)", kind="number", group="Compensation & Company",
         min=0, max=20, step=0.1, default=4.5),
    dict(name="job_openings", label="Similar job openings", kind="number", group="Compensation & Company",
         min=0, max=100, step=1, default=17),
    dict(name="interview_rounds", label="Interview rounds", kind="number", group="Compensation & Company",
         min=1, max=10, step=1, default=4),
    dict(name="weekly_hours", label="Weekly hours", kind="number", group="Compensation & Company",
         min=20, max=70, step=0.5, default=45),
    dict(name="vacation_days", label="Vacation days / year", kind="number", group="Compensation & Company",
         min=0, max=40, step=1, default=20),
    dict(name="tax_rate_percent", label="Effective tax rate (%)", kind="number", group="Compensation & Company",
         min=0, max=60, step=0.5, default=27),

    # --- Market Signals ---
    dict(name="economic_index", label="Economic index", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=72),
    dict(name="cost_of_living_index", label="Cost of living index", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=65),
    dict(name="offer_acceptance_rate", label="Offer acceptance rate (%)", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=75),
    dict(name="hiring_difficulty_score", label="Hiring difficulty score", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=55),
    dict(name="layoff_risk", label="Layoff risk (0-1)", kind="number", group="Market Signals",
         min=0, max=1, step=0.01, default=0.18),
    dict(name="ai_adoption_score", label="AI adoption score", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=71),
    dict(name="ai_maturity_years", label="Company AI maturity (years)", kind="number", group="Market Signals",
         min=0, max=20, step=1, default=8),
    dict(name="automation_risk", label="Automation risk score", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=50),
    dict(name="skill_demand_score", label="Skill demand score", kind="number", group="Market Signals",
         min=0, max=100, step=1, default=50),

    # --- Outlook Scores ---
    dict(name="job_security_score", label="Job security score", kind="number", group="Outlook Scores",
         min=0, max=100, step=1, default=76),
    dict(name="career_growth_score", label="Career growth score", kind="number", group="Outlook Scores",
         min=0, max=100, step=1, default=57),
    dict(name="work_life_balance_score", label="Work-life balance score", kind="number", group="Outlook Scores",
         min=0, max=100, step=1, default=69),
    dict(name="promotion_speed", label="Promotion speed score", kind="number", group="Outlook Scores",
         min=0, max=100, step=1, default=50),
    dict(name="salary_percentile", label="Salary percentile", kind="number", group="Outlook Scores",
         min=0, max=100, step=1, default=50),
    dict(name="employee_satisfaction", label="Employee satisfaction score", kind="number", group="Outlook Scores",
         min=0, max=100, step=1, default=70),
]

GROUP_ORDER = ["Role & Experience", "Compensation & Company", "Market Signals", "Outlook Scores"]


# def grouped_fields():
#     """Return only the fields that actually affect the prediction, grouped for display."""
#     out = {}
#     for g in GROUP_ORDER:
#         visible = [f for f in FIELDS if f["group"] == g and f["name"] in SELECTED_FEATURES]
#         if visible:
#             out[g] = visible
#     return out

def grouped_fields():
    """Return fields organised by group, in display order."""
    out = {g: [] for g in GROUP_ORDER}
    for f in FIELDS:
        out[f["group"]].append(f)
    return out
