LOAN_DOCUMENTS = [
    {
        "id": "loan_policy_001",
        "title": "Loan Approval Policy — Credit Score Requirements",
        "content": """LoanCo Approval Policy — Section 1: Credit Score Requirements

Credit scores are the primary factor in loan approval decisions.

Score bands and outcomes:
- 750 and above: Excellent. Auto-approved for standard loan amounts. Interest rate tier: Prime.
- 700 to 749: Good. Approved with standard review. Interest rate tier: Standard.
- 650 to 699: Fair. Conditional approval — requires additional income verification.
- 600 to 649: Poor. High scrutiny required. Approval only if debt-to-income ratio is below 30%.
- Below 600: Very poor. Application rejected unless supported by collateral equal to 150% of loan value.

Applicants with no credit history are treated as score band 620 and require a co-signer.
Credit score is sourced from EXT_SOURCE_2 and EXT_SOURCE_3 in our data pipeline."""
    },
    {
        "id": "loan_policy_002",
        "title": "Loan Approval Policy — Debt-to-Income Ratio",
        "content": """LoanCo Approval Policy — Section 2: Debt-to-Income (DTI) Ratio

DTI ratio = Total monthly debt payments divided by Gross monthly income.

DTI thresholds:
- Below 28%: Low risk. No additional review required.
- 28% to 36%: Moderate risk. Standard processing.
- 37% to 43%: High risk. Requires supervisor approval and income proof.
- Above 43%: Very high risk. Application rejected unless credit score is above 720 and employment tenure exceeds 5 years.

AMT_ANNUITY divided by income gives the annuity-to-income ratio which feeds DTI calculation.
Applicants with high DTI and unstable employment (DAYS_EMPLOYED less than 365) are auto-rejected."""
    },
    {
        "id": "loan_policy_003",
        "title": "Loan Approval Policy — Employment and Income",
        "content": """LoanCo Approval Policy — Section 3: Employment and Income Requirements

Employment stability is a key default risk factor.

Requirements:
- Minimum employment tenure: 6 months (182 days) for any loan.
- Employed applicants with tenure above 2 years receive a risk reduction bonus.
- Self-employed applicants must provide 2 years of tax returns.
- Unemployed applicants are rejected unless they have a co-applicant with stable income.

Income requirements:
- Loan amount must not exceed 5x annual gross income.
- AMT_CREDIT divided by AMT_INCOME_TOTAL ratio must be below 5.0.
- Pensioners and retirees are eligible if pension income meets the income threshold.

DAYS_EMPLOYED in our dataset represents employment duration."""
    },
    {
        "id": "loan_policy_004",
        "title": "Loan Approval Policy — Rejection Reasons",
        "content": """LoanCo Approval Policy — Section 4: Common Rejection Reasons

Common rejection reasons and thresholds:
1. Low credit score: EXT_SOURCE average below 0.35 indicates high default risk.
2. High debt burden: AMT_ANNUITY exceeds 40% of monthly income.
3. Short employment: DAYS_EMPLOYED shows less than 6 months of continuous work.
4. Loan amount too high: AMT_CREDIT exceeds 5x annual income.
5. Negative credit history: Previous defaults or delinquencies detected.
6. Insufficient income: AMT_INCOME_TOTAL below the minimum threshold for requested loan.
7. Age factor: Very young applicants without credit history are higher risk.

SHAP values explain which of these factors most influenced the rejection decision for each applicant."""
    },
    {
        "id": "loan_policy_005",
        "title": "Loan Risk Categories",
        "content": """LoanCo Risk Classification — Section 5: Risk Categories

All loan applicants are classified into risk tiers after model scoring:

Tier 1 — Low Risk (model score below 0.2): Auto-approved. No manual review required.
Tier 2 — Medium Risk (model score 0.2 to 0.5): Approved with standard review.
Tier 3 — High Risk (model score 0.5 to 0.75): Conditional approval. Supervisor sign-off mandatory.
Tier 4 — Very High Risk (model score above 0.75): Application rejected. Applicant receives written explanation citing top 3 SHAP factors. Re-application allowed after 6 months.

Default probability is the XGBoost model output. SHAP values provide per-feature explanations."""
    },
]

FRAUD_DOCUMENTS = [
    {
        "id": "fraud_rules_001",
        "title": "Insurance Fraud Detection — Red Flag Indicators",
        "content": """FraudShield Policy — Section 1: Red Flag Indicators

High-risk indicators of fraudulent insurance claims:
1. Policy holder at fault: Claims where the policy holder caused the accident have 8x higher fraud rate.
2. No police report: Absence of a police report for accidents above $5,000 is a red flag.
3. Minor damage but high claim: Vehicle shows minor damage but claim amount is disproportionately large.
4. Multiple prior claims: Claimant has filed more than 2 claims in the past 12 months.
5. Claim filed immediately: Claim filed within 24 hours of policy purchase.
6. Inconsistent statements: Witness statements contradict the claimant's account.
7. Expensive vehicle, low premium: High-value vehicle insured at suspiciously low premium level."""
    },
    {
        "id": "fraud_rules_002",
        "title": "Insurance Fraud Detection — Fault Type Analysis",
        "content": """FraudShield Policy — Section 2: Fault Type and Fraud Rates

Fraud rates by fault type from our dataset:
- Policy Holder fault: 18.4% fraud rate — highest risk category.
- Third Party Vehicle fault: 9.2% fraud rate — moderate risk.
- Third Party fault: 2.3% fraud rate — lowest risk category.

Why policy holder fault has higher fraud:
- Claimant controls the narrative with no conflicting third party.
- Staged accidents are easier when the policy holder controls the vehicle.
- Soft-tissue injuries like whiplash are unverifiable and often inflated.

Investigation protocol for policy holder fault claims above $10,000 requires independent medical examination."""
    },
    {
        "id": "fraud_rules_003",
        "title": "Insurance Fraud Detection — Anomaly Scores",
        "content": """FraudShield Policy — Section 3: Anomaly Detection and Scoring

Dual-model approach: XGBoost classifier and Isolation Forest anomaly detector.

Isolation Forest anomaly scores:
- Score above 0.6: Normal claim. Process immediately.
- Score 0.4 to 0.6: Borderline. Flag for manual review within 48 hours.
- Score below 0.4: Anomalous. Immediate investigation. Claim payment suspended.

XGBoost fraud probability thresholds:
- Below 0.3: Likely legitimate. Approve and process.
- 0.3 to 0.6: Uncertain. Request additional documentation.
- Above 0.6: Likely fraudulent. Escalate to Special Investigations Unit.

When both models agree (XGBoost above 0.6 AND Isolation Forest below 0.4), fraud confidence is very high."""
    },
    {
        "id": "fraud_rules_004",
        "title": "Insurance Fraud — Vehicle and Claim Patterns",
        "content": """FraudShield Policy — Section 4: Vehicle Price and Claim Patterns

Vehicle price range is a significant fraud predictor.

Findings from Oracle ML dataset analysis:
- Vehicles in the $0 to $20,000 range: Highest fraud volume by absolute count.
- Vehicles in the $40,000 to $60,000 range: Higher fraud rate per claim proportionally.
- Luxury vehicles above $60,000: Fraud tends to involve inflated repair costs.

Suspicious patterns by vehicle age:
- New vehicles (0 to 2 years) with total loss claims: Investigate for staged theft.
- Old vehicles (10+ years) with high repair claims: Investigate for over-valuation.
- Vehicles with recent ownership transfer: High risk for pre-planned fraud."""
    },
    {
        "id": "fraud_rules_005",
        "title": "Insurance Fraud — Investigation Checklist",
        "content": """FraudShield Policy — Section 5: Investigation Checklist

Step 1 — Document verification:
- Verify policy was active at time of incident.
- Check policy inception date vs claim date (claims under 30 days after purchase are suspicious).
- Confirm all named drivers match records.

Step 2 — Incident verification:
- Request police report number and verify with local authority.
- Check for prior claims on same vehicle or by same claimant.
- Review repair shop history for inflated estimates.

Step 3 — Medical verification for injury claims:
- Request GP records for pre-existing conditions.
- Order independent medical examination for soft tissue injuries.

Step 4 — Decision:
- Legitimate: Approve and process within 5 business days.
- Suspicious but unconfirmed: Negotiate reduced settlement.
- Confirmed fraud: Reject claim, notify authorities, flag policy for cancellation."""
    },
]

ALL_DOCUMENTS = LOAN_DOCUMENTS + FRAUD_DOCUMENTS
