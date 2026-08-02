# Week 3 — API Validation Notes (Day 4)

Tested /predict/churn and /predict/ltv with two contrasting customer profiles
to confirm the models behave sensibly.

## Test 1: At-risk customer
- Profile: 2 months tenure, month-to-month contract, no add-on services, fiber optic internet
- Churn probability: 0.8352 (Yes)
- Predicted LTV: 187.25

## Test 2: Loyal customer
- Profile: 60 months tenure, two-year contract, all add-on services, DSL internet
- Churn probability: 0.116 (No)
- Predicted LTV: 3314.43

## Conclusion
Both models responded correctly to contrasting inputs — the at-risk customer
showed high churn probability and low LTV, while the loyal customer showed
low churn probability and high LTV. This confirms the models are learning
meaningful patterns rather than outputting arbitrary values.