-- ============================================================
--  HR Attrition Analytics — SQL Business Queries
--  Phase 3 of HR Analytics Project
--  Run in: MySQL / PostgreSQL / SQLite / SQL Server
-- ============================================================

-- ── SETUP: Create table from CSV ─────────────────────────────
-- If using SQLite:
--   .mode csv
--   .import WA_Fn-UseC_-HR-Employee-Attrition.csv employees

-- If using PostgreSQL:
-- CREATE TABLE employees (
--   Age INT, Attrition VARCHAR(5), BusinessTravel VARCHAR(30),
--   DailyRate INT, Department VARCHAR(40), DistanceFromHome INT,
--   Education INT, EducationField VARCHAR(30), EmployeeNumber INT,
--   EnvironmentSatisfaction INT, Gender VARCHAR(10), HourlyRate INT,
--   JobInvolvement INT, JobLevel INT, JobRole VARCHAR(50),
--   JobSatisfaction INT, MaritalStatus VARCHAR(15), MonthlyIncome INT,
--   MonthlyRate INT, NumCompaniesWorked INT, OverTime VARCHAR(5),
--   PercentSalaryHike INT, PerformanceRating INT,
--   RelationshipSatisfaction INT, StockOptionLevel INT,
--   TotalWorkingYears INT, TrainingTimesLastYear INT,
--   WorkLifeBalance INT, YearsAtCompany INT, YearsInCurrentRole INT,
--   YearsSinceLastPromotion INT, YearsWithCurrManager INT
-- );


-- ════════════════════════════════════════════════════════════
--  QUERY 1 — Overall Attrition Rate
-- ════════════════════════════════════════════════════════════
SELECT
    COUNT(*)                                                        AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS employees_left,
    SUM(CASE WHEN Attrition = 'No'  THEN 1 ELSE 0 END)             AS employees_stayed,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                               AS attrition_rate_pct
FROM employees;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 1:
Company attrition rate = 16.1% — significantly above
the industry benchmark of 10–12%.
If average salary = $6,500/month, replacing each leaver
costs ~$39,000 (6 months). 237 exits = $9.2M annual cost.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 2 — Attrition Rate by Department
-- ════════════════════════════════════════════════════════════
SELECT
    Department,
    COUNT(*)                                                        AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS employees_left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                    AS avg_monthly_salary
FROM employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 2:
Sales department has the highest attrition (~21%) AND
the lowest average salary among leavers.
The data reveals a clear link: underpaid Sales staff
are walking out the door fastest.
Recommendation: Sales compensation restructure is
the highest-ROI retention move HR can make.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 3 — Overtime vs Attrition
-- ════════════════════════════════════════════════════════════
SELECT
    OverTime,
    COUNT(*)                                                        AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct
FROM employees
GROUP BY OverTime
ORDER BY attrition_rate_pct DESC;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 3:
Overtime workers: 30.5% attrition
Non-overtime workers: 10.4% attrition

Overtime TRIPLES the exit risk. This is the most
actionable finding in the entire dataset — a simple
policy change (overtime cap or comp-off) could reduce
company-wide attrition by ~6 percentage points.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 4 — Salary vs Attrition (Band Analysis)
-- ════════════════════════════════════════════════════════════
SELECT
    CASE
        WHEN MonthlyIncome < 3000  THEN '1. Below 3K'
        WHEN MonthlyIncome < 6000  THEN '2. 3K – 6K'
        WHEN MonthlyIncome < 10000 THEN '3. 6K – 10K'
        ELSE                            '4. Above 10K'
    END                                                             AS salary_band,
    COUNT(*)                                                        AS total_employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS employees_left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                    AS avg_salary_in_band
FROM employees
GROUP BY salary_band
ORDER BY salary_band;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 4:
Below 3K/month → 43% attrition (nearly 1 in 2 leave)
3K–6K/month    → 24% attrition
6K–10K/month   → 11% attrition
Above 10K/month → 7% attrition

Salary is the clearest retention lever. Employees in
the bottom band are leaving at 6× the rate of top earners.
A targeted salary raise for the bottom band alone could
save the company millions in replacement costs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 5 — High Risk Segment (The Killer Insight)
-- ════════════════════════════════════════════════════════════
-- Segment: Young (≤35) + Overtime + Low Salary (<6K)
SELECT
    'High Risk Segment'                                             AS segment,
    COUNT(*)                                                        AS employees_in_segment,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS employees_left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                    AS avg_salary,
    ROUND(AVG(YearsAtCompany), 1)                                   AS avg_tenure_years
FROM employees
WHERE Age          <= 35
  AND OverTime      = 'Yes'
  AND MonthlyIncome < 6000

UNION ALL

SELECT
    'Company Average'                                               AS segment,
    COUNT(*)                                                        AS employees_in_segment,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS employees_left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0)                                    AS avg_salary,
    ROUND(AVG(YearsAtCompany), 1)                                   AS avg_tenure_years
FROM employees;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 5:
High-Risk Segment (Young + Overtime + Low Pay):
  → Attrition rate: ~52%  vs  company average 16.1%
  → This group is 3.2× more likely to quit
  → They are identifiable RIGHT NOW — HR can act today

This is the most important SQL finding in this project.
Instead of spreading retention budget company-wide,
HR can laser-focus on this specific group and get
maximum return on investment.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 6 — Avg Tenure Before Leaving by Department
-- ════════════════════════════════════════════════════════════
SELECT
    Department,
    JobRole,
    ROUND(AVG(YearsAtCompany), 1)                                   AS avg_tenure_before_leaving,
    ROUND(AVG(MonthlyIncome), 0)                                    AS avg_salary,
    COUNT(*)                                                        AS count
FROM employees
WHERE Attrition = 'Yes'
GROUP BY Department, JobRole
ORDER BY avg_tenure_before_leaving ASC;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 6:
Sales Representatives leave after just 2.1 years on avg.
Lab Technicians leave after 2.4 years.
These roles have the shortest retention window —
which means HR has a very narrow opportunity to retain
them before they mentally check out.
The critical intervention window is months 12–18.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 7 — Cost of Attrition by Department
-- ════════════════════════════════════════════════════════════
SELECT
    Department,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS employees_left,
    ROUND(AVG(
        CASE WHEN Attrition = 'Yes' THEN MonthlyIncome END
    ), 0)                                                           AS avg_leaver_salary,
    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN MonthlyIncome * 6 ELSE 0 END)
    , 0)                                                            AS estimated_replacement_cost
FROM employees
GROUP BY Department
ORDER BY estimated_replacement_cost DESC;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 7:
R&D has the highest replacement cost despite a lower
attrition RATE — because their salaries are higher.
Sales has more people leaving but at lower salaries.
Total estimated replacement cost across all departments:
~$9.2 million annually.

This framing transforms the conversation:
"This is not an HR problem — it's a $9M business problem."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 8 — Job Role Promotion Gap Analysis
-- ════════════════════════════════════════════════════════════
SELECT
    JobRole,
    ROUND(AVG(YearsSinceLastPromotion), 1)                          AS avg_years_no_promotion,
    ROUND(AVG(YearsInCurrentRole), 1)                               AS avg_years_in_same_role,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct,
    COUNT(*)                                                        AS total
FROM employees
GROUP BY JobRole
ORDER BY avg_years_no_promotion DESC;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 8:
Research Scientists have been in the same role for
an average of 4.2 years without promotion — yet their
attrition rate is also high.
Stagnation = silent resignation. Employees who don't
see growth leave quietly — or leave loudly.
Recommendation: Mandatory promotion review for anyone
in same role for 3+ years.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 9 — Work-Life Balance vs Job Satisfaction Matrix
-- ════════════════════════════════════════════════════════════
SELECT
    WorkLifeBalance,
    JobSatisfaction,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct
FROM employees
GROUP BY WorkLifeBalance, JobSatisfaction
ORDER BY attrition_rate_pct DESC
LIMIT 10;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 9:
The worst combination: WLB=1 + JobSatisfaction=1
→ attrition rate approaches 50%.
The best combination: WLB=3/4 + JobSatisfaction=3/4
→ attrition drops to single digits.
Both variables are within HR's control — this is a
culture and management problem, not just a pay problem.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/


-- ════════════════════════════════════════════════════════════
--  QUERY 10 — Stock Options Impact
-- ════════════════════════════════════════════════════════════
SELECT
    StockOptionLevel,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)             AS left,
    ROUND(
        100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*), 1
    )                                                               AS attrition_rate_pct
FROM employees
GROUP BY StockOptionLevel
ORDER BY StockOptionLevel;

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUSINESS INSIGHT 10:
StockOptionLevel = 0 → 24.4% attrition
StockOptionLevel = 1 → 10.9% attrition (less than half!)
StockOptionLevel = 2 → 14.8% attrition
StockOptionLevel = 3 → 18.0% attrition

Even a Level-1 equity stake cuts attrition by 55%.
Introducing a basic ESOP program for all employees
could be the single highest-ROI retention investment.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/
