from agent.tools import check_job_application_status


# Test 1: 23-day priority application crosses threshold
result = check_job_application_status("APP-0001")

assert result["record_id"] == "APP-0001"
assert result["status"] == "Applied"
assert result["escalation_score"] == 0.8833
assert result["recommend_escalation"] is True
print("Test 1 passed: Escalation score and recommendation are correct for APP-0001")



# Test 2: older priority application has higher score
result = check_job_application_status("APP-0031")

assert result["escalation_score"] == 0.95
assert result["recommend_escalation"] is True
print("Test 2 passed: Escalation score and recommendation are correct for APP-0031")


# Test 3: invalid record ID raises an error
try:
    check_job_application_status("APP-9999")
    assert False, "Expected ValueError"
except ValueError:
    print("Test 3 passed: ValueError raised for invalid record ID")