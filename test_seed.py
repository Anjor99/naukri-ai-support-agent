# Designed to find a seed that generates a valid set of job applications (Especially for the flagged_priority_review field)

import random
from dataset import generate_applications, validate_applications

for seed in range(241, 11000):
    random.seed(seed)
    
    JOB_APPLICATIONS = generate_applications(50)

    try:
        validate_applications(JOB_APPLICATIONS)
        print("Successful seed:", seed)
        break
    except (ValueError, SyntaxError):
        continue