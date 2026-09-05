import random

random.seed(1550)

required_fields = {
    "record_id",
    "category",
    "status",
    "expected_salary_inr",
    "days_since_created",
    "flagged_priority_review"
}

categories = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate"
]

status_opts = ["Applied","Screening","Interview Scheduled","Offered","Rejected"]

def generate_applications(n : int) -> list[dict]:
    
    """Generate a list of job applications with random data.

    Args:
        n (int): Number of applications to generate

    Returns:
        list[dict]: A list of dictionaries, each representing a job application with the following fields:
            - record_id (str): Unique identifier for the application.
            - category (str): The category of the job application.
            - status (str): The status of the job application.
            - expected_salary_inr (int): The expected salary in INR.
            - days_since_created (int): The number of days since the application was created.
            - flagged_priority_review (bool): A flag indicating if the application requires priority review.
    """
    
    JOB_APPLICATIONS = []
    # Making sure "flagged_priority_review" retains only around 20% True
    # priority_count = int(n * 0.20)

    # priority_indices = random.sample(
    #     range(1, n + 1),
    #     priority_count
    # )
    # For Populating JOB_APPLICATIONS
    for i in range(1,n+1):
        application = {}
        application["record_id"] = f"APP-{i:04d}"
        application["category"] = random.choice(categories)
        application["status"] = random.choice(status_opts)
        """
         I referred https://www.ambitionbox.com/salaries/naukri-salaries to get the range of salaries on naukri. 
         I have taken the range of 3 LPA to 30 LPA as the expected salary for job applications
          to cover maximum scenarios of earnings who apply on the platform.
        """
        application["expected_salary_inr"] = random.randint(300000,3000000)
        application["days_since_created"] = random.randint(0,30)
        # if i in priority_indices:
        #     application["flagged_priority_review"] = True
        # else:
        #     application["flagged_priority_review"] = False
        application["flagged_priority_review"] = random.choices([True, False])[0]
        JOB_APPLICATIONS.append(application)
        
    return JOB_APPLICATIONS
            
def validate_applications(job_appls : list[dict]) -> bool:
    """Validate a list of job applications.
    
    Args:
        job_appls (list[dict]): A list of dictionaries representing job applications.

    Returns:
        count (dict): A dictionary containing counts of applications by category, status, and flagged priority review.

    Raises:
        ValueError: If any application fails validation.
    """
    # Validations for generated JOB_APPLICATIONS
    if len(job_appls) < 40:
        raise ValueError(f"Atleast 40 applications required, Current count : {len(job_appls)}")
    
    count = {
        "flagged_priority_review":0,
        "category" : {},
        "status" : {}
    }
    
    for cat in categories:
        count["category"][cat] = 0
        
    for stat in status_opts:
        count["status"][stat] = 0
    
    for appl in job_appls:
        # Check if the application contains all the required fields
        if not required_fields.issubset(appl.keys()):
            raise SyntaxError(f"Missing keys detected in an application record, Required fields: {required_fields}")
        
        # Check that every category belongs to categories decided
        category = appl.get('category')
        
        if category not in categories:
            raise ValueError(f"Invalid category: {category}")
        
        # Raise count for the specific category
        count["category"][category]+=1
        
        # Check that every status belongs to categories decided
        status = appl.get('status')
        
        if status not in status_opts:
            raise ValueError(f"Invalid status: {status}")
        
        # Raise count for the specific status
        count["status"][status]+=1
        
        # Condition : 0 <= days_since_created <=30
        days = appl.get('days_since_created')
        if days < 0 or days > 30:
            raise ValueError("days_since_created value should be between or equal to 0 & 30")
        
        # Count flag priority review True values
        if appl.get('flagged_priority_review'):
            count["flagged_priority_review"] += 1
    
    # Making sure each application has 
    for category in count["category"]:
        if count["category"][category] < 3:
            raise ValueError(f"Too few applications with category : {category}")
        
    for status in count["status"]:
        if count["status"][status] < 1:
            raise ValueError(f"Too few applications with status : {status}")
        
    priority_percentage = (count["flagged_priority_review"] / len(job_appls))*100
    
    if priority_percentage > 30 or priority_percentage < 10:
        raise ValueError(f"flagged_priority_review True percentage must be between 10 and 30, Current is : {priority_percentage} %")
    
    return count

JOB_APPLICATIONS = generate_applications(50) # To universally generate applications for any other module that imports this dataset.py file
            
def main():
    JOB_APPLICATIONS = generate_applications(50)
    try:
        valid_app_count = validate_applications(JOB_APPLICATIONS)
        print("All applications are valid.")
        print(f"Total applications: {len(JOB_APPLICATIONS)}")
        print(f"flagged_priority_review True percentage: {valid_app_count['flagged_priority_review'] / len(JOB_APPLICATIONS) * 100:.2f}%")
        print(f"Category counts: {valid_app_count['category']}")
        print(f"Status counts: {valid_app_count['status']}")
        
    except (ValueError, SyntaxError) as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    main()