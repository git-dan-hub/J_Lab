
custom_yaml = """
dependencies:
  - pip
  - pip:
    - --index-url https://test.pypi.org/simple/
    - lightgbm==3.3.1
"""

with open("tmp.yaml","w") as f:
    f.write(custom_yaml)

print("tmp.yaml written")


def score(input_data) :
    import subprocess
    
    # ignore input_data
    my_log_msgs = []
    
    runlog = subprocess.run(["conda","env","update","--file","tmp.yaml"], check=False, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    my_log_msgs.append(runlog.stdout)
    try:
        import lightgbm   
    except Exception as exc:
        my_log_msgs.append("import failed")
    
    
    score_response = {
            'predictions': [{ 'fields':["log_msgs"], 
                              'values': [my_log_msgs] }]
    } 
    return score_response
