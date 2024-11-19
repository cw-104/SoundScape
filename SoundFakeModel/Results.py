from sys import maxsize as MAX_VALUE_SIZE
# MINIMIZE FALSE POSITIVES

"""
# Range of classifications for DF values so - values
# DF Values at base are 
# - is DF + is Real
# to minimize False Positive Rate a certainty for DF should be more further into the negatives

# min: minimum value for the range
# max: maximum value for the range
# class_name: name of the class 
"""
class ResultClass: 
    def __init__(self, min : float, max : float, class_name : str):
        '''
        min: minimum value for the range
        max: maximum value for the range
        class_name: name of the class
        '''
        self.min = min
        self.max = max
        self.class_name = class_name

# IN BASE RESULTS: 
# DF is -
# REAL is +

# Goal is to MINIMIZE FALSE POSITIVES
# to do so we have to say that a DF is only certain if it is very negative

# we want to prevent a real file being classified as a DF so we need to give the model room 
# to ERROR based on the model's EER
RESULT_CLASSES = [
    ResultClass(0, -1, "DF Low Certainty"),
]

def classify_result(value : float):
    for result_class in RESULT_CLASSES:
        if value >= result_class.min and value < result_class.max:
            return result_class.class_name
    # if no class found then it is real
    return "Real"