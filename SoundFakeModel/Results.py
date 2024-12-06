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
class DfResultHandler: 
    def __init__(self, center_point : float, lower_class_name : str, higher_class_name : str, percentage_divisor : float, percentage_cap : float = .99):
        '''
        center_point: minimum value for the range,
            - ex center point -5 means the model would have to have a value below -5 to be certain it is a df
        lower_class_name: if below center what is class name
        higher_class_name: if above center what is class name
        percentage_divisor : what should the distance from center be dist/x, larger score = higher certainty so farther dist = higher percentage
        percentage_cap : percentage value the certainty should never exceed for purposes of error regulation
        '''
        self.center_point = center_point
        self.lower_class_name = lower_class_name
        self.higher_class_name = higher_class_name
        self.percentage_divisor = percentage_divisor
        self.percentage_cap = percentage_cap
    
    class DfResult:
        def __init__(self,
                    is_lower_class : bool,
                    classification: str = None,
                    raw_value: float = 0.,
                    shifted_value: float = 0.,
                    percent_certainty: float = 0.,
                    file_name: str = None
                    ):
            '''
            --- result of the evaluation 
            .classification {str} classification name of audio file

            .raw_value {float} raw model results
            .shifted_value {float} shifted model results according to classification ranges
            .percent_certainty {float} how sure the model is of its result
            '''
            
            self.classification = classification          
            self.raw_value = raw_value              
            self.shifted_value = shifted_value
            self.percent_certainty = percent_certainty
            self.is_lower_class = is_lower_class
            self.file_name = file_name
        def __str__(self):
            return f"{self.file_name:<32}: {self.classification:<5} | {(self.percent_certainty*100):3.0f}% certain (adjusted: {(self.shifted_value):2.2f} | raw: {(self.raw_value):2.2f})"

    def generate_result(self, fname, score):
        isLower = score < self.center_point
        shifted = score - self.center_point
        return DfResultHandler.DfResult(
            classification = self.lower_class_name if isLower else self.higher_class_name,
            raw_value = score,
            shifted_value = shifted,
            percent_certainty = min(abs(shifted)/self.percentage_divisor, self.percentage_cap),
            is_lower_class = isLower,
            file_name=fname
        )

        
