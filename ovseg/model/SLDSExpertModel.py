from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.preprocessing.SLDSExpertPreprocessing import SLDSExpertPreprocessing

class RegionexpertModel(RegionexpertModel):
    
    def _create_preprocessing_object(self):
        
        self.preprocessing = SLDSExpertPreprocessing(**self.model_parameters['preprocessing'])
