from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

raw_data = ['OV04']

preprocessing = SegmentationPreprocessing(use_only_classes=[9])
preprocessing.plan_preprocessing_raw_data(raw_data)
preprocessing.preprocess_raw_data(raw_data)
