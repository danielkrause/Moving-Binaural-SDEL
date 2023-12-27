# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters


argm=['867', '868']

for task in argm:
    params = parameters.get_params(task)

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)
    
    # # Extract features and normalize them
    dev_feat_cls.extract_all_feature()
    dev_feat_cls.preprocess_features()

    # # Extract labels in regression mode
    dev_feat_cls.extract_all_labels()

