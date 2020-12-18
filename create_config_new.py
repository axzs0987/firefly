import configparser
config = configparser.ConfigParser()

config["DEFAULT"] = {

}

config["models"] = {
    'model_dic' : {
        "RandomForestClassifier": 1,
        "LogisticRegression": 2,
        "KNeighborsClassifier": 3,
        "DecisionTreeClassifier": 4,
        "LinearRegression": 5,
        "RandomForestRegressor": 6,
        "XGBClassifier": 7,
        "GradientBoostingClassifier": 8,
        "KMeans": 9,
        "DecisionTreeRegressor": 10,
        "GradientBoostingRegressor": 11,
        "XGBRegressor": 12,
        "AdaBoostClassifier": 13,
        "AdaBoostRegressor": 14,
        "BaggingClassifier": 15,
        "BaggingRegressor": 16,
        "BayesianRidge": 17,
        "BernoulliNB": 18,
        "CalibratedClassifierCV": 19,
        "CatBoostClassifier": 20,
        "ElasticNet": 21,
        "ExtraTreeClassifier": 22,
        "ExtraTreesClassifier": 23,
        "GaussianMixture": 24,
        "GaussianNB": 25,
        "GridSearchCV": 26,
        "KNeighborsRegressor": 27,
        "Lasso": 28,
        "LassoCV": 29,
        "LinearSVC": 30,
        "MLPClassifier": 31,
        "MLPRegressor": 32,
        "MultinomialNB": 33,
        "RandomizedSearchCV": 34,
        "Ridge": 35,
        "RidgeCV": 36,
        "SGDClassifier": 37,
        "SGDRegressor": 38,
        "SVC": 39,
        "SVR": 40,
        "VotingClassifier": 41,
    },
    'model_task': [1,1,1,1,2,2,1,1,3,2,2,2,1]
}

config["database"] = {
    "host": '10.77.70.123',
    # "host": '127.0.0.1',
    "user": 'root',
    "passwd": 'mysqlinfo500',
    "dataset": 'kagglenotebook'
}

config["operators1"] = {
    "operations1":{
        "SimpleImputer": {
            "index": 1,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_constant", "filling_by_stratage"],
            "params": ["missing_values", "strategy", "fill_value", "verbose", "copy", "add_indicator"],
            'need_param': [0,1],
            'default_param_numeric': ['np.nan','most_frequent'],
            'default_param_str': ['np.nan','most_frequent'],
        },
        "dropna": {
            "index": 2,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["drop"],
            "params": ["axis", "how", "thresh", "subset", "inplace"],
            'need_param': [0,1,2,3],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "to_numeric": {
            "index": 3,
            "call_type": 2,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'TypeConversion',
            "physic_operations": ["type_conversion"],
            "params": ["arg", "errors", "downcast"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "LabelEncoder": {
            "index": 4,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": [],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "StandardScaler": {
            "index": 5,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["StandardScale"],
            "params": ["copy", "with_mean", "with_std"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "Normalizer": {
            "index": 6,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Normalization',
            "physic_operations": ["normalization"],
            "params": ["X", "norm", "axis", "copy", "return_norm"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "PCA": {
            "index": 7,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["PCA"],
            "params": ["n_components", "copy", "whiten", "svd_solver", "tol", "iterated_power", "random_state"],
            'default_param_numeric': [2],
            'default_param_str': [2],
        },
        "drop": {
            "index": 8,
            "call_type": 0,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["drop_column"],
            "params": ["labels", "axis", "index", "columns", "level", "inplace", "errors"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "drop_duplicates":{
            "index": 9,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Unique',
            "physic_operations": ["unique"],
            "params": ["subset", "keep", "inplace", "ignore_index"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "boxcox": {
            "index": 10,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew_and_kurt"],
            "params": ["x", "lmbda", "alpha"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "log": {
            "index": 11,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew"],
            "params": ["x", "out", "where"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "cut": {
            "index": 12,
            "call_type": 2,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Cut',
            "physic_operations": ["cut"],
            "params": ["x", "bins", "right", "labels", "retbins", "precision", "include_lowest", "duplicates"],
            'default_param_numeric': [5],
            'default_param_str': [5],
        },
    }
}
config["operators"] = {
    "operations":{
        "fillna": {
            "index": 1,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_constant", "filling_by_stratage"],
            "params": ["value", "method", "axis", "inplace", "limit", "downcast"],
            'need_param': [0],
            'default_param_numeric': [0],
            'default_param_str': ['none'],
        },
        "SimpleImputer": {
            "index": 3,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_constant", "filling_by_stratage"],
            "params": ["missing_values", "strategy", "fill_value", "verbose", "copy", "add_indicator"],
            'need_param': [0,1],
            'default_param_numeric': ['np.nan','mean'],
            'default_param_str': ['np.nan','mean'],
        },
        "Imputer": {
            "index": -1,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_by_stratage"],
            "params": ["missing_values", "strategy", "axis", "verbose", "copy"],
            'need_param': [0,1,2],
            'default_param_numeric': ['np.nan','mean'],
            'default_param_str': ['np.nan','mean'],
        },
        "dropna": {
            "index": 4,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["drop"],
            "params": ["axis", "how", "thresh", "subset", "inplace"],
            'need_param': [0,1,2,3],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "astype": {
            "index": 5,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'TypeConversion',
            "physic_operations": ["type_conversion"],
            "params": ["dtype", "copy", "errors"],
            'need_param': [0], #这个不用ast分析，简单判断等号，catagory
            'default_param_numeric': ['str'],
            'default_param_str': ['int'],
        },
        "to_numeric": {
            "index": 7,
            "call_type": 2,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'TypeConversion',
            "physic_operations": ["type_conversion"],
            "params": ["arg", "errors", "downcast"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "OneHotEncoder": {
            "index": 8,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["feature_encoding"],
            "params": ["categories", "drop", "sparse", "dtype", "handle_unknown"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "get_dummies": {
            "index": 9,
            "call_type": 2,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": ["data", "prefix", "prefix_sep", "dummy_na", "columns", "sparse"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "LabelEncoder": {
            "index": 10,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": [],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "LabelBinarizer": {
            "index": 11,
            "call_type": 3,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": ["neg_label", "pos_label", "sparse_output"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "StandardScaler": {
            "index": 12,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["StandardScale"],
            "params": ["copy", "with_mean", "with_std"],
            'need_param': [],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "MinMaxScaler": {
            "index": 13,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["Scaler"],
            "params": ["feature_range", "copy"],
            'need_param': [0],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "RobustScaler": {
            "index": 14,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["Scaler"],
            "params": ["with_centering", "with_scaling", "quantile_range", "copy"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "clip": {
            "index": 15,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["clip"],
            "params": ["a", "a_min", "a_max", "out"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "Normalizer": {
            "index": 16,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Normalization',
            "physic_operations": ["normalization"],
            "params": ["X", "norm", "axis", "copy", "return_norm"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "l2_normalize":{
            "index": -1,
            "call_type": 4,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Normalization',
            "physic_operations": ["normalization"],
            "params": ["x", "axis", "epsilon", "name", "dim"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "PCA": {
            "index": 17,
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["PCA"],
            "params": ["n_components", "copy", "whiten", "svd_solver", "tol", "iterated_power", "random_state"],
            'default_param_numeric': [2],
            'default_param_str': [2],
        },
        "drop": {
            "index": 18,
            "call_type": 0,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["drop_column"],
            "params": ["labels", "axis", "index", "columns", "level", "inplace", "errors"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "iloc": {
            "index": -1,
            "call_type": 5,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["drop_column"],
            "params": ["labels", "axis", "index", "columns", "level", "inplace", "errors"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "drop_duplicates":{
            "index": 19,
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Unique',
            "physic_operations": ["unique"],
            "params": ["subset", "keep", "inplace", "ignore_index"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        # "unique":{
        #     "index": -1,
        #     "call_type": 2,
        # # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
        #     "logic_operations": 'Unique',
        #     "physic_operations": ["unique"],
        #     "params": ["values"],
        #     'default_param_numeric': [],
        #     'default_param_str': [],
        # },
        "boxcox": {
            "index": 20,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew_and_kurt"],
            "params": ["x", "lmbda", "alpha"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "boxcox1p": {
            "index": 21,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew_and_kurt"],
            "params": ["x", "lmbda"],
            'default_param_numeric': [0.15],
            'default_param_str': [0.15],
        },
        "log": {
            "index": 22,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew"],
            "params": ["x", "out", "where"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "log1p": {
            "index": 23,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew"],
            "params": ["x", "out", "where"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "expm1": {
            "index": 24,
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["inverse_skew"],
            "params": ["x", "out", "where"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
        "cut": {
            "index": 25,
            "call_type": 2,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Cut',
            "physic_operations": ["cut"],
            "params": ["x", "bins", "right", "labels", "retbins", "precision", "include_lowest", "duplicates"],
            'default_param_numeric': [5],
            'default_param_str': [5],
        },
        "qcut": {
            "index": -1,
            "call_type": 2,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Cut',
            "physic_operations": ["cut"],
            "params": ["x", "q","labels", "retbins", "precision", "duplicates"],
            'default_param_numeric': [4],
            'default_param_str': [4],
        },
        "map": {
            "index": -1,
            "call_type": 0,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'ValueTransform',
            "physic_operations": ["value_transform"],
            "params": ["arg", "na_action"],
            'default_param_numeric': [],
            'default_param_str': [],
        },
    }
}
config["dtype"] = {
    "dtype":{
        'uint8':1,
        'uint16': 1,
        'int64': 1,
        'int': 1,
        'int32': 1,
        'int16': 1,
        'np.int32': 1,
        'np.int64': 1,
        'np.int': 1,
        'np.int16': 1,
        'float64': 2,
        'float': 2,
        'float32': 2,
        'float16': 2,
        'np.float32': 2,
        'np.float64': 2,
        'np.float': 2,
        'np.float16': 2,
        'str':3,
        'Category':4,
        'object':4,
    }
}
config["metrics"] = {
    "metrics":{
        ".score(":{
            "type": 1,
            "len": 6,
            "name": "score",
        },
        ".evaluate(":{
            "type": 1,
            "len": 9,
            "name": "evaluate",
        },
        ".best_score_": {
            "type": 2,
            "len": 11,
            "name": "best_score_",
        },
        "accuracy_score(": {
            "type": 3,
            "len": 14,
            "name": "accuracy_score",
            "pred_index":1,
            "pred_type":0, #y_pred
        },
        "auc(": {
            "type": 3,
            "len": 3,
            "name": "auc",
            "pred_index":1,
            "pred_type":0, #y_pred
        },
        'f1_score(': {
            "type": 3,
            "len": 8,
            "name": "f1_score",
            "pred_index":1,
            "pred_type":0, #y_pred
        },
        'r2_score(': {
            "type": 3,
            "len": 8,
            "name": "r2_score",
            "pred_index":1,
            "pred_type":0, #y_pred
        },
        'cross_val_score(': {
            "type": 3,
            "len": 15,
            "name": "cross_val_score",
            "pred_index":0,
            "pred_type":2, #y_pred
        },
        'average_precision_score(': {
            "type": 3,
            "len": 23,
            "name": "average_precision_score",
            "pred_index":1,
            "pred_type":1, #y_score
        },
        'balanced_accuracy_score(': {
            "type": 3,
            "len": 23,
            "name": "balanced_accuracy_score",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'brier_score_loss(': {
            "type": 3,
            "len": 16,
            "name": "brier_score_loss",
            "pred_index":1,
            "pred_type":1, #y_score
        },
        'classification_report(': {
            "type": 3,
            "len": 21,
            "name": "classification_report",
        },
        'cohen_kappa_score(': {
            "type": 3,
            "len": 18,
            "name": "cohen_kappa_score",
            "pred_index":1,
            "pred_type":1, #y_score
        },
        'confusion_matrix(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "confusion_matrix",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'dcg_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "dcg_score",
            "pred_index":1,
            "pred_type":1, #y_score
        },
        'fbeta_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "fbeta_score",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'hamming_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "hamming_loss",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'hinge_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "hinge_loss",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'jaccard_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "jaccard_score",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'log_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "log_loss",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'matthews_corrcoef(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "matthews_corrcoef",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'multilabel_confusion_matrix(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "multilabel_confusion_matrix",
            "pred_index":1,
            "pred_type":0, #y_score
        },
        'ndcg_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "ndcg_score",
            "pred_index":1,
            "pred_type":1, #y_score
        },'precision_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "precision_score",
            "pred_index":1,
            "pred_type":0, #y_score
        },'recall_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "recall_score",
            "pred_index":1,
            "pred_type":0, #y_score
        },'roc_auc_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "roc_auc_score",
            "pred_index":1,
            "pred_type":1, #y_score
        },'zero_one_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "zero_one_loss",
            "pred_index":1,
            "pred_type":0, #y_score
        },'explained_variance_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "explained_variance_score",
            "pred_index":1,
            "pred_type":0, #y_score
        },'max_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "max_error",
            "pred_index":1,
            "pred_type":0, #y_score
        },'mean_absolute_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_absolute_error",
            "pred_index":1,
            "pred_type":0, #y_score
        },'mean_squared_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_squared_error",
            "pred_index":1,
            "pred_type":0, #y_score
        },'mean_squared_log_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_squared_log_error",
            "pred_index":1,
            "pred_type":0, #y_score
        },'median_absolute_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "median_absolute_error",
            "pred_index": 1,
            "pred_type": 0,  # y_score
        },'mean_poisson_deviance(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_poisson_deviance",
            "pred_index": 1,
            "pred_type": 0,  # y_score
        },'mean_gamma_deviance(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_gamma_deviance",
            "pred_index": 1,
            "pred_type": 0,  # y_score
        },'mean_tweedie_deviance(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_tweedie_deviance",
            "pred_index": 1,
            "pred_type": 0,  # y_score
        },'adjusted_mutual_info_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "adjusted_mutual_info_score",
        },'adjusted_rand_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "adjusted_rand_score",
        },'calinski_harabasz_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "calinski_harabasz_score",
        },'davies_bouldin_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "davies_bouldin_score",
        },'completeness_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "completeness_score",
        },'cluster.contingency_matrix(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "cluster.contingency_matrix",
        },'fowlkes_mallows_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "fowlkes_mallows_score",
        },'homogeneity_completeness_v_measure(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "homogeneity_completeness_v_measure",
        },'homogeneity_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "homogeneity_score",
        },'mutual_info_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mutual_info_score",
        },'normalized_mutual_info_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "normalized_mutual_info_score",
        },'silhouette_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "silhouette_score",
        },'silhouette_samples(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "silhouette_samples",
        },'v_measure_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "v_measure_score",
        },

    }
}

config["server"] = {
    "server":{
        '10.77.70.122':{
            'wpath': '',
            'npath': '',
            'dpath': '',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.123':{
            'wpath': '/root/walklogs/',
            'npath': '/root/notebook/',
            'dpath': '/root/unzip_dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.124':{
            'wpath': '/root/walklogs/',
            'npath': '/root/notebook/',
            'dpath': '/root/unzip_dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.125':{
            'wpath': '/root/walklogs/',
            'npath': '/root/notebook/',
            'dpath': '/root/unzip_dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.126':{
            'wpath': '/root/walklogs/',
            'npath': '/root/notebook/',
            'dpath': '/root/unzip_dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.127':{
            'wpath': '/root/walklogs/',
            'npath': '/root/notebook/',
            'dpath': '/root/unzip_dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.128':{
            'wpath': '/root/walklogs/',
            'npath': '/root/notebook/',
            'dpath': '/root/unzip_dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '39.99.150.216':{
            'wpath': '/home/chsibei//walklogs/',
            'npath': '/home/chsibei/spider/notebook/',
            'dpath': '/home/chsibei/spider/unzip_dataset/',
            'port' : 22,
            'username' : 'chsibei',
            'password' : 'info500',
        },
        'localhost':{
            'wpath': '/Users/baiping/fsdownload/walklogs/',
            'npath': '/Users/baiping/fsdownload/spider/notebook/',
            # 'dpath': '/home/chsibei/spider/unzip_dataset',
            # 'port' : 22,
            # 'username' : 'chsibei',
            # 'password' : 'info500',
        }
    }
}

config["train"] = {
    "train":{
        "nepisode": 1000,
        "obs_dim": 1962,
        "learning_rate": 0.05,
        "gamma": 0.99,
        "column_num": 101,
        'dense_dim': 100,
        'epsilon': 0.5,
        'epsilon_step': 10,
        'batch_size': 20,
    }
}

with open('config.ini','w') as f:
    config.write(f)
