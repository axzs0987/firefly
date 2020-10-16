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

config["operators"] = {
    "operations":{
        "fillna": {
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_constant", "filling_by_stratage"],
            "params": ["value", "method", "axis", "inplace", "limit", "downcast"],
        },
        "SimpleImputer": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_constant", "filling_by_stratage"],
            "params": ["missing_values", "strategy", "fill_value", "verbose", "copy", "add_indicator"],
        },
        "Imputer": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["filling_by_stratage"],
            "params": ["missing_values", "strategy", "axis", "verbose", "copy"],
        },
        "dropna": {
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Imputer',
            "physic_operations": ["drop"],
            "params": ["axis", "how", "thresh", "subset", "inplace"],
        },
        "astype": {
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'TypeConversion',
            "physic_operations": ["type_conversion"],
            "params": ["dtype", "copy", "errors"],
        },
        "to_numeric": {
            "call_type": 2,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'TypeConversion',
            "physic_operations": ["type_conversion"],
            "params": ["arg", "errors", "downcast"],
        },
        "OneHotEncoder": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["feature_encoding"],
            "params": ["categories", "drop", "sparse", "dtype", "handle_unknown"],
        },
        "get_dummies": {
            "call_type": 2,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": ["data", "prefix", "prefix_sep", "dummy_na", "columns", "sparse"],
        },
        "LabelEncoder": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": [],
        },
        "LabelBinarizer": {
            "call_type": 3,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'OnehotEncoding',
            "physic_operations": ["label_encoding"],
            "params": ["neg_label", "pos_label", "sparse_output"],
        },
        "StandardScaler": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["StandardScale"],
            "params": ["copy", "with_mean", "with_std"],
        },
        "MinMaxScaler": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["Scaler"],
            "params": ["feature_range", "copy"],
        },
        "RobustScaler": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["Scaler"],
            "params": ["with_centering", "with_scaling", "quantile_range", "copy"],
        },
        "clip": {
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Scale',
            "physic_operations": ["clip"],
            "params": ["a", "a_min", "a_max", "out"],
        },
        "Normalizer": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Normalization',
            "physic_operations": ["normalization"],
            "params": ["X", "norm", "axis", "copy", "return_norm"],
        },
        "l2_normalize":{
            "call_type": 4,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Normalization',
            "physic_operations": ["normalization"],
            "params": ["x", "axis", "epsilon", "name", "dim"],
        },
        "PCA": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["PCA"],
            "params": ["n_components", "copy", "whiten", "svd_solver", "tol", "iterated_power", "random_state"],
        },
        "drop": {
            "call_type": 0,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'DimensionReduction',
            "physic_operations": ["drop_column"],
            "params": ["labels", "axis", "index", "columns", "level", "inplace", "errors"],
        },
        "drop_duplicates":{
            "call_type": 0,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Unique',
            "physic_operations": ["unique"],
            "params": ["subset", "keep", "inplace", "ignore_index"],
        },
        "unique":{
            "call_type": 2,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Unique',
            "physic_operations": ["unique"],
            "params": ["values"],
        },
        "boxcox": {
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew_and_kurt"],
            "params": ["x", "lmbda", "alpha"],
        },
        "boxcox1p": {
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew_and_kurt"],
            "params": ["x", "lmbda"],
        },
        "log": {
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew"],
            "params": ["x", "out", "where"],
        },
        "log1p": {
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["skew"],
            "params": ["x", "out", "where"],
        },
        "expm1": {
            "call_type": 4,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Smooth',
            "physic_operations": ["inverse_skew"],
            "params": ["x", "out", "where"],
        },
        "cut": {
            "call_type": 2,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Cut',
            "physic_operations": ["cut"],
            "params": ["x", "bins", "right", "labels", "retbins", "precision", "include_lowest", "duplicates"],
        },
        "qcut": {
            "call_type": 2,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'Cut',
            "physic_operations": ["cut"],
            "params": ["x", "q","labels", "retbins", "precision", "duplicates"],
        },
        "map": {
            "call_type": 0,
            # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, xxx.operator(data, ...): 4;
            "logic_operations": 'ValueTransform',
            "physic_operations": ["value_transform"],
            "params": ["arg", "na_action"],
        },
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
        "best_score_": {
            "type": 2,
            "len": 10,
            "name": "best_score_",
        },
        "accuracy_score(": {
            "type": 3,
            "len": 14,
            "name": "accuracy_score",
        },
        "auc(": {
            "type": 3,
            "len": 3,
            "name": "auc",
        },
        'f1_score(': {
            "type": 3,
            "len": 8,
            "name": "f1_score",
        },
        'r2_score(': {
            "type": 3,
            "len": 8,
            "name": "r2_score",
        },
        'cross_val_score(': {
            "type": 3,
            "len": 15,
            "name": "cross_val_score",
        },
        'average_precision_score(': {
            "type": 3,
            "len": 23,
            "name": "average_precision_score",
        },
        'balanced_accuracy_score(': {
            "type": 3,
            "len": 23,
            "name": "balanced_accuracy_score",
        },
        'brier_score_loss(': {
            "type": 3,
            "len": 16,
            "name": "brier_score_loss",
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
        },
        'confusion_matrix(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "confusion_matrix",
        },
        'dcg_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "dcg_score",
        },
        'fbeta_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "fbeta_score",
        },
        'hamming_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "hamming_loss",
        },
        'hinge_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "hinge_loss",
        },
        'jaccard_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "jaccard_score",
        },
        'log_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "log_loss",
        },
        'matthews_corrcoef(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "matthews_corrcoef",
        },
        'multilabel_confusion_matrix(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "multilabel_confusion_matrix",
        },
        'ndcg_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "ndcg_score",
        },'precision_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "precision_score",
        },'recall_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "recall_score",
        },'roc_auc_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "roc_auc_score",
        },'zero_one_loss(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "zero_one_loss",
        },'explained_variance_score(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "explained_variance_score",
        },'max_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "max_error",
        },'mean_absolute_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_absolute_error",
        },'mean_squared_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_squared_error",
        },'mean_squared_log_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_squared_log_error",
        },'median_absolute_error(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "median_absolute_error",
        },'mean_poisson_deviance(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_poisson_deviance",
        },'mean_gamma_deviance(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_gamma_deviance",
        },'mean_tweedie_deviance(': {
            "type": 3,
            "len": 18, #开始乱写了
            "name": "mean_tweedie_deviance",
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
            'npath': '',
            'dpath': '',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.123':{
            'npath': '/root/notebook/',
            'dpath': '/root/dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.124':{
            'npath': '/root/notebook/',
            'dpath': '/root/dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.125':{
            'npath': '/root/notebook/',
            'dpath': '/root/dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.126':{
            'npath': '/root/notebook/',
            'dpath': '/root/dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.127':{
            'npath': '/root/notebook/',
            'dpath': '/root/dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '10.77.70.128':{
            'npath': '/root/notebook/',
            'dpath': '/root/dataset/',
            'port' : 22,
            'username' : 'root',
            'password' : 'pei54188',
        },
        '39.99.150.216':{
            'npath': '/home/chsibei/spider/notebook',
            'dpath': '/home/chsibei/spider/unzip_dataset',
            'port' : 22,
            'username' : 'chsibei',
            'password' : 'info500',
        },
    }
}
with open('config.ini','w') as f:
    config.write(f)
