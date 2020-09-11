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
        ###########################
        'KNN': 3,
        'knn': 3,
        'logic_regression': 2,
        'linear_regression': 5,
        'kmeans': 9,
        'xgb': 12,
        'ada_boost': 13,
        'AdaBoost': 13,
    }
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
        "Normalizer": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3;
            "logic_operations": 'Normalization',
            "physic_operations": ["normalization"],
            "params": ["X", "norm", "axis", "copy", "return_norm"],
        },
        "PCA": {
            "call_type": 3,
        # data.operator() : 0, operator(data, ...) : 1, pd.operator(data, ...) : 2, estimiter = operator() : 3, fit = 4;
            "logic_operations": 'PCA',
            "physic_operations": ["PCA"],
            "params": ["n_components", "copy", "whiten", "svd_solver", "tol", "iterated_power", "random_state"],
        },
    }
}
with open('config.ini','w') as f:
    config.write(f)
