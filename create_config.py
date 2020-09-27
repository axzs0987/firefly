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
    # "host": '10.77.70.123',
    "host": '127.0.0.1',
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
            "call_type": 4,
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
with open('config.ini','w') as f:
    config.write(f)
