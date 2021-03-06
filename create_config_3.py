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
        ## add by lxt show 2020/10/18 ##
        "show": {
            "call_type": 4,
            "params": [
                "block"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "subplots": {
            "call_type": 4,
            "params": [
                "nrows",
                "ncols",
                "sharex",
                "sharey",
                "squeeze",
                "num",
                "subplot_kw",
                "gridspec_kw",
                "**fig_kw"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "barplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "data",
                "order",
                "hue_order",
                "estimator",
                "ci",
                "n_boot",
                "units.seed",
                "orient",
                "color",
                "palette",
                "saturation",
                "errcolor",
                "errwidth",
                "capsize",
                "dodge",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "countplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "data",
                "order",
                "hue_order",
                "estimator",
                "ci",
                "n_boot",
                "units.seed",
                "orient",
                "color",
                "palette",
                "saturation",
                "errcolor",
                "errwidth",
                "capsize",
                "dodge",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "distplot": {
            "call_type": 4,
            "params": [
                "a",
                "bins",
                "hist",
                "kde",
                "rug",
                "fit",
                "hist_kws",
                "rug_kws",
                "color",
                "vertical",
                "norm_hist",
                "axlabel",
                "ax"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "heatmap": {
            "call_type": 4,
            "params": [
                "data",
                "vmin",
                "vmax",
                "cmap",
                "center",
                "robust",
                "annot",
                "fmt",
                "annot_kws",
                "linewidths",
                "linecolor",
                "cbar",
                "cba_kws",
                "cbar_ax",
                "square",
                "xticklabels",
                "yticklabels",
                "mask",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "imshow": {
            "call_type": 4,
            "params": [
                "X",
                "cmap",
                "norm",
                "aspect",
                "interpolation",
                "alpha",
                "vmin",
                "vmax",
                "origin",
                "extent",
                "filternorm",
                "filterrad",
                "resample",
                "url",
                ""
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "scatterplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "size",
                "style",
                "data",
                "palette",
                "hue_order",
                "hue_norm",
                "sizes",
                "size_order",
                "size_norm",
                "markers",
                "style_order",
                "{x",
                "y}_bins",
                "units",
                "estimator",
                "ci",
                "n_boot",
                "alpha",
                "{x",
                "y}_jitter",
                "legend",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "subplot": {
            "call_type": 4,
            "params": [
                "*args",
                "projection",
                "ploar",
                "sharex",
                "sharey",
                "label"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "matshow": {
            "call_type": 4,
            "params": [
                "A",
                "fignum",
                ""
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "lineplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "size",
                "style",
                "data",
                "palette",
                "hue_order",
                "hue_norm",
                "sizes",
                "size_order",
                "size_norm",
                "dashes",
                "markers",
                "style_order",
                "units",
                "estimator",
                "ci",
                "n_boot",
                "seed",
                "sort",
                "err_style",
                "err_kws",
                "legend",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "kdeplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "shade",
                "vertical",
                "kernel",
                "bw",
                "gridsize",
                "cut",
                "clip",
                "legend",
                "shade_lowest",
                "cbar",
                "cbar_ax",
                "cbar_kws",
                "ax",
                "hue",
                "palette",
                "hue_order",
                "hue_norm",
                "multiple",
                "common_norm",
                "common_grid",
                "levels",
                "thresh",
                "bw_method",
                "bw_adjust",
                "log_scale",
                "color",
                "fill",
                "data",
                "keargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "Scatter": {
            "call_type": 4,
            "params": [
                "name",
                "visible",
                "showlegend",
                "legendgroup",
                "opacity",
                "mode",
                "ids",
                "x",
                "x0",
                "dx",
                "y",
                "y0",
                "dy",
                "text",
                "textposition",
                "texttemplate",
                "hovertext",
                "hoverinfo",
                "hovertemplate",
                "meta",
                "customdata",
                "xaxis",
                "yaxis",
                "orientation",
                "groupnorm",
                "stackgroup",
                "xperiod",
                "xperiodalignment",
                "xperiod0",
                "yperiod",
                "yperiodalignment",
                "yperiod0",
                "marker",
                "line",
                "textfont",
                "error_x",
                "error_y",
                "selectedpoints",
                "selected",
                "unselected",
                "cliponaxis",
                "connectgaps",
                "fill",
                "fillcolor",
                "hoverlabel",
                "hoveron",
                "stackgaps",
                "xclendar",
                "ycalendar",
                "uirvision"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "pairplot": {
            "call_type": 4,
            "params": [
                "data",
                "hue",
                "hue_order",
                "palette",
                "vars",
                "{x",
                "y}_vars",
                "diag_kind",
                "markers",
                "height",
                "corner",
                "dropna",
                "{plot",
                "diag",
                "grid}_kws"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "Bar": {
            "call_type": 4,
            "params": [
                "name",
                "visible",
                "showlegend",
                "legendgroup",
                "opacity",
                "ids",
                "x",
                "x0",
                "dx",
                "y",
                "y0",
                "dy",
                "base",
                "width",
                "offset",
                "text",
                "textposition",
                "texttemplate",
                "hovertext",
                "hoverinfo",
                "hovertemplate",
                "meta",
                "customdata",
                "xaxis",
                "yaxis",
                "orientation",
                "alignmentgroup",
                "offsetgroup",
                "xperiod",
                "xperiodalignment",
                "xperiod0",
                "yperiod",
                "yperiodalignment",
                "yperiod0",
                "marker",
                "textangle",
                "textfont",
                "error_x",
                "error_y",
                "selectedpoints",
                "selected",
                "unselected",
                "cliponaxis",
                "constraintext",
                "hoverlabel",
                "insidetextanchor",
                "insidetextfont",
                "outsidetextfont",
                "xcalendar",
                "ycalandar",
                "uirevision"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "FacetGrid": {
            "call_type": 4,
            "params": [
                "data",
                "row",
                "col",
                "hue",
                "col_wrap",
                "share{x",
                "y}",
                "height",
                "aspect",
                "palette",
                "{row",
                "col",
                "hue}_order",
                "hue_kws",
                "legend_out",
                "despine",
                "margin_titles",
                "{x",
                "y}lim",
                "subplot_kws",
                "gridspec_kes"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "catplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "data",
                "row",
                "col",
                "col_wrap",
                "estimator",
                "ci",
                "n_boot",
                "units",
                "seed",
                "order",
                "hue_order",
                "row_order",
                "col_order",
                "kind",
                "height",
                "aspect",
                "orient",
                "color",
                "palette",
                "legend",
                "legend_out",
                "share{x",
                "y}",
                "margin_titles",
                "facet_kws",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "jointplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "data",
                "kind",
                "color",
                "height",
                "ratio",
                "space",
                "dropna",
                "{x",
                "y}lim",
                "marginal_ticks",
                "{joint",
                "marginal}_kws",
                "hue",
                "palette",
                "hue_order",
                "hue_norm",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "pointplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "data",
                "order",
                "hue_order",
                "estimator",
                "ci",
                "n_boot",
                "units",
                "seed",
                "markers",
                "linestyles",
                "dodge",
                "join",
                "scale",
                "orient",
                "color",
                "palette",
                "errwidth",
                "capsize",
                "ax",
                ""
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "generate": {
            "call_type": 4,
            "params": [
                "text"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "histogram": {
            "call_type": 4,
            "params": [
                "a",
                "bins",
                "range",
                "normed",
                "weights",
                "density",
                ""
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "swarmplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "data",
                "order",
                "hue_order",
                "orient",
                "color",
                "palette",
                "size",
                "edgecolor",
                "linewidth",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "probplot": {
            "call_type": 4,
            "params": [
                "x",
                "sparams",
                "dist",
                "fit",
                "plot"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "lmplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "data",
                "hue",
                "col",
                "row",
                "palette",
                "col_wrap",
                "height",
                "aspect",
                "markers",
                "share{x",
                "y}",
                "{hue",
                "col",
                "row}_order",
                "legend",
                "legend_out",
                "x_estimator",
                "x_bins",
                "x_ci",
                "scatter",
                "fit_reg",
                "ci",
                "n_boot",
                "units",
                "seed",
                "order",
                "logistic",
                "lowess",
                "robust",
                "logx",
                "{x",
                "y}_partial",
                "truncate",
                "{x",
                "y}_jitter",
                "{scatter",
                "line}_kws",
                ""
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "Pie": {
            "call_type": 4,
            "params": [
                "data_frame",
                "names",
                "values",
                "colors",
                "color_discrete_sequence",
                "color_discrete_map",
                "hover_name",
                "hover_data",
                "custom_data",
                "labels",
                "title",
                "template",
                "width",
                "height",
                "opacity",
                "hole"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "stripplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "data",
                "order",
                "hue_order",
                "jitter",
                "dodge",
                "orient",
                "color",
                "palette",
                "size",
                "edgecolor",
                "linewidth",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "Layout": {
            "call_type": 4,
            "params": [
                "arg",
                "activeshape",
                "angularaxis",
                "annotations",
                "annotationdefaults",
                "autosize",
                "bargap",
                "bargroupgap",
                "barnorm",
                "boxgap",
                "boxgroupgap",
                "boxmode",
                "calendar",
                "clickmode",
                "coloraxis",
                "colorscale",
                "colorway",
                "datarevision",
                "direction",
                "dragmode",
                "editrevision",
                "extendfunnelareacolors",
                "extendpiecolors",
                "extendsunburstcolors",
                "extendtreemapcolors",
                "font",
                "etra"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "gcf": {
            "call_type": 4,
            "params": [
                ""
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "boxenplot": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "hue",
                "data",
                "order",
                "hue_order",
                "orient",
                "color",
                "palette",
                "saturation",
                "width",
                "dodge",
                "k_depth",
                "linewidth",
                "scale",
                "outlier_prop",
                "trust_alpha",
                "showfilters",
                "ax",
                "kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "head": {
            "call_type": 0,
            "params": [
                "n"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "plot": {
            "call_type": 0,
            "params": [
                "data",
                "x",
                "y",
                "kind",
                "ax",
                "subplots",
                "sharex",
                "sharey",
                "layout",
                "figsize",
                "use_index",
                "title",
                "grid",
                "legend",
                "style",
                "logy",
                "loglog",
                "xticks",
                "yticks",
                "xlim",
                "ylim",
                "xlabel",
                "rot",
                "fontsize",
                "colorbar",
                "position",
                "table",
                "yerr",
                "xerr",
                "stacked",
                "sort_columns",
                "secondary_y",
                "mark_right",
                "include_bool",
                "backend",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "scatter": {
            "call_type": 0,
            "params": [
                "x",
                "y",
                "s",
                "c",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "boxplot": {
            "call_type": 0,
            "params": [
                "column",
                "by",
                "ax",
                "fontsize",
                "rot",
                "grid",
                "figsize",
                "layout",
                "return_type",
                "backend",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "bar": {
            "call_type": 0,
            "params": [
                "x",
                "y",
                "color",
                "**kargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "hist": {
            "call_type": 0,
            "params": [
                "by",
                "bins",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "tail": {
            "call_type": 0,
            "params": [
                "n"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "pie": {
            "call_type": 0,
            "params": [
                "y",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "barh": {
            "call_type": 0,
            "params": [
                "x",
                "y",
                "color",
                "**kargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "pivot_table": {
            "call_type": 4,
            "params": [
                "values",
                "index",
                "columns",
                "aggfunc",
                "fill_value",
                "margins",
                "dropna",
                "margins_name",
                "observed"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "line": {
            "call_type": 0,
            "params": [
                "x",
                "y",
                "color",
                "**kargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "scatter_matrix": {
            "call_type": 4,
            "params": [
                "frame",
                "alpha",
                "figsize",
                "ax",
                "grid",
                "diagnal",
                "marker",
                "density_kwds",
                "hist_kwds",
                "range_padding",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "background_gradient": {
            "call_type": 4,
            "params": [
                "cmap",
                "low",
                "high",
                "axis",
                "subset",
                "text_color",
                "threshold",
                "vmin",
                "vmax"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "area": {
            "call_type": 0,
            "params": [
                "x",
                "y",
                "stacked",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "box": {
            "call_type": 0,
            "params": [
                "by",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "render": {
            "call_type": 4,
            "params": [
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },
        "density": {
            "call_type": 0,
            "params": [
                "bw_method",
                "ind",
                "**kwargs"
            ],
            "logic_operations": "Show",
            "physic_operations": "show"
        },    
        ## add by lxt show end 2020/10/18 ##

        ## add by lxt show config 2020/10/18##
        "figure": {
            "call_type": 4,
            "params": [
                "num",
                "figsize",
                "dpi",
                "facecolor",
                "edgecolor",
                "frameon",
                "FigureClass",
                "clear",
                ""
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "ylabel": {
            "call_type": 4,
            "params": [
                "ylabel",
                "labelpad",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "xlabel": {
            "call_type": 4,
            "params": [
                "xlabel",
                "labelpad",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "legend": {
            "call_type": 4,
            "params": [
                "handles",
                "labels"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "xticks": {
            "call_type": 4,
            "params": [
                "ticks",
                "labels",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "text": {
            "call_type": 4,
            "params": [
                "x",
                "y",
                "s",
                "fontdict",
                "withdash"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "grid": {
            "call_type": 4,
            "params": [
                "b",
                "which",
                "axis",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "tight_layout": {
            "call_type": 4,
            "params": [
                "pad",
                "h_pad",
                "w_pad",
                "rect"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "tick_params": {
            "call_type": 4,
            "params": [
                "axis",
                "reset",
                "which",
                "direction",
                "length",
                "width",
                "color",
                "pad",
                "labelsize",
                "colors",
                "zoder",
                "bottom",
                "top",
                "left",
                "right",
                "labelrotation",
                "grid_color",
                "grid_alpha",
                "grid_linewidth",
                "grid_linestyle"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "suptitle": {
            "call_type": 4,
            "params": [
                "t",
                "x",
                "y",
                "horizontalalignment",
                "ha",
                "verticalalignment",
                "va",
                "fontsize",
                "size",
                "fontweight",
                "weight",
                ""
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "xlim": {
            "call_type": 4,
            "params": [
                "**args",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "ylim": {
            "call_type": 4,
            "params": [
                "**args",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "annotate": {
            "call_type": 4,
            "params": [
                "text",
                "xy",
                "xytext",
                "xycoords",
                "textcoords",
                "arrowprops",
                "annotation_clip",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "yticks": {
            "call_type": 4,
            "params": [
                "ticks",
                "labels",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "set_size_inches": {
            "call_type": 4,
            "params": [
                "left",
                "bottom",
                "right",
                "top",
                "wspace",
                "hspace"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "subplots_adjust": {
            "call_type": 4,
            "params": [
                "left",
                "bottom",
                "right",
                "top",
                "wspace",
                "hspace"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        "setp": {
            "call_type": 4,
            "params": [
                "obj",
                "*args",
                "**kwargs"
            ],
            "logic_operations": "Show_config",
            "physic_operations": "show_config"
        },
        ## add by lxt show config end 2020/10/18##

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
