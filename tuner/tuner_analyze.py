# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap


pd.set_option("display.max_rows", 50)


def get_candidates(result, num: int = 5, band: float = 1.05):
    id2conf = result.get_id2config_mapping()
    all_runs = result.get_all_runs(only_largest_budget=False)
    all_ids = []
    all_configs = []
    all_infos = []
    for r in all_runs:
        all_ids.append(str(r.config_id))
        config = id2conf[r.config_id]["config"]
        all_configs.append(config)
        all_infos.append(r.info)
    df = pd.concat(
        [pd.DataFrame(all_ids), pd.DataFrame(all_configs), pd.DataFrame(all_infos)],
        axis=1,
    )
    df.rename(columns={0: "ID"}, inplace=True)
    # real Pareto points
    feats = ["rsmt", "congestion", "density"]
    undominated = (df[feats].values[:, None] >= df[feats].values).all(axis=2).sum(
        axis=1
    ) == 1
    true_paretos = df[undominated].nsmallest(num, "cost")
    # density should be deciding factor among low rsmt/congestion
    dfGood = df[
        (df["rsmt"] <= df["rsmt"].min() * band)
        & (df["congestion"] < df["congestion"].min() * band)
    ]
    feats = ["density"]
    undominated = (dfGood[feats].values[:, None] >= dfGood[feats].values).all(
        axis=2
    ).sum(axis=1) == 1
    fake_paretos = dfGood[undominated].nsmallest(num, "density")
    proposed_paretos = pd.concat([true_paretos, fake_paretos])
    proposed_paretos.drop_duplicates(inplace=True)
    return proposed_paretos, df


def plot_pareto(result, num, filename):
    pareto, df = get_candidates(result, num)
    x, y, z = df["congestion"].values, df["density"].values, df["rsmt"].values
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")
    ax.set_xlim3d(0.9 * x.min(), 1.1 * x.min())
    ax.set_ylim3d(0.9 * y.min(), 1.1 * y.min())
    ax.set_zlim3d(0.9 * z.min(), 1.1 * z.min())
    ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)
    colors = np.repeat(0, len(x))
    colors[list(pareto.index)] = 1
    ax.scatter3D(x, y, z, alpha=1.0, c=colors, marker="o", cmap=cm.coolwarm)
    ax.set_xlabel("congestion", fontsize=15)
    ax.set_ylabel("density", fontsize=15)
    ax.set_zlabel("rsmt", fontsize=15)
    ax.set_title("Pareto DREAMPlace", fontsize=15)
    ax.view_init(20, 70)
    plt.savefig(filename, dpi=800)


def analyze(df, classification=True):
    # df = pd.read_pickle(".dataframe.pkl")
    print(df.corr()["cost"])
    print(df["cost"].describe())

    # preprocess the dataframe
    df["convergent"] = df.apply(
        lambda row: int(row["cost"] < df["cost"].quantile(0.9)), axis=1
    )
    # df = df.loc[df["convergent"] == 1]
    # print(df["rsmt"].corr(df["hpwl"]))
    if "GP_optimizer" in df.columns:
        le_opt = preprocessing.LabelEncoder().fit(df["GP_optimizer"])
        df["GP_optimizer"] = le_opt.transform(df["GP_optimizer"])
        print(le_opt.classes_)
    if "GP_wirelength" in df.columns:
        le_wl = preprocessing.LabelEncoder().fit(df["GP_wirelength"])
        df["GP_wirelength"] = le_wl.transform(df["GP_wirelength"])
        print(le_wl.classes_)

    train, test = train_test_split(
        df, test_size=0.1, shuffle=True, stratify=df["convergent"]
    )
    features = list(
        set(df.columns)
        - set(
            [
                "convergent",
                "iteration",
                "objective",
                "overflow",
                "max_density",
                "cost",
                "rsmt",
                "hpwl",
                "congestion",
                "density",
                "ID",
            ]
        )
    )

    # default_params = {
    #     "eta": 0.3,
    #     "gamma": 0,
    #     "max_depth": 6,
    #     "min_child_weight": 1,
    #     "subsample": 1,
    #     "colsample_bytree": 1,
    #     "lambda": 1,
    #     "alpha": 0,
    # }

    # parameters = {
    #     "eta": [0.01, 0.015, 0.025, 0.05, 0.1],
    #     "gamma": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    #     "max_depth": [3, 5, 7, 9],
    #     "n_estimators": [100, 200, 500, 1000],
    #     "learning_rate": [0.1, 0.01, 0.05],
    #     "subsample": [0.5, 0.6, 0.7, 0.8],
    #     "min_child_weight": [1, 3, 5, 7],
    #     "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    # }

    params = {"max_depth": 8, "eta": 0.1, "nthread": 4}
    num_rounds = 500
    early_stopping_rounds = 20

    if classification:
        target = "convergent"
        metric = {"ams@0.15", "auc", "logloss", "error"}
        params["objective"] = "binary:logistic"
    else:
        target = "cost"
        metric = {"rmse"}
        params["objective"] = "reg:squarederror"

    Xtrain, ytrain = train[features], train[target]
    Xtest, ytest = test[features], test[target]
    dtrain = xgb.DMatrix(Xtrain, label=ytrain)
    dtest = xgb.DMatrix(Xtest, label=ytest)

    if classification:

        def fpreproc(dtrain, dtest, param):
            label = dtrain.get_label()
            ratio = float(np.sum(label == 0)) / np.sum(label == 1)
            param["scale_pos_weight"] = ratio
            return (dtrain, dtest, param)

        xgb.cv(
            params,
            dtrain,
            num_rounds,
            stratified=True,
            nfold=5,
            metrics=metric,
            seed=0,
            fpreproc=fpreproc,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        xgb.cv(
            params,
            dtrain,
            num_rounds,
            nfold=5,
            metrics=metric,
            seed=0,
            early_stopping_rounds=early_stopping_rounds,
        )

    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dtest, "test")],
        verbose_eval=100,
        early_stopping_rounds=early_stopping_rounds,
    )

    if classification:
        ypred = (model.predict(dtest) > 0.5).astype(int)
        print(metrics.classification_report(ytest, ypred))
    else:
        print(metrics.r2_score(ytest, model.predict(dtest)))

    # plot
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.clf()
    xgb.plot_importance(model)
    plt.title("xgboost.plot_importance(model)")
    plt.savefig("xgboost_importance.png", dpi=400)

    # Shapley values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xtrain)
    # pred = model.predict(dtrain, output_margin=True) # pred_contribs=True
    # np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
    plt.clf()
    shap.summary_plot(shap_values, Xtrain)
    # shap.plots.bar(shap_values.abs.mean(0))
    plt.savefig("shapley_summary.png", dpi=400)
    for name in Xtrain.columns:
        plt.clf()
        shap.dependence_plot(name, shap_values, Xtrain)
        plt.savefig(f"shapley_dep_{name}.png", dpi=400)
