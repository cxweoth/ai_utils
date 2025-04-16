import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor


class ValuePredictReport:
    def __init__(self, train_data_path, id_col, label, report_folder):
        self.train_data = pd.read_csv(train_data_path)
        self.id_col = id_col
        self.label = label
        self.report_folder = report_folder
        self.model_path = os.path.join(report_folder, "models")
        self.vp_report_path = os.path.join(report_folder, "vp_report.md")
        self.eval_metric = "mean_squared_error"

        # 移除 ID 欄位並保存
        self.train_id = self.train_data[self.id_col]
        self.train_data.drop(columns=[self.id_col], inplace=True)
        self.original_features = [col for col in self.train_data.columns if col != self.label]
        self.vp_report_lines = []

    def generate_report(self):
        print("Generating Report...")
        report_lines, recommended_features = self.generate_basic_part() 
        self.vp_report_lines.extend(report_lines)
        self.save_report()
        print("Report Saved at:", self.vp_report_path)

    def save_report(self):
        with open(self.vp_report_path, "w") as f:
            f.write("\n".join(self.vp_report_lines))

    def generate_basic_part(self):
        predictor = TabularPredictor(label=self.label, path=self.model_path, eval_metric=self.eval_metric)\
            .fit(self.train_data)

        leaderboard_df = predictor.leaderboard(self.train_data, silent=True)
        feature_importance_df = predictor.feature_importance(self.train_data)
        used_features = predictor.feature_metadata.get_features()
        unused_features = [f for f in self.original_features if f not in used_features]

        # SHAP 可視化
        X, _ = predictor.load_data_internal('train')
        lightgbm_model_name = next(m for m in predictor.model_names() if "LightGBM" in m)
        print(f"使用 SHAP 的模型：{lightgbm_model_name}")

        lgbm_model = predictor._trainer.load_model(lightgbm_model_name).model
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X)

        # SHAP summary plot
        shap_summary_plot_path = os.path.join(self.report_folder, "shap_summary_plot.png")
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_plot_path)

        # SHAP bar plot
        shap_bar_plot_path = os.path.join(self.report_folder, "shap_bar_plot.png")
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(shap_bar_plot_path)

        # 建立報告內容
        report_lines = [
            "# AutoML 預測任務報告",
            "",
            "## 🎯 預測目標",
            f"- 預測欄位：**{self.label}**",
            f"- 訓練資料筆數：{len(self.train_data)}",
            f"- 原始特徵數量：{len(self.original_features)}",
            "",
            "## 特徵分析",
            "### Top 30 重要特徵",
            feature_importance_df.head(30).to_markdown(index=True),
            "",
            "| 欄位 | 說明 |",
            "|------|------|",
            "| importance | 特徵重要性分數，表示該特徵對模型預測的影響程度 |",
            "| stddev | 特徵重要性的標準差，表示估計的不確定性 |",
            "| p_value | 統計顯著性檢驗的 p 值 |",
            "| n | 用於計算的重要樣本數 |",
            "| p99_high / p99_low | 99% 置信區間上下限 |",
            "",
            "> **置信區間說明**：範圍小表示估計精確；若不包含零，則具統計顯著性。",
            "",
            "### 🧠 SHAP 模型解釋（基於 LightGBM）",
            "#### 1. SHAP Summary Plot（點圖）",
            f"![summary]({shap_summary_plot_path})",
            "",
            "#### 2. SHAP Feature Importance（條狀圖）",
            f"![bar]({shap_bar_plot_path})",
            "",
            "> SHAP 可視化幫助了解每個特徵對預測的正負影響與重要程度。",
            "",
            "## Baseline 模型表現",
            "### 模型使用特徵",
            "```",
            ", ".join(used_features),
            "```",
            "",
            "### 模型未使用特徵",
            "```",
            ", ".join(unused_features),
            "```",
            "",
            "### 模型 Baseline Leaderboard",
            leaderboard_df.to_markdown(index=False),
            "",
            "| 欄位 | 說明 |",
            "|------|------|",
            "| model | 模型名稱 |",
            "| score_test / score_val | 負的 RMSE（越高越好） |",
            "| pred_time_* / fit_time | 預測與訓練耗時（秒） |",
            "| stack_level | 模型堆疊層級 |",
            "| can_infer | 是否可用於推論 |",
        ]

        # 推薦特徵
        recommended_features = self.recommend_features(feature_importance_df)

        report_lines.append(f"## 推薦特徵")
        report_lines.append(f"- 推薦特徵數量：{len(recommended_features)}")
        report_lines.append(f"- 推薦特徵：")
        report_lines.append(f"  ```")
        report_lines.append(f"  {', '.join(recommended_features)}")
        report_lines.append(f"  ```")

        return report_lines, recommended_features
    
    def recommend_features(self, feature_importance_df, top_k=30):
        # 根據 importance 選出前 top_k 特徵
        importance_top = feature_importance_df.head(top_k).index.tolist()
        return importance_top
