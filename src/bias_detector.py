import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from src.utils import load_params, load_metadata, ensure_dir, setup_mlflow, log_metrics

class BiasDetector:
    def __init__(self):
        self.params = load_params()
        self.meta = load_metadata()
        
    def analyze(self):
        """Analyze bias in predictions"""
        print("🔍 Analyzing Bias...")
        
        # Load predictions
        df = pd.read_csv(self.params["paths"]["biased_predictions"])
        
        # Create distribution plot
        self._plot_distributions(df)
        
        # Calculate fairness metrics
        metrics_all = {}
        
        for attr in self.meta["protected"]:
            print(f"\n📊 Analyzing attribute: {attr}")
            metrics = self._calculate_fairness_metrics(df, attr)
            metrics_all[attr] = metrics
            
            # Plot group rates
            self._plot_group_rates(df, attr)
            
            # Print suggestions
            self._print_suggestions(metrics)
        
        # Save metrics
        ensure_dir(self.params["paths"]["metadata"])
        with open("outputs/bias_metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)
        
        # Log to MLflow
        mlflow = setup_mlflow()
        with mlflow.start_run(run_name="bias_detection"):
            for attr, metrics in metrics_all.items():
                log_metrics({f"{attr}_{k}": v for k, v in metrics.items()})
        
        print("\n✅ Bias analysis complete")
        return metrics_all
    
    def _calculate_fairness_metrics(self, df, attr):
        """Calculate fairness metrics for given attribute"""
        # Binarize labels
        df_copy = df.copy()
        df_copy["label"] = (df_copy["actual"] >= df_copy["actual"].median()).astype(int)
        df_copy["pred"] = (df_copy["pred_biased"] >= df_copy["pred_biased"].median()).astype(int)
        df_copy["protected"] = (df_copy[attr] == df_copy[attr].unique()[0]).astype(int)
        
        # Encode categorical columns
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                df_copy[col] = df_copy[col].astype("category").cat.codes
        
        # Create AIF360 dataset
        dataset = BinaryLabelDataset(
            df=df_copy,
            label_names=["label"],
            protected_attribute_names=["protected"]
        )
        
        # Calculate metrics
        priv_groups = [{"protected": 1}]
        unpriv_groups = [{"protected": 0}]
        
        dataset_metrics = BinaryLabelDatasetMetric(
            dataset, privileged_groups=priv_groups, unprivileged_groups=unpriv_groups
        )
        
        clf_metrics = ClassificationMetric(
            dataset, dataset, 
            unprivileged_groups=unpriv_groups, 
            privileged_groups=priv_groups
        )
        
        return {
            "disparate_impact": dataset_metrics.disparate_impact(),
            "statistical_parity_diff": dataset_metrics.statistical_parity_difference(),
            "equal_opportunity_diff": clf_metrics.equal_opportunity_difference(),
            "average_odds_diff": clf_metrics.average_odds_difference(),
            "theil_index": clf_metrics.theil_index(),
            "consistency": clf_metrics.consistency().mean(),
            "fpr_diff": clf_metrics.false_positive_rate_difference(),
            "fnr_diff": clf_metrics.false_negative_rate_difference()
        }
    
    def _plot_distributions(self, df):
        """Plot prediction distributions"""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df["actual"], label="Actual", fill=True, color="black")
        sns.kdeplot(df["pred_biased"], label="Biased Prediction", fill=True, color="red")
        plt.legend()
        plt.title("Prediction Distribution")
        plt.tight_layout()
        plt.savefig("outputs/prediction_distribution.png")
        plt.close()
        print("✅ Saved prediction_distribution.png")
    
    def _plot_group_rates(self, df, attr):
        """Plot group treatment rates"""
        df_copy = df.copy()
        df_copy["label"] = (df_copy["actual"] >= df_copy["actual"].median()).astype(int)
        
        group_rates = df_copy.groupby(attr)["label"].mean()
        
        plt.figure(figsize=(8, 6))
        group_rates.plot(kind="bar", color=["gray", "blue"])
        plt.title(f"Selection Rate by Group ({attr})")
        plt.ylabel("Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"outputs/group_treatment_rates_{attr}.png")
        plt.close()
        print(f"✅ Saved group_treatment_rates_{attr}.png")
    
    def _print_suggestions(self, metrics):
        """Print fairness improvement suggestions"""
        print("\n💡 Suggestions:")
        
        if abs(1 - metrics["disparate_impact"]) > 0.2:
            print(f"- Disparate Impact = {metrics['disparate_impact']:.3f} → Try Reweighing")
        
        if abs(metrics["statistical_parity_diff"]) > 0.1:
            print(f"- Statistical Parity Diff = {metrics['statistical_parity_diff']:.3f} → Try Disparate Impact Remover")
        
        if abs(metrics["equal_opportunity_diff"]) > 0.1:
            print(f"- Equal Opportunity Diff = {metrics['equal_opportunity_diff']:.3f} → Try Equalized Odds")
        
        if abs(metrics["average_odds_diff"]) > 0.1:
            print(f"- Average Odds Diff = {metrics['average_odds_diff']:.3f} → Try Adversarial Debiasing")