import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import RejectOptionClassification, EqOddsPostprocessing
from aif360.algorithms.inprocessing import MetaFairClassifier, AdversarialDebiasing

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from src.data_handler import DataHandler
from src.model_trainer import ModelTrainer
from src.utils import load_params, load_metadata, ensure_dir, setup_mlflow, log_params, log_metrics
from sklearn.metrics import mean_squared_error

class FairnessEngine:
    def __init__(self):
        self.params = load_params()
        self.meta = load_metadata()
        self.data_handler = DataHandler()
        self.trainer = ModelTrainer()
        
    def apply_mitigation(self):
        """Apply fairness mitigation techniques"""
        print("⚖️ Fairness Mitigation Pipeline")
        
        # Load data
        X, y, df, meta = self.data_handler.get_train_data()
        y_bin = (y >= y.median()).astype(int)
        
        # Track with MLflow
        mlflow = setup_mlflow()
        
        with mlflow.start_run(run_name="fairness_mitigation"):
            # Interactive mitigation
            X_current, y_current, weights = X.copy(), y_bin.copy(), np.ones(len(y_bin))
            applied_methods = []
            
            while True:
                print("\n⚖️ Choose Fairness Technique:")
                print("1. None  2. Reweighing  3. Disparate Impact")
                print("4. Reject Option  5. Equalized Odds")
                print("6. MetaFair  7. Adversarial Debiasing")
                print("Type 'train' to proceed to training")
                
                choice = input("Your choice: ").strip().lower()
                
                if choice == "train":
                    break
                
                # Select protected attribute
                if len(meta["protected"]) > 1:
                    print("\n🔒 Protected Attributes:")
                    for i, p in enumerate(meta["protected"], 1):
                        print(f"{i}. {p}")
                    attr_idx = int(input("Select attribute: ")) - 1
                    attr = meta["protected"][attr_idx]
                else:
                    attr = meta["protected"][0]
                
                print(f"\n🔍 Applying technique on '{attr}'...")
                
                # Apply mitigation
                result = self._apply_technique(choice, X_current, y_current, attr, df)
                
                if result:
                    X_current, y_current, weights = result
                    method_name = self._get_method_name(choice)
                    applied_methods.append(f"{method_name}({attr})")
                    print(f"✅ {method_name} applied successfully")
                else:
                    print("❌ Failed. Try another technique.")
            
            # Log applied methods
            log_params({"fairness_methods": ", ".join(applied_methods) if applied_methods else "None"})
            
            # Train fair model
            print("\n🧠 Training Fair Model...")
            model, model_name = self.trainer.train(X_current, y, weights)
            
            # Generate predictions
            df["pred_fair"] = model.predict(X_current)
            
            # Calculate metrics
            rmse = root_mean_squared_error(y, df["pred_fair"])
            print(f"📊 Fair Model RMSE: {rmse:.3f}")
            
            log_metrics({"fair_rmse": rmse, "model": model_name})
            
            # Save results
            ensure_dir(self.params["paths"]["fair_predictions"])
            df.to_csv(self.params["paths"]["fair_predictions"], index=False)
            self.trainer.save_model(model)
            
            print("✅ Fairness mitigation complete!")
            return model, df
    
    def _apply_technique(self, choice, X, y, attr, df_orig):
        """Apply selected fairness technique"""
        try:
            methods = {
                "1": self._none,
                "2": self._reweighing,
                "3": self._disparate_impact,
                "4": self._reject_option,
                "5": self._equalized_odds,
                "6": self._metafair,
                "7": self._adversarial
            }
            
            if choice in methods:
                return methods[choice](X, y, attr, df_orig)
            else:
                print("❌ Invalid choice")
                return None
                
        except Exception as e:
            print(f"⚠️ Error: {e}")
            return None
    
    def _get_aif_dataset(self, X, y, attr, df_orig):
        """Create AIF360 dataset"""
        X_copy = X.copy()
        X_copy["label"] = y
        X_copy["protected"] = (df_orig[attr] == df_orig[attr].mode()[0]).astype(int)
        return BinaryLabelDataset(
            df=X_copy, 
            label_names=["label"], 
            protected_attribute_names=["protected"]
        )
    
    def _none(self, X, y, attr, df_orig):
        """No mitigation"""
        return X, y, np.ones(len(y))
    
    def _reweighing(self, X, y, attr, df_orig):
        """Apply Reweighing"""
        dataset = self._get_aif_dataset(X, y, attr, df_orig)
        rw = Reweighing(
            privileged_groups=[{"protected": 1}],
            unprivileged_groups=[{"protected": 0}]
        )
        rw_data = rw.fit_transform(dataset)
        return X, y, rw_data.instance_weights
    
    def _disparate_impact(self, X, y, attr, df_orig):
        """Apply Disparate Impact Remover"""
        dataset = self._get_aif_dataset(X, y, attr, df_orig)
        dir_remover = DisparateImpactRemover(repair_level=1.0)
        repaired = dir_remover.fit_transform(dataset)
        return pd.DataFrame(repaired.features, columns=X.columns), y, np.ones(len(y))
    
    def _reject_option(self, X, y, attr, df_orig):
        """Apply Reject Option Classification"""
        dataset = self._get_aif_dataset(X, y, attr, df_orig)
        roc = RejectOptionClassification(
            privileged_groups=[{"protected": 1}],
            unprivileged_groups=[{"protected": 0}]
        )
        roc.fit(dataset, dataset)
        return X, y, np.ones(len(y))
    
    def _equalized_odds(self, X, y, attr, df_orig):
        """Apply Equalized Odds"""
        dataset = self._get_aif_dataset(X, y, attr, df_orig)
        eq_odds = EqOddsPostprocessing(
            privileged_groups=[{"protected": 1}],
            unprivileged_groups=[{"protected": 0}]
        )
        eq_odds.fit(dataset, dataset)
        return X, y, np.ones(len(y))
    
    def _metafair(self, X, y, attr, df_orig):
        """Apply MetaFairClassifier"""
        dataset = self._get_aif_dataset(X, y, attr, df_orig)
        model = MetaFairClassifier(sensitive_attr="protected", tau=0.8, seed=42)
        model.fit(dataset)
        return X, y, np.ones(len(y))
    
    def _adversarial(self, X, y, attr, df_orig):
        """Apply Adversarial Debiasing"""
        dataset = self._get_aif_dataset(X, y, attr, df_orig)
        sess = tf.Session()
        model = AdversarialDebiasing(
            privileged_groups=[{"protected": 1}],
            unprivileged_groups=[{"protected": 0}],
            scope_name='adv_debiasing',
            debias=True,
            sess=sess
        )
        model.fit(dataset)
        sess.close()
        return X, y, np.ones(len(y))
    
    def _get_method_name(self, choice):
        """Get method name from choice"""
        names = {
            "1": "None", "2": "Reweighing", "3": "Disparate Impact",
            "4": "Reject Option", "5": "Equalized Odds",
            "6": "MetaFair", "7": "Adversarial Debiasing"
        }
        return names.get(choice, "Unknown")