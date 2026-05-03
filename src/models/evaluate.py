import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve)
import shap
import joblib
import numpy as np

class ModelEvaluator:
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
    def full_report(self, X_test, y_test):
        """Generate complete evaluation report"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))
        
        print(f"\n ROC_AUC_Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        # ROC Curve
        self._plot_roc_curve(y_test, y_pred_proba)
        
        # Feature Importance
        self._plot_feature_importance()
        
        self._plot_shap(X_test)
            
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Died', 'Survived'],
                   yticklabels=['Died', 'Survived'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('reports/figures/confusion_matrix.png', dpi=150)
        plt.close()
        
    def _plot_roc_curve(self, y_true, y_pred_proba):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
        plt.fill_between(fpr, tpr, alpha=0.1, color='#2563eb')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('reports/figures/roc_curve.png', dpi=150)
        plt.close()
        
    def _plot_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color='#10b981')
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            plt.savefig('reports/figures/feature_importance.png', dpi=150)
            plt.close()
            

