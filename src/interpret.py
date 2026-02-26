import matplotlib.pyplot as plt 

def plot_feature_importance(model,feature_names):

    importance = model.feature_importances_

    sorted_idx = importance.argsort()

    plt.figure(figsize=(10,6))
    plt.barh(range(len(sorted_idx)),importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()