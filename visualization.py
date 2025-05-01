import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from captum.attr import visualization
import matplotlib 




def visualize_attributions(attributions_dict: Dict, top_k: Optional[int] = None, save_path: str = "attribution_bar_chart.png") -> None:
    """
    Visualize token attributions with a horizontal bar chart and save it.

    Args:
        attributions_dict: Attribution results
        top_k: Limit visualization to top k tokens by attribution magnitude
        save_path: Path to save the generated plot image.
    """
    tokens = attributions_dict["tokens"]
    attr_values = attributions_dict["attributions"]

    df = pd.DataFrame({
        "Token": tokens,
        "Attribution": attr_values,
        "Abs_Attribution": np.abs(attr_values)
    }).sort_values("Abs_Attribution", ascending=False)

    if top_k is not None and top_k < len(df):
        df = df.head(top_k)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.3))) 
    colors = ['green' if x > 0 else 'red' for x in df["Attribution"]]

    ax.barh(df["Token"], df["Attribution"], color=colors)
    ax.set_xlabel("Attribution Value")
    ax.set_ylabel("Token")
    ax.set_title(f"Token Attributions for Predicted Token: '{attributions_dict['predicted_token']}'")
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(alpha=0.3)

    ax.invert_yaxis()

    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Attribution plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close(fig)

    mean_delta = np.mean(attributions_dict["convergence_delta"])
    print(f"Mean convergence delta: {mean_delta:.4e}")
    
    


def visualize_text_attributions(text: str, attributions_dict: Dict) -> Any:
    """
    Visualize attributions directly within the text using color-coding.

    Args:
        text: Original input text (not strictly needed for visualization but good context)
        attributions_dict: Attribution results

    Returns:
        HTML visualization object (e.g., for display in Jupyter)
    """
    tokens = attributions_dict["tokens"]
    attr_values = attributions_dict["attributions"]

    token_list = []
    attr_list = []

    current_token = ""
    current_attr = 0.0
    is_first_token = True

    for token, attr in zip(tokens, attr_values):
        if token.startswith(" "):
            if not is_first_token:
                token_list.append(current_token)
                attr_list.append(current_attr)
            current_token = token[1:]
            current_attr = attr
            is_first_token = False
        else:
            current_token += token
            current_attr += attr 

    if current_token:
        token_list.append(current_token)
        attr_list.append(current_attr)

    attr_list_numeric = [float(a) for a in attr_list]

    html = visualization.visualize_text([visualization.VisualizationDataRecord(
        word_attributions=attr_list_numeric,
        pred_prob=None, 
        pred_class=attributions_dict['predicted_token'],
        true_class=None, 
        attr_score=sum(attr_list_numeric), 
        raw_input_ids=token_list,
        convergence_score=attributions_dict.get('convergence_delta', [0.0])[0] 
    )])
    return html





def compare_attributions(all_results: List[Dict], target_tokens: List[str], save_path: str = "comparison_bar_chart.png") -> None:
    """
    Visualize comparison of attributions for different target tokens and save it.

    Args:
        all_results: List of attribution dictionaries
        target_tokens: List of target tokens corresponding to results
        save_path: Path to save the generated plot image.
    """
    num_targets = len(target_tokens)
    fig, axes = plt.subplots(num_targets, 1, figsize=(15, 6 * num_targets), squeeze=False) # Ensure axes is always 2D

    for i, (result, target_token) in enumerate(zip(all_results, target_tokens)):
        ax = axes[i, 0]

        tokens = result["tokens"]
        attrs = result["attributions"]

        idx = np.argsort(np.abs(attrs))[::-1][:10] 

        if len(idx) == 0:
            print(f"Warning: No attributions to plot for target '{target_token}'")
            continue

        plot_tokens = [tokens[j] for j in idx]
        plot_attrs = attrs[idx]

        colors = ['green' if x > 0 else 'red' for x in plot_attrs]

        ax.barh(plot_tokens, plot_attrs, color=colors)
        ax.set_xlabel("Attribution Value")
        ax.set_title(f"Attributions for Target Token: '{target_token}'")
        ax.grid(alpha=0.3)
        ax.invert_yaxis() 

    plt.tight_layout()

    try:
        plt.savefig(save_path)
        print(f"Comparison plot saved to: {save_path}")
    except Exception as e:
        print(f"Error saving comparison plot: {e}")

    plt.close(fig)

    token_values = {}
    for result in all_results:
        for token, attr in zip(result["tokens"], result["attributions"]):
            if token not in token_values:
                token_values[token] = []
            token_values[token].append(attr)
    
    token_variance = {}
    for token, values in token_values.items():
        if len(values) == len(target_tokens):
            token_variance[token] = np.var(values)
    
    top_variance_tokens = sorted(token_variance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTokens with highest attribution variance across targets:")
    for token, var in top_variance_tokens:
        print(f"'{token}': Variance = {var:.4f}")