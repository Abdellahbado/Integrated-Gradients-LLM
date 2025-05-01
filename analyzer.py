from typing import List, Dict, Optional, Tuple, Any
import torch
from captum.attr import IntegratedGradients

from model_utils import load_model, predict_next_token
from attribution import AttributionCalculator
from visualization import (
    visualize_attributions,
    visualize_text_attributions,
    compare_attributions,
)


class QwenAttributionAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen-7B", device: str = None):
        """
        Initialize the attribution analyzer for Qwen model.

        Args:
            model_name: The specific Qwen model variant to use
            device: Device to run the model on (auto-detected if None)
        """
        self.model, self.tokenizer, self.device = load_model(model_name, device)

        self.attribution_calculator = AttributionCalculator(
            self.model, self.forward_func
        )

    def forward_func(self, inputs):
        """Forward function for attribution that returns logits for next token prediction."""
        if inputs.dtype != torch.long and inputs.dtype != torch.int:
            inputs = inputs.round().to(dtype=torch.long)

        outputs = self.model(inputs)
        return outputs.logits[:, -1, :]

    def predict_next_token(self, text: str) -> Tuple[str, float, int]:
        """Predict the most likely next token, its probability, and its ID."""
        return predict_next_token(self.model, self.tokenizer, text, self.device)

    def compute_attributions(
        self,
        text: str,
        target_token_id: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: int = 4,
    ) -> Dict[str, Any]:
        """Compute attributions for a specific target token ID."""
        if target_token_id is None:
            _, _, predicted_token_id = self.predict_next_token(text)  # Unpack 3 values
            target_token_id = predicted_token_id
            print(
                f"No target_token_id provided, using predicted token ID: {target_token_id}"
            )
        else:
            print(f"Using provided target_token_id: {target_token_id}")

        attributions_dict = self.attribution_calculator.compute_attributions(
            tokenizer=self.tokenizer,
            text=text,
            device=self.device,
            target_class=target_token_id,
            n_steps=n_steps,
        )
        return attributions_dict

    def visualize_attributions(
        self,
        attributions_dict: Dict,
        top_k: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:  # Added save_path parameter
        """Visualize attributions using the function from visualization.py."""
        visualize_attributions(
            attributions_dict, top_k=top_k, save_path=save_path
        )  # Pass save_path here

    def visualize_text_attributions(self, text: str, attributions_dict: Dict) -> Any:
        """
        Visualize attributions directly within the text using color-coding.

        Args:
            text: Original input text
            attributions_dict: Attribution results from compute_attributions
        """
        return visualize_text_attributions(text, attributions_dict)

    def compare_attributions_for_targets(
        self,
        text: str,
        target_tokens: List[str],
        n_steps: int = 50,
        save_path: Optional[str] = None,
    ) -> None:
        """Compute and compare attributions for multiple target token strings."""
        all_results = []
        valid_target_tokens = []
        for target_token_str in target_tokens:
            target_token_ids = self.tokenizer.encode(
                target_token_str, add_special_tokens=False
            )

            if not target_token_ids:
                print(
                    f"Warning: Could not encode target token '{target_token_str}'. Skipping."
                )
                continue
            if len(target_token_ids) > 1:
                print(
                    f"Warning: Target '{target_token_str}' tokenizes into multiple IDs ({target_token_ids}). Using the first ID: {target_token_ids[0]}."
                )

            target_id = target_token_ids[0]
            valid_target_tokens.append(target_token_str)

            print(
                f"\nComputing attributions for target: '{target_token_str}' (ID: {target_id})"
            )
            try:
                attributions = self.compute_attributions(
                    text=text, target_token_id=target_id, n_steps=n_steps
                )
                attributions["target_token_str"] = target_token_str
                all_results.append(attributions)
            except Exception as e:
                print(
                    f"Error computing attributions for target '{target_token_str}': {e}"
                )

        if all_results:
            compare_attributions(
                all_results, target_tokens=valid_target_tokens, save_path=save_path
            )  
        else:
            print("No valid attribution results to compare.")
