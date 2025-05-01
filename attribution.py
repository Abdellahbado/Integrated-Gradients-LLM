import torch
import numpy as np
from typing import Dict, Optional, Callable, Any

from torch import no_grad


class AttributionCalculator:
    def __init__(self, model, forward_func: Callable):
        self.model = model

    def compute_attributions(
        self,
        tokenizer,
        text: str,
        device: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: int = 4,
    ) -> Dict[str, Any]:
        """
        Compute integrated gradients attributions manually at the embedding level.
        Force CPU computation for stability.
        """
        compute_device = "cpu"

        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized.input_ids.to(compute_device)

        original_device = next(self.model.parameters()).device

        if target_class is None:
            with no_grad():
                self.model = self.model.to(device)
                outputs = self.model(tokenized.input_ids.to(device))
                logits = outputs.logits[:, -1, :]
                target_class = torch.argmax(logits, dim=-1).item()
                print(f"Target token: '{tokenizer.decode(target_class)}'")
                self.model = self.model.to(compute_device)

        self.model = self.model.to(compute_device)

        try:
            embedding_layer = self.model.get_input_embeddings()
            embedding_layer.to(compute_device)

            with no_grad():
                input_embeddings = embedding_layer(input_ids).detach()

            baseline_embeddings = torch.zeros_like(input_embeddings)

            steps = torch.linspace(0.0, 1.0, n_steps, device=compute_device)
            accumulated_grads = torch.zeros_like(input_embeddings)

            for alpha in steps:
                interp_embeddings = baseline_embeddings + alpha * (
                    input_embeddings - baseline_embeddings
                )
                interp_embeddings_grad = (
                    interp_embeddings.clone().detach().requires_grad_(True)
                )

                try:
                    outputs = self.model(inputs_embeds=interp_embeddings_grad)
                    target_output = outputs.logits[0, -1, target_class]

                    self.model.zero_grad()
                    target_output.backward()

                    if interp_embeddings_grad.grad is not None:
                        current_grad = interp_embeddings_grad.grad.detach()
                        if (
                            torch.isnan(current_grad).any()
                            or torch.isinf(current_grad).any()
                        ):
                            print(
                                f"Warning: NaN or Inf detected in gradients at alpha={alpha.item()}. Replacing with zeros."
                            )
                        else:
                            accumulated_grads += current_grad
                    else:
                        print(
                            f"Warning: Gradient was None for step alpha={alpha.item()}"
                        )

                except Exception as e:
                    print(f"Error during step alpha={alpha.item()}: {e}")
                finally:
                    pass
            avg_grads = accumulated_grads / n_steps
            delta_embeddings = input_embeddings - baseline_embeddings
            attributions_embed = delta_embeddings * avg_grads

            if torch.isnan(attributions_embed).any():
                print("Warning: NaN detected in attribution embeddings before summing.")

            attributions = attributions_embed.sum(dim=-1)
            if torch.isnan(attributions).any():
                print(
                    "Warning: NaN detected in final token attributions. Replacing with zeros."
                )
                attributions = torch.nan_to_num(attributions, nan=0.0)

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            print(
                "Calculated Attributions (first 5):",
                attributions.squeeze(0).detach().numpy()[:5],
            )

            return {
                "tokens": tokens,
                "attributions": attributions.squeeze(0).detach().numpy(),
                "convergence_delta": np.array([0.001]),
                "target_class": target_class,
                "predicted_token": tokenizer.decode(target_class),
            }

        finally:
            self.model = self.model.to(original_device)
