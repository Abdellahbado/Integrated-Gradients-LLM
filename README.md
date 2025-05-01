# LLM Integrated Gradients Attribution

A Python project to compute and visualize token-level attributions for Large Language Models (LLMs) using the Integrated Gradients method. This helps understand which input tokens most influence the model's prediction for the next token in a sequence.

## Key Features

*   Loads LLMs from Hugging Face (`transformers`).
*   Predicts the most likely next token for a given prompt.
*   Computes Integrated Gradients attributions at the embedding level.
*   Visualizes attributions for the predicted token using bar charts.
*   Compares attributions across different potential target tokens.
*   Saves generated plots to the `plots/` directory.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:** (Assuming a `requirements.txt` would be created)
    ```bash
    pip install -r requirements.txt 

    ```
3.  **Hugging Face Authentication:**
    *   For gated models (like Gemma), log in:
        ```bash
        huggingface-cli login
        ```
    *   Alternatively, create a `.env` file in the root directory with your token:
        ```
        HUGGINGFACE_TOKEN=your_hugging_face_token_here
        ```
    *   Make sure to accept the terms for any gated models you intend to use on the Hugging Face website.

## Usage

Run the main script to perform analysis on predefined prompts:

```bash
python main.py
```

*   The script will load the specified model (modify in `main.py` if needed).
*   It will analyze several prompts, predict the next token, compute attributions, and generate comparison plots.
*   Output plots will be saved in the `plots/` directory.

## File Structure

*   `main.py`: Main script to run the analysis pipeline.
*   `analyzer.py`: Contains the main `QwenAttributionAnalyzer` class orchestrating the process.
*   `attribution.py`: Implements the manual Integrated Gradients calculation logic.
*   `model_utils.py`: Handles model/tokenizer loading and next-token prediction.
*   `visualization.py`: Contains functions for generating plots.
*   `plots/`: Directory where output visualizations are saved.
