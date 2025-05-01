import traceback
from analyzer import QwenAttributionAnalyzer

def analyze_prompt(analyzer: QwenAttributionAnalyzer, prompt: str, compare_targets: list = None, n_steps: int = 50, top_k_vis: int = 10):
    """Helper function to analyze a single prompt."""
    print("-" * 80)
    print(f"Analyzing Prompt: '{prompt}'")
    print("-" * 80)

    try:
        next_token_str, prob, next_token_id = analyzer.predict_next_token(prompt) 
        print(f"Predicted next token: '{next_token_str}' (ID: {next_token_id}) with probability {prob:.4f}")

        print(f"Computing attributions for the predicted token: '{next_token_str}' (ID: {next_token_id})...")
        attributions = analyzer.compute_attributions(
            prompt,
            target_token_id=next_token_id, 
            n_steps=n_steps,
        )
        attributions['predicted_token'] = next_token_str 

        print("Visualizing attributions for the predicted token...")
        vis_filename_pred = f"attribution_{prompt[:15].replace(' ', '_')}_pred_{next_token_str.strip().replace(' ', '_')}.png"
        analyzer.visualize_attributions(
            attributions,
            top_k=top_k_vis,
            save_path=vis_filename_pred 
        )
        print(f"Saved prediction attribution plot to: {vis_filename_pred}")

        if compare_targets:
            print(f"\nComparing attributions for target completions: {compare_targets}...")
            comp_filename = f"comparison_{prompt[:15].replace(' ', '_')}.png"
            analyzer.compare_attributions_for_targets(
                prompt,
                target_tokens=compare_targets, 
                n_steps=n_steps,
                save_path=comp_filename 
            )
            print(f"Saved comparison attribution plot to: {comp_filename}")


    except Exception as e:
        print(f"An error occurred analyzing prompt '{prompt}': {str(e)}")
        traceback.print_exc() 


def main():
    try:
        analyzer = QwenAttributionAnalyzer(model_name="Qwen/Qwen3-0.6B", device="mps") 
        print("Analyzer initialized successfully.")
        n_steps_for_analysis = 30 

        prompt1 = "The color of the sky is usually "
        compare1 = ["pink", "green", "red", "cloudy", "blue"] 
        analyze_prompt(analyzer, prompt1, compare_targets=compare1, n_steps=n_steps_for_analysis)

        prompt2 = "Steve Jobs was the CEO of "
        compare2 = ["Tesla", "Microsoft", "Google", "Apple"] 
        analyze_prompt(analyzer, prompt2, compare_targets=compare2, n_steps=n_steps_for_analysis)

        prompt3 = "A whale is not a "
        compare3 = ["fish", "mammal", "boat", "bird"] 
        analyze_prompt(analyzer, prompt3, compare_targets=compare3, n_steps=n_steps_for_analysis)

        prompt4 = "The movie was fantastic, truly inspiring. I felt "
        compare4 = ["sad", "happy", "bored", "inspired"] 
        analyze_prompt(analyzer, prompt4, compare_targets=compare4, n_steps=n_steps_for_analysis)


    except Exception as e:
        print(f"A critical error occurred during initialization or setup: {str(e)}")
        traceback.print_exc() 

if __name__ == "__main__":
    main()