import mlflow
from mlflow.tracking.client import MlflowClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
REGISTERED_MODEL_NAME = "SentimentAnalysisModel"
METRIC_NAME = "test_f1_macro"
MIN_IMPROVEMENT_THRESHOLD = 0.001 # New model must be slightly better

def promote_model():
    """
    Checks the latest model run against the current production model
    and promotes the latest if it performs better.
    """
    client = MlflowClient()
    
    # 1. Get the latest run ID that logged the model
    # We assume the most recent run for the experiment is the one to check.
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name("Ebay_Sentiment_Analysis_Project").experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("No runs found in the experiment.")
        return

    latest_run = runs[0]
    latest_run_id = latest_run.info.run_id
    latest_metric = latest_run.data.metrics.get(METRIC_NAME)

    if latest_metric is None:
        print(f"Latest run ({latest_run_id}) did not log metric '{METRIC_NAME}'.")
        return

    print(f"Latest Model ({latest_run_id}): {METRIC_NAME}={latest_metric:.4f}")

    # 2. Get the current Production model version (if one exists)
    try:
        production_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        current_prod_version = production_version[0] if production_version else None
    except Exception:
        # Handle case where the model hasn't been registered yet
        current_prod_version = None

    # 3. Decision Logic
    if current_prod_version:
        prod_run_id = current_prod_version.run_id
        prod_metric = client.get_run(prod_run_id).data.metrics.get(METRIC_NAME)
        prod_version = current_prod_version.version
        
        print(f"Production Model (V{prod_version}): {METRIC_NAME}={prod_metric:.4f}")

        if latest_metric > prod_metric + MIN_IMPROVEMENT_THRESHOLD:
            print(f"\n✅ New model (V{current_prod_version.version + 1}) is better! Promoting to Staging...")
            # Promote the latest version to Staging
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=current_prod_version.version + 1, # Use the next version number
                stage="Staging"
            )
            print("Successfully transitioned to Staging.")
        else:
            print(f"\n❌ New model ({latest_metric:.4f}) is not significantly better than Production ({prod_metric:.4f}).")
    else:
        print("\n⭐ No Production model found. Promoting latest version (V1) to Staging...")
        # Promote the first version (which should be 1, if logging was successful)
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=1,
            stage="Staging"
        )
        print("Successfully transitioned Version 1 to Staging.")


if __name__ == "__main__":
    # Note: If running this, ensure you run 'mlflow ui' locally first or configure
    # a remote MLFLOW_TRACKING_URI in your environment.
    promote_model()