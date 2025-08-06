from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
import os

# Azure authentication
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
)

# Register model
model = ml_client.models.create_or_update(
    Model(
        path="outputs/model.pkl",  # This path should exist after training
        name="my-ml-model",        # You can name your model here
        description="Model registered from GitHub Action pipeline",
        type="custom_model",
        tags={"stage": "dev"}
    )
)

print(f"Model registered: {model.name} | Version: {model.version}")
