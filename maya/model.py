"""The one place that creates the Bedrock chat model."""

from langchain_aws import ChatBedrockConverse

from maya.config import MODEL_ID, REGION


def get_model(model_id: str = MODEL_ID, **kwargs) -> ChatBedrockConverse:
    """Create a Bedrock model: MODEL_ID for every step, unless one asks for another.

    Temperature 0 makes the model pick its most likely answer, so repeated runs vary
    less.
    """

    return ChatBedrockConverse(
        model=model_id,
        region_name=REGION,
        temperature=0,
        **kwargs,
    )
