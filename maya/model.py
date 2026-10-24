"""The one place that creates the Bedrock chat model."""

from langchain_aws import ChatBedrockConverse

from maya.config import MODEL_ID, REGION


def get_model(**kwargs) -> ChatBedrockConverse:
    """Create the Bedrock model that every step uses.

    Temperature 0 makes the model pick its most likely answer, so repeated runs vary
    less.
    """

    return ChatBedrockConverse(
        model=MODEL_ID,
        region_name=REGION,
        temperature=0,
        **kwargs,
    )
