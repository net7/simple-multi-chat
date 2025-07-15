from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field


class MultiChatSettings(BaseModel):
    max_chats: int = Field(
        default=4,
        description="Maximum number of chats a user can create. Set to -1 to disable the limit.",
    )
    default_chat_name: str = Field(
        default="New Unnamed Chat",
        description="Default name for a new chat if not specified.",
    ) 

@plugin
def settings_model():
    """Returns the Pydantic model for the plugin's settings."""
    return MultiChatSettings