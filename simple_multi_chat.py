import time
from cat.memory.vector_memory_collection import VectorMemoryCollection
from cat.mad_hatter.decorators import hook
from typing import Dict, List
from pydantic import BaseModel
from cat.looking_glass.stray_cat import StrayCat
from cat.log import log


class MemoryPointBase(BaseModel):
    content: str
    metadata: Dict = {}


class MemoryPoint(MemoryPointBase):
    id: str
    vector: List[float]


@hook  # default priority = 1
def before_cat_sends_message(message, cat: StrayCat):
    """
    Hook executed just before the assistant sends a message to the user.

    This function performs two key operations to update memory state:

    1. **Episodic Memory Update:**
       - It retrieves the most recent memory point in the "episodic" collection
         that matches the current `user_id`, `chat_id`, and user message `text`.
       - If a matching point is found, it updates its metadata with the assistant's
         reply (`bot` field) and overwrites the payload.

    2. **Chat Metadata Update:**
       - It fetches the corresponding "chat" memory point (by `chat_id`), updates
         its `last_update` timestamp, and overwrites the payload.

    These updates ensure that both short-term context (episodic memory) and long-term
    conversation state (chat metadata) are synchronized before the reply is sent.
    """

    ensure_chat_collection_exists(cat)

    # - Start auto-naming logic -
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
        vector_memory = cat.memory.vectors
        chat_id = cat.working_memory.chat_id

        # Retrieve chat details
        chat_point = vector_memory.vector_db.retrieve(
            collection_name="chat",
            ids=[chat_id],
            with_payload=True
        )[0]
        chat_metadata = chat_point.payload.get("metadata", {})
        chat_name = chat_metadata.get("name", "")

        # If the chat name is the default one, it's the first interaction.
        # Let's generate a meaningful name.
        if chat_name == settings["default_chat_name"]:
            user_text = cat.working_memory.user_message_json["text"]
            bot_text = message.text

            # Meta-prompt to ask the LLM to summarize the conversation
            prompt = f"""
            Summarize the following conversation with a short, descriptive title of 5 words or less.
            Respond only with the title, without any introductory text, explanation, or quotes.

            Conversation:
            - User: "{user_text}"
            - Bot: "{bot_text}"
            """

            generated_title = cat.llm(prompt)

            # Clean up the generated title
            if generated_title:
                clean_title = generated_title.strip().strip('"').strip("'")
                if clean_title:
                    # Update the chat name in the vector DB
                    updated_metadata = chat_metadata.copy()
                    updated_metadata["name"] = clean_title
                    vector_memory.vector_db.overwrite_payload(
                        collection_name="chat",
                        payload={"page_content": chat_point.payload.get("page_content", " "), "metadata": updated_metadata},
                        points=[chat_point.id],
                    )
    except Exception as e:
        log.error(f"Error during chat auto-naming: {e}")
    # - End auto-naming logic -

    vector_memory = cat.memory.vectors
    collection = vector_memory.collections.get("episodic")
    text = cat.working_memory.user_message_json.text

    query_filter = collection._qdrant_filter_from_dict(
        {"chat_id": cat.working_memory.chat_id, "user_id": cat.user_id, "text": text})
    points = vector_memory.vector_db.scroll(
        collection_name="episodic",
        scroll_filter=query_filter,
        with_payload=True,
        with_vectors=False,
        limit=10,
    )[0]
    if not points:
       return message
    sorted_points = sorted(points, key=lambda p: p.payload["metadata"]["when"], reverse=True)
    thePoint = sorted_points[0]

    metadata = thePoint.payload.get("metadata", "")
    page_content = thePoint.payload.get("page_content", "")
    if page_content is None:
        page_content = ""
    updated_metadata = metadata
    bot = message.text
    updated_metadata["bot"] = bot
    vector_memory.vector_db.overwrite_payload(
        collection_name="episodic",
        payload={"page_content": page_content, "metadata": updated_metadata},
        points=[thePoint.id],
    )

    chat = vector_memory.vector_db.retrieve(
        collection_name="chat",
        ids=[cat.working_memory.chat_id],
        with_payload=True
    )[0]
    existing_metadata = chat.payload.get("metadata", {})
    updated_metadata = existing_metadata.copy()
    updated_metadata["last_update"] = time.time()
    vector_memory.vector_db.overwrite_payload(
        collection_name="chat",
        payload={
            "page_content": chat.payload.get("page_content", " "),
            "metadata": updated_metadata
        },
        points=[chat.id],
    )
    return message


@hook
def fast_reply(fast_reply, cat):
    """
    Hook executed to ensure that a chat context (`chat_id`) exists for the current interaction.

    If no `chat_id` is present in the incoming user message:
    - It looks for an existing vector memory point in the "chat" collection
      associated with the current user and not marked as deleted.
    - If found, it reuses the existing point (if the name is default) and sets its ID as the current `chat_id`.
    - If not found, it creates a new point with minimal metadata (`name=default`)
      and stores it in the vector database, initializing a new chat context.

    Finally, the `chat_id` is stored in `cat.working_memory` for further processing.
    This ensures that all messages are tied to a persistent, user-specific chat thread.
    """

    ensure_chat_collection_exists(cat)

    if "chat_id" not in cat.working_memory.user_message_json:
        vector_memory = cat.memory.vectors
        collection = vector_memory.collections.get("chat")
        settings = cat.mad_hatter.get_plugin().load_settings()
        default_chat_name = settings["default_chat_name"]
        qdrant_payload_metadata = {
            "source": cat.user_id,
            "when": time.time(),
            "content": "",
        }
        qdrant_payload_metadata.update({
          "name" : default_chat_name,
          "deleted" : False
        })
        if "deleted" not in qdrant_payload_metadata:
            qdrant_payload_metadata["deleted"] = False
        content_for_embedding = qdrant_payload_metadata.get("content")
        if not content_for_embedding or not str(content_for_embedding).strip():
            content_for_embedding = qdrant_payload_metadata["name"]
        qdrant_payload_metadata["content"] = content_for_embedding
        query_filter = collection._qdrant_filter_from_dict({"source": cat.user_id})
        points = vector_memory.vector_db.scroll(
            collection_name="chat",
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=10000
        )[0]
        havechat=False
        if points:
            for p in points:
                existing_point_metadata = p.payload.get("metadata", {})
                if existing_point_metadata.get("deleted", False):
                    continue
                metadata = p.payload.get("metadata", {})
                if metadata.get("name") != default_chat_name:
                    continue
                else:
                    cat.working_memory.user_message_json["chat_id"] = p.id
                    havechat = True
        if not havechat:
            embedding = cat.embedder.embed_query(content_for_embedding)
            qdrant_point = vector_memory.collections["chat"].add_point(
                content=content_for_embedding,
                vector=embedding,
                metadata=qdrant_payload_metadata
            )
            log.info(f"Chat created (first chat): {qdrant_point.payload['metadata']}")
            log.info(f"Chat created (new chat added): {qdrant_point.id}")
            point = MemoryPoint(
                metadata=qdrant_point.payload["metadata"],
                content=qdrant_point.payload["page_content"],
                vector=qdrant_point.vector,
                id=qdrant_point.id,
            )
            cat.working_memory.user_message_json["chat_id"] = qdrant_point.id

    doc = cat.working_memory.user_message_json.chat_id
    cat.working_memory.chat_id = doc
    return fast_reply


def ensure_chat_collection_exists(cat: StrayCat):
    """
    Idempotent function to create the "chat" collection if it does not exist.
    This is called at the beginning of an interaction to make sure that the plugin
    is working also if activated at runtime, without requiring a restart.
    """
    try:
        new_collection_name = "chat"
        
        if new_collection_name in cat.memory.vectors.collections:
            return

        # Get embedder size
        embedder_size = len(cat.embedder.embed_query("hello world"))

        # Get embedder name
        if hasattr(cat.embedder, "model"):
            embedder_name = cat.embedder.model
        elif hasattr(cat.embedder, "repo_id"):
            embedder_name = cat.embedder.repo_id
        else:
            embedder_name = "default_embedder"

        new_collection = VectorMemoryCollection(
            client=cat.memory.vectors.vector_db,
            collection_name=new_collection_name,
            embedder_name=embedder_name,
            embedder_size=embedder_size,
        )

        cat.memory.vectors.collections[new_collection_name] = new_collection
        setattr(cat.memory.vectors, new_collection_name, new_collection)
        log.info(f"'{new_collection_name}' collection created and registered for the first time.")

    except Exception as e:
        log.error(f"Failed to create 'chat' collection: {e}")


@hook
def before_cat_stores_episodic_memory(doc, cat):
    """
    Hook executed before storing a document into the cat's episodic memory.

    This function enriches the message  with key metadata needed for
    identification, traceability, and later retrieval or analysis.

    Specifically, it performs the following:
    - Sets the `user_id` in the metadata based on the current cat instance.
    - Sets the `chat_id` from the most recent user message.
    - Ensures `page_content` is not None by assigning an empty string if missing.
    - Copies `page_content` into the `text` field of the metadata for consistency.
    - Initializes a `bot` field in the metadata as an empty string,
      to be filled in later with the assistant's response.

    The modified message is then returned and ready to be stored.
    """

    doc.metadata["user_id"] = cat.user_id
    doc.metadata["chat_id"] = cat.working_memory.user_message_json.chat_id
    if doc.page_content is None:
        doc.page_content = ""
    doc.metadata["text"] = doc.page_content
    doc.metadata["bot"] = ""
    return doc