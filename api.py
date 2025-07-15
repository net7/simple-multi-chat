import time
from typing import Dict, List

from cat.auth.connection import HTTPAuth
from cat.auth.permissions import AuthPermission, AuthResource
from cat.convo.messages import CatMessage, UserMessage
from cat.looking_glass.stray_cat import StrayCat
from cat.mad_hatter.decorators import endpoint
from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel
from qdrant_client.http.models import PointIdsList
from qdrant_client.models import FieldCondition, Filter, MatchValue
from .simple_multi_chat import ensure_chat_collection_exists
from cat.log import log


class MemoryPointBase(BaseModel):
    content: str
    metadata: Dict = {}


# TODOV2: annotate all endpoints and align internal usage (no qdrant PointStruct, no langchain Document)
class MemoryPoint(MemoryPointBase):
    id: str
    vector: List[float]

@endpoint.post(path="/createChat", prefix="")
async def create_chat(
        request: Request,
        metadata: Dict = {},
        cat: StrayCat = Depends(HTTPAuth(AuthResource.MEMORY, AuthPermission.READ)),
) -> MemoryPoint:
    """
    Endpoint to create a new chat session for the authenticated user, enforcing a configurable chat limit.

    Key points:
    - Receives optional metadata from the frontend to customize the new chat, including content and additional nested metadata.
    - Automatically sets default metadata fields such as 'source' (user ID), timestamp ('when'), 'deleted' flag (False), and a default chat name if not provided.
    - Determines the content to embed by prioritizing provided content; if empty, it falls back to the chat's name.
    - Queries the vector database to retrieve all existing chat points for the user, including deleted ones.
    - If no existing chats are found, it creates and returns the first chat point with embedded content and metadata.
    - If existing chats are present, it filters out deleted chats and checks the count of active chats.
    - Enforces a configurable limit on active chats per user. If the limit (`max_chats` setting) is reached, it raises an HTTP 400 error. The limit can be disabled by setting `max_chats` to -1.
    - If under the limit, it creates and returns a new chat point similarly to the first case.

    This endpoint ensures users do not exceed the allowed number of chat sessions, maintaining system limits and resource control, while allowing flexible metadata handling and proper embedding for efficient vector storage and retrieval.
    """

    ensure_chat_collection_exists(cat)

    settings = cat.mad_hatter.get_plugin().load_settings()
    max_chats = settings["max_chats"]
    default_chat_name = settings["default_chat_name"]

    vector_memory = cat.memory.vectors
    collection = vector_memory.collections.get("chat")

    nested_metadata_from_frontend = metadata.get("metadata", {})
    qdrant_payload_metadata = {
        "source": cat.user_id,
        "when": time.time(),
        "content": metadata.get("content", ""),
    }
    qdrant_payload_metadata.update(nested_metadata_from_frontend)
    if not qdrant_payload_metadata.get("name"):
        qdrant_payload_metadata["name"] = default_chat_name

    if "deleted" not in qdrant_payload_metadata:
        qdrant_payload_metadata["deleted"] = False

    #    Prioritize content from the frontend, then the chat's name if content is empty.
    content_for_embedding = qdrant_payload_metadata.get("content")
    if not content_for_embedding or not str(content_for_embedding).strip():
        content_for_embedding = qdrant_payload_metadata["name"]
    qdrant_payload_metadata["content"] = content_for_embedding  # Ensure 'content' is always present in saved metadata

    query_filter = collection._qdrant_filter_from_dict({"source": cat.user_id})
    points = vector_memory.vector_db.scroll(
        collection_name="chat",
        scroll_filter=query_filter,
        with_payload=True,
        with_vectors=False,
        limit=10000
    )[0]

    if not points:
        # Case 1: No existing chats for this user. Create the first one.
        embedding = cat.embedder.embed_query(content_for_embedding)

        qdrant_point = vector_memory.collections["chat"].add_point(
            content=content_for_embedding,
            vector=embedding,
            metadata=qdrant_payload_metadata
        )

        return MemoryPoint(
            metadata=qdrant_point.payload["metadata"],
            content=qdrant_point.payload["page_content"],
            vector=qdrant_point.vector,
            id=qdrant_point.id,
        )

    # Filter out deleted chats for the count
    deleted_chat_ids = set()
    for p in points:
        existing_point_metadata = p.payload.get("metadata", {})
        id = p.id
        if existing_point_metadata.get("deleted", False):
            deleted_chat_ids.add(id)

    matched_points = []
    for p in points:
        id = p.id
        if id in deleted_chat_ids:
            continue
        matched_points.append({
            "id": p.id,
            "metadata": p.payload.get("metadata", {}),
        })

    # Enforce chat limit only if max_chats is not -1 (which means unlimited)
    if max_chats != -1 and len(matched_points) >= max_chats:
        raise HTTPException(status_code=400, detail=f"Too many chats created, you can have a maximum of {max_chats}")
    else:
        # Case 2: User has existing chats, but under the limit or the limit is disabled. Create a new one.
        embedding = cat.embedder.embed_query(content_for_embedding)

        qdrant_point = vector_memory.collections["chat"].add_point(
            content=content_for_embedding,
            vector=embedding,
            metadata=qdrant_payload_metadata
        )
        return MemoryPoint(
            metadata=qdrant_point.payload["metadata"],
            content=qdrant_point.payload["page_content"],
            vector=qdrant_point.vector,
            id=qdrant_point.id,
        )



@endpoint.post(path="/memory/collections/{collection_id}/points/by_metadata_chat", prefix="")
async def get_points_metadata_only_chat(
        collection_id: str,
        cat: StrayCat = Depends(HTTPAuth(AuthResource.MEMORY, AuthPermission.READ)),
) -> Dict:
    """
    Endpoint to retrieve all active (non-deleted) chat records associated with the authenticated user from a specified collection.

    Functionality:
    - Validates that the requested collection exists; returns an error if it does not.
    - Constructs a query filter to find points where the 'metadata.source' matches the current user's ID and the 'metadata.deleted' flag is False, ensuring only active chats are returned.
    - Uses the vector database's scroll method to fetch all matching points, limited to 10,000 entries.
    - Returns a structured response containing the list of matched chat points with their metadata and the total count.
    - If no chats match the criteria, returns an empty list with a descriptive message.

    This endpoint is used to list all the user's existing chat sessions that have not been deleted, facilitating retrieval and management of conversation histories.
    """

    if collection_id == "chat":
        ensure_chat_collection_exists(cat)

    vector_memory = cat.memory.vectors
    collection = vector_memory.collections.get(collection_id)

    if not collection:
        raise HTTPException(
            status_code=400,
            detail={"error": "Collection does not exist."}
        )

    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=cat.user_id)
            ),
            FieldCondition(
                key="metadata.deleted",
                match=MatchValue(value=False)
            )
        ]
    )

    points = vector_memory.vector_db.scroll(
        collection_name=collection_id,
        scroll_filter=query_filter,
        with_payload=True,
        with_vectors=False,
        limit=10000
    )[0]

    if not points:
        return {
            "points": [],
            "count": 0,
            "message": "No points found matching metadata criteria"
        }

    matched_points = [{
        "id": p.id,
        "metadata": p.payload.get("metadata", {}),
    } for p in points]


    return {
        "points": matched_points,
        "count": len(matched_points),
    }


@endpoint.post(path="/memory/collections/{collection_id}/points/by_metadata_messages", prefix="")
async def get_points_metadata_only_message(
        request: Request,
        collection_id: str,
        metadata: Dict = {},
        cat: StrayCat = Depends(HTTPAuth(AuthResource.MEMORY, AuthPermission.READ)),
) -> Dict:
    """
    Endpoint to retrieve episodic memory messages from a specified collection, filtered by metadata.

    Functionality:
    - Accepts a collection ID and metadata filters (such as user_id and chat_id) to narrow down relevant memory points.
    - Queries the vector database to scroll through all points matching the filter criteria.
    - Returns messages sorted chronologically by their 'when' timestamp, representing the flow of conversation history.
    - Reconstructs the conversation by populating the bot's working memory history with alternating user and AI messages.
    - Handles the case where no matching points are found, returning an empty list with a descriptive message.

    This endpoint is essential for reconstructing past chat interactions and maintaining context across sessions.
    """

    vector_memory = cat.memory.vectors
    collection = vector_memory.collections.get(collection_id)

    if not collection:
        raise HTTPException(
            status_code=400,
            detail={"error": "Collection does not exist."}
        )

    query_filter = collection._qdrant_filter_from_dict({
        "user_id": cat.user_id,
        "chat_id": metadata.get("chat_id"),
    })
    points = vector_memory.vector_db.scroll(
        collection_name=collection_id,
        scroll_filter=query_filter,
        with_payload=True,
        with_vectors=False,
        limit=10000
    )[0]
    cat.working_memory.history = []

    if not points:
        return {
            "points": [],
            "count": 0,
            "message": "No points found matching metadata criteria"
        }

    matched_points = [{
        "id": p.id,
        "metadata": p.payload.get("metadata", {}),
    } for p in points]
    matched_points.sort(key=lambda x: x['metadata']['when'])
    
    for point in matched_points:
        metadata = point.get('metadata', {})

        if 'bot' in metadata and 'text' in metadata:
            bot_message = metadata['bot']
            message = metadata['text']
        else:
            bot_message = None
            message = ""
        if message != "":
            cat.working_memory.history.append(UserMessage(
                user_id=cat.user_id,
                who=cat.user_id,
                text=message
            ))
        if bot_message is not None:
            cat.working_memory.history.append(CatMessage(
                user_id=cat.user_id,
                who="AI",
                text=bot_message,
                why=None,
            ))
    return {
        "points": matched_points,
        "count": len(matched_points)
    }




@endpoint.delete(path="/delete_chat", prefix="")
async def del_chat(
        request: Request,
        chat_id: str,
        cat: StrayCat = Depends(HTTPAuth(AuthResource.MEMORY, AuthPermission.READ)),
) -> bool:
    """
    Endpoint that permanently removes a chat from the 'chat' vector collection.

    Workflow:
    - Attempts to delete the chat with the specified `chat_id` from the vector database.
    - If the operation succeeds, returns `True`.
    - If any exception occurs during the deletion process, logs the error and returns `False`.

    Note:
    - This operation is irreversible and removes the chat point entirely.
    - Does not verify ownership — it assumes upstream authentication ensures the user has proper rights.
    """

    ensure_chat_collection_exists(cat)

    try:
        vector_memory = cat.memory.vectors
        vector_memory.vector_db.delete(
            collection_name="chat",
            points_selector=PointIdsList(points=[chat_id]),
        )

        collection = vector_memory.collections.get("episodic")
        query_filter = collection._qdrant_filter_from_dict({
            "user_id": cat.user_id,
            "chat_id": chat_id
        })
        vector_memory.vector_db.delete(
            collection_name="episodic",
            points_selector=query_filter,
        )
        return True
    except Exception as e:
        log.error(e)
        return False

@endpoint.post(path="/memory/collections/points/changeNameChat", prefix="")
async def changeNameChat(
        chat_id: str,
        name: str,
        request: Request,
        cat: StrayCat = Depends(HTTPAuth(AuthResource.MEMORY, AuthPermission.READ)),
) -> bool:
    """
    Endpoint that allows a user to rename an existing chat.

    Workflow:
    - Retrieves the chat point from the "chat" vector collection using the given `chat_id`.
    - Verifies that the chat belongs to the current user by checking the `source` field in metadata.
    - If ownership is confirmed, updates the `name` field in the metadata with the new provided name.
    - Overwrites the payload in the vector database with the updated metadata.

    Returns:
    - `True` if the name change was successful.
    - `False` if the chat does not exist or does not belong to the user.

    This endpoint ensures chat metadata integrity by enforcing user ownership before making changes.
    """

    ensure_chat_collection_exists(cat)

    vector_memory = cat.memory.vectors
    point = vector_memory.vector_db.retrieve(
        collection_name="chat",
        ids=[chat_id],
        with_payload=True
    )[0]
    if point:
        existing_metadata = point.payload
        updated_metadata = existing_metadata["metadata"].copy()
        if updated_metadata["source"] != cat.user_id:
            return False
        updated_metadata["name"] = name
        vector_memory.vector_db.overwrite_payload(
            collection_name="chat",
            payload={
                "page_content": point.payload.get("page_content", " "),
                "metadata": updated_metadata},
            points=[point.id]
        )
        return True
    return False



@endpoint.post(path="/giveAll", prefix="")
async def giveAll(
        chat_id: str,
        request: Request,
        cat: StrayCat = Depends(HTTPAuth(AuthResource.MEMORY, AuthPermission.READ)),
):
    """
    Endpoint that returns all messages and metadata associated with a given chat ID.

    Main steps:
    - Retrieves the chat metadata from the "chat" vector collection using the provided `chat_id`.
    - Extracts all episodic memory points (messages) linked to the same `chat_id` and `user_id`.
    - Sorts the messages chronologically by their `when` timestamp.
    - Reconstructs the conversation history by appending each message (user and bot)
      into the `cat.working_memory.history` structure for further use.

    Returns a structured response containing:
    - A list of matched points with metadata.
    - The total count of messages.
    - The name of the chat associated with the `chat_id`.

    Useful for reconstructing past conversations or exporting complete chat threads.
    """

    ensure_chat_collection_exists(cat)

    vector_memory = cat.memory.vectors
    chat_point = vector_memory.vector_db.retrieve(
        collection_name="chat",
        ids=[chat_id],
        with_payload=True
    )[0]
    payload = chat_point.payload.get("metadata", {})
    chat_name = payload.get("name")
    collection = vector_memory.collections.get("episodic")
    query_filter = collection._qdrant_filter_from_dict({
        "user_id": cat.user_id,
        "chat_id": chat_id
    })
    points = vector_memory.vector_db.scroll(
        collection_name="episodic",
        scroll_filter=query_filter,
        with_payload=True,
        with_vectors=False,
        limit=10000
    )[0]
    cat.working_memory.history = []

    if not points:
        messages = {
            "points": [],
            "count": 0,
        }
    else:
        matched_points = [{
            "id": p.id,
            "metadata": p.payload.get("metadata", {}),
        } for p in points]
        matched_points.sort(key=lambda x: x['metadata']['when'])
        for point in matched_points:
            metadata = point.get('metadata', {})
            if 'bot' in metadata and 'text' in metadata:
                bot_message = metadata['bot']
                message = metadata['text']
            else:
                bot_message = None
                message = ""
            if message != "":
                cat.working_memory.history.append(UserMessage(
                    user_id=cat.user_id,
                    who=cat.user_id,
                    text=message
                ))
            if bot_message is not None:
                cat.working_memory.history.append(CatMessage(
                    user_id=cat.user_id,
                    who="AI",
                    text=bot_message,
                    why=None,
                ))
        messages = {
            "points": matched_points,
            "count": len(matched_points)
        }
    return {
        "Messages": messages,
        "Name": chat_name
    }