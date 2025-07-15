# Simple Multi Chat Plugin

A plugin for multi-chat management, built for the Cheshire Cat framework. 
It provides full support for chat creation, deletion, renaming, and retrieval, with automatic episodic memory tracking.

---

## 📦 How to Use

1. **Installation**
   - Add this folder to your plugins directory within the Cheshire Cat system.
   - Restart the Cat to apply the changes.

2. **Configuration**
   - You can customize the plugin settings from the admin interface.
   - `max_chats`: Set the maximum number of chats a user can create (-1 for unlimited).
   - `default_chat_name`: Define the default name for new, unnamed chats.

3. **Memory Architecture**
   - Chats are stored in the `chat` collection.
   - Messages are stored in the `episodic` collection.

---

## ✨ Features

- **Automatic Chat Naming**: New chats are automatically named based on the summary of the first interaction, so you don't have to name them manually.

---

## 🎨 Companion UI

A user-friendly Gradio interface is available to use this plugin. It provides a clean, ChatGPT-like experience for managing multiple conversations.

- **[Check it out here](https://github.com/net7/simple-multi-chat-ui)**

---

## 🔁 Available Endpoints

### 🟢 Chat Management

- `POST /createChat`  
  Creates a new chat. If `content` is empty, the chat `name` will be used for embedding. Custom metadata is supported.

- `DELETE /delete_chat?chat_id={chat_id}`  
  Permanently deletes a chat and all its messages based on its `chat_id`.

- `POST /memory/collections/points/changeNameChat?chat_id={chat_id}&name={new_name}`  
  Renames a chat. The requesting user must be the owner.

- `POST /memory/collections/chat/points/by_metadata_chat`  
  Retrieves all active (non-deleted) chats for the current user.

### 📨 Message Retrieval

- `POST /giveAll?chat_id={chat_id}`  
  Retrieves all messages and metadata associated with a given `chat_id`.  
  Messages are returned in chronological order.

- `POST /memory/collections/episodic/points/by_metadata_messages`  
  Returns all messages from the `episodic` collection for a specific chat. Requires `chat_id` in the request body.


## 📌 Important Notes

- If no `chat_id` is provided in a message, the system will use an existing default chat or create a new one for the user.
---

## ✅ Example Input Message

```json
{
  "text": "Hello, how are you?",
  "chat_id": "d91e4f3f-xxxx-xxxx-xxxx-5a7f0130fc9e"
}
```
