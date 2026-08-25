from utils.database import SessionLocal, ChatHistory

class UserMemory:
    def __init__(self, user_id: int):
        self.user_id = user_id

    def add_interaction(self, question: str, answer: str):
        db = SessionLocal()
        try:
            chat = ChatHistory(user_id=self.user_id, question=question, answer=answer)
            db.add(chat)
            db.commit()
        finally:
            db.close()

    def get_history(self, limit: int = 10):
        db = SessionLocal()
        try:
            # Get latest limit records and return in chronological order
            chats = db.query(ChatHistory).filter(ChatHistory.user_id == self.user_id).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
            return [{"question": c.question, "answer": c.answer} for c in reversed(chats)]
        finally:
            db.close()

    def get_history_string(self, limit: int = 5):
        history = self.get_history(limit)
        if not history:
            return "No previous conversation history."
        
        context = ""
        for item in history:
            context += f"User: {item['question']}\nAssistant: {item['answer']}\n\n"
        return context

# We don't instantiate a global memory anymore.
# Instances will be created per-user in Streamlit.