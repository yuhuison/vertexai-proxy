"""
Thought Signature Storage Management (Firestore)
Used for storing and retrieving Gemini 3.0+ thought_signature

Uses Firestore for persistent storage, supporting Cloud Run's stateless architecture
"""
import base64
import time
from typing import Optional

from google.cloud import firestore

# Firestore client (lazy initialization)
_db: Optional[firestore.Client] = None

# Collection name
COLLECTION_NAME = "signatures"

# Cache expiration time (seconds) - 60 minutes
CACHE_TTL = 3600


def _get_db() -> firestore.Client:
    """Get Firestore client (lazy initialization)"""
    global _db
    if _db is None:
        # Use named database (not default)
        _db = firestore.Client(database="thought-signture-cache")
    return _db


def store_thought_signature(tool_call_id: str, thought_signature: bytes) -> None:
    """
    Store thought_signature in Firestore
    
    Args:
        tool_call_id: Tool call ID
        thought_signature: Signature data (bytes)
    """
    if not tool_call_id or not thought_signature:
        return
    
    try:
        db = _get_db()
        doc_ref = db.collection(COLLECTION_NAME).document(tool_call_id)
        
        # Convert bytes to base64 string for storage (Firestore does not directly support bytes)
        signature_b64 = base64.b64encode(thought_signature).decode('utf-8')
        
        doc_ref.set({
            "signature": signature_b64,
            "created_at": time.time(),
            "expires_at": time.time() + CACHE_TTL
        })
    except Exception as e:
        print(f"Firestore store error: {e}")


def get_thought_signature(tool_call_id: str) -> Optional[bytes]:
    """
    Get thought_signature from Firestore
    
    Args:
        tool_call_id: Tool call ID
        
    Returns:
        Signature data (bytes) or None
    """
    if not tool_call_id:
        return None
    
    try:
        db = _get_db()
        doc_ref = db.collection(COLLECTION_NAME).document(tool_call_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            expires_at = data.get("expires_at", 0)
            
            # Check if expired
            if time.time() < expires_at:
                signature_b64 = data.get("signature")
                if signature_b64:
                    return base64.b64decode(signature_b64)
            else:
                # Expired, delete document
                doc_ref.delete()
        
        return None
    except Exception as e:
        print(f"Firestore get error: {e}")
        return None


def cleanup_expired() -> int:
    """
    Cleanup expired signatures (can be called periodically)
    
    Returns:
        Number of documents deleted
    """
    try:
        db = _get_db()
        current_time = time.time()
        
        # Query expired documents
        expired_docs = db.collection(COLLECTION_NAME)\
            .where("expires_at", "<", current_time)\
            .limit(100)\
            .stream()
        
        count = 0
        for doc in expired_docs:
            doc.reference.delete()
            count += 1
        
        return count
    except Exception as e:
        print(f"Firestore cleanup error: {e}")
        return 0
