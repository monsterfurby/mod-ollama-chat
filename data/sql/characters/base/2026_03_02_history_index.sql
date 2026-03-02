-- Add index to timestamp for optimized history cleanup
ALTER TABLE mod_ollama_chat_history ADD INDEX idx_timestamp (timestamp);
