CREATE TABLE IF NOT EXISTS `mod_ollama_chat_important_bot_backgrounds` (
  `bot_guid` BIGINT NOT NULL PRIMARY KEY,
  `personality_key` VARCHAR(64) NOT NULL,
  `mbti` VARCHAR(4) NOT NULL,
  `background` TEXT NOT NULL,
  `generated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
