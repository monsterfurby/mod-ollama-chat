#include "mod-ollama-chat_config.h"
#include "Config.h"
#include "Log.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_personality.h"
#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_sentiment.h"
#include <fmt/core.h>
#include <atomic>
#include <fstream>
#include <sstream>
#include <thread>
#include <set>

// --------------------------------------------
// Distance/Range Configuration
// --------------------------------------------
float g_SayDistance = 30.0f;
float g_YellDistance = 100.0f;
float g_RandomChatterRealPlayerDistance = 40.0f;
float g_EventChatterRealPlayerDistance = 40.0f;

// --------------------------------------------
// Bot/Player Chatter Probability & Limits
// --------------------------------------------
// Per-channel-type reply chances
uint32_t g_PlayerReplyChance_Say = 90;
uint32_t g_BotReplyChance_Say = 10;
uint32_t g_PlayerReplyChance_Channel = 50;
uint32_t g_BotReplyChance_Channel = 5;
uint32_t g_PlayerReplyChance_Party = 90;
uint32_t g_BotReplyChance_Party = 10;
uint32_t g_PlayerReplyChance_Guild = 70;
uint32_t g_BotReplyChance_Guild = 5;

uint32_t g_MaxBotsToPick = 2;
uint32_t g_RandomChatterBotCommentChance = 5;
uint32_t g_RandomChatterMaxBotsPerPlayer = 2;
uint32_t g_EventChatterBotCommentChance = 15;
uint32_t g_EventChatterBotSelfCommentChance = 5;
uint32_t g_EventChatterMaxBotsPerPlayer = 2;

// --------------------------------------------
// Ollama LLM API Configuration
// --------------------------------------------
std::string g_OllamaUrl = "http://localhost:11434/api/generate";
std::string g_OllamaModel = "llama3.2:1b";
uint32_t g_OllamaNumPredict = 40;
float g_OllamaTemperature = 0.8f;
float g_OllamaTopP = 0.95f;
float g_OllamaRepeatPenalty = 1.1f;
uint32_t g_OllamaNumCtx = 0;
uint32_t g_OllamaNumThreads = 0;
std::string g_OllamaStop = "";
std::string g_OllamaSystemPrompt = "";
std::string g_OllamaSeed = "";

// --------------------------------------------
// OpenRouter API Configuration
// --------------------------------------------
bool g_UseOpenRouter = false;
std::string g_OpenRouterApiKey = "";
std::string g_OpenRouterUrl = "https://openrouter.ai/api/v1/chat/completions";
uint32_t g_OpenRouterMaxCallsPerPeriod = 0;
uint32_t g_OpenRouterPeriodLengthSeconds = 60;

// --------------------------------------------
// Concurrency/Queueing
// --------------------------------------------
uint32_t g_MaxConcurrentQueries = 0;

// --------------------------------------------
// Feature Toggles & Core Settings
// --------------------------------------------
bool g_Enable = true;
bool g_DisableRepliesInCombat = true;
bool g_EnableRandomChatter = true;
bool g_EnableEventChatter = true;
bool g_EnableRPPersonalities = false;
bool g_EnableWhisperReplies = false;
bool g_DebugEnabled = false;
bool g_DebugShowFullPrompt = false;

// --------------------------------------------
// Think Mode Support
// --------------------------------------------
bool g_ThinkModeEnableForModule = false;

// --------------------------------------------
// Random Chatter Timing
// --------------------------------------------
uint32_t g_MinRandomInterval = 45;
uint32_t g_MaxRandomInterval = 180;

// --------------------------------------------
// Conversation History Settings
// --------------------------------------------
uint32_t g_MaxConversationHistory = 5;
uint32_t g_ConversationHistorySaveInterval = 10;

// --------------------------------------------
// Prompt Templates
// --------------------------------------------
std::string g_RandomChatterPromptTemplate;
std::vector<std::string> g_RandomChatterPromptVariations;
std::vector<std::string> g_RandomChatterQuestionVariations;
std::string g_EventChatterPromptTemplate;
std::string g_ChatPromptTemplate;
std::string g_ChatExtraInfoTemplate;

// --------------------------------------------
// Personality and Prompt Data
// --------------------------------------------
std::unordered_map<uint64_t, std::string> g_BotPersonalityList;
std::unordered_map<std::string, std::string> g_PersonalityPrompts;
std::vector<std::string> g_PersonalityKeys;
std::vector<std::string> g_PersonalityKeysRandomOnly;
std::string g_DefaultPersonalityPrompt;

// --------------------------------------------
// Chat History Templates and Toggles
// --------------------------------------------
bool g_EnableChatHistory = true;
std::string g_ChatHistoryHeaderTemplate;
std::string g_ChatHistoryLineTemplate;
std::string g_ChatHistoryFooterTemplate;

// --------------------------------------------
// Chatbot Snapshot Template
// --------------------------------------------
bool g_EnableChatBotSnapshotTemplate = false;
std::string g_ChatBotSnapshotTemplate;

// --------------------------------------------
// Conversation History Store and Mutex
// --------------------------------------------
std::unordered_map<
    uint64_t, std::unordered_map<
                  uint64_t, std::deque<std::pair<std::string, std::string>>>>
    g_BotConversationHistory;
std::mutex g_ConversationHistoryMutex;
time_t g_LastHistorySaveTime = 0;

// --------------------------------------------
// Bot-Player Sentiment Tracking System
// --------------------------------------------
bool g_EnableSentimentTracking = true;
float g_SentimentDefaultValue = 0.5f; // Default sentiment value (0.5 = neutral)
float g_SentimentAdjustmentStrength =
    0.1f; // How much to adjust sentiment per message
uint32_t g_SentimentSaveInterval =
    10; // How often to save sentiment to DB (minutes)
std::string g_SentimentAnalysisPrompt =
    "Analyze the sentiment of this message: \"{message}\". Respond only with: "
    "POSITIVE, NEGATIVE, or NEUTRAL.";
std::string g_SentimentPromptTemplate =
    "Your relationship sentiment with {player_name} is {sentiment_value} "
    "(0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone and "
    "response.";

// In-memory sentiment storage and mutex
std::unordered_map<uint64_t, std::unordered_map<uint64_t, float>>
    g_BotPlayerSentiments;
std::mutex g_SentimentMutex;
time_t g_LastSentimentSaveTime = 0;

// --------------------------------------------
// RAG (Retrieval-Augmented Generation) System
// --------------------------------------------
bool g_EnableRAG = false;
std::string g_RAGDataPath = "rag/";
uint32_t g_RAGMaxRetrievedItems = 3;
float g_RAGSimilarityThreshold = 0.3f;
std::string g_RAGPromptTemplate;
uint32_t g_RAGMaxVocabularySize = 10000;

class OllamaRAGSystem;
OllamaRAGSystem *g_RAGSystem = nullptr;

// --------------------------------------------
// Global Shutdown Flag
// --------------------------------------------
std::atomic<bool> g_isShuttingDown(false);

// --------------------------------------------
// Important Bot Backgrounds
// --------------------------------------------
bool g_EnableImportantBotBackgrounds = true;
std::string g_ImportantBotBackgroundGenerationPrompt;
std::vector<std::string> g_ImportantBotTraitWords;
uint32_t g_ImportantBotTraitWordCount = 3;
std::unordered_map<uint64_t, std::string> g_ImportantBotBackgrounds;
std::mutex g_ImportantBotBackgroundsMutex;

// --------------------------------------------
// Blacklist: Prefixes for Commands (not chat)
// --------------------------------------------
std::vector<std::string> g_BlacklistCommands = {
    ".playerbots",
    "playerbot",
};

// --------------------------------------------
// Environment/Contextual Random Chatter Templates
// --------------------------------------------
std::vector<std::string> g_EnvCommentCreature;
std::vector<std::string> g_EnvCommentGameObject;
std::vector<std::string> g_EnvCommentEquippedItem;
std::vector<std::string> g_EnvCommentBagItem;
std::vector<std::string> g_EnvCommentBagItemSell;
std::vector<std::string> g_EnvCommentSpell;
std::vector<std::string> g_EnvCommentQuestArea;
std::vector<std::string> g_EnvCommentVendor;
std::vector<std::string> g_EnvCommentQuestgiver;
std::vector<std::string> g_EnvCommentBagSlots;
std::vector<std::string> g_EnvCommentDungeon;
std::vector<std::string> g_EnvCommentUnfinishedQuest;

// --------------------------------------------
// Guild-Specific Random Chatter Templates
// --------------------------------------------
std::vector<std::string> g_GuildEnvCommentGuildMember;
std::vector<std::string> g_GuildEnvCommentGuildRank;
std::vector<std::string> g_GuildEnvCommentGuildBank;
std::vector<std::string> g_GuildEnvCommentGuildMOTD;
std::vector<std::string> g_GuildEnvCommentGuildInfo;
std::vector<std::string> g_GuildEnvCommentGuildOnlineMembers;
std::vector<std::string> g_GuildEnvCommentGuildRaid;
std::vector<std::string> g_GuildEnvCommentGuildEndgame;
std::vector<std::string> g_GuildEnvCommentGuildStrategy;
std::vector<std::string> g_GuildEnvCommentGuildGroup;
std::vector<std::string> g_GuildEnvCommentGuildPvP;
std::vector<std::string> g_GuildEnvCommentGuildCommunity;

// --------------------------------------------
// Guild-Specific Random Chatter Configuration
// --------------------------------------------
bool g_EnableGuildEventChatter = true;
bool g_EnableGuildRandomAmbientChatter = true;
uint32_t g_GuildRandomChatterChance = 10;
uint32_t g_GuildChatterBotCommentChance = 25;
uint32_t g_GuildChatterMaxBotsPerEvent = 2;

// --------------------------------------------
// Guild-Specific Event Chatter Templates
// --------------------------------------------
std::string g_GuildEventTypeLevelUp = "";
std::string g_GuildEventTypeDungeonComplete = "";
std::string g_GuildEventTypeEpicGear = "";
std::string g_GuildEventTypeRareGear = "";
std::string g_GuildEventTypeGuildJoin = "";
std::string g_GuildEventTypeGuildLeave = "";
std::string g_GuildEventTypeGuildPromotion = "";
std::string g_GuildEventTypeGuildDemotion = "";
std::string g_GuildEventTypeGuildLogin = "";
std::string g_GuildEventTypeGuildAchievement = "";

// --------------------------------------------
// Event Chatter Templates
// --------------------------------------------
std::string g_EventTypeDefeated;       // "defeated"
std::string g_EventTypeDefeatedPlayer; // "defeated player"
std::string g_EventTypePetDefeated;    // "pet defeated"
std::string g_EventTypeGotItem;        // "got item"
std::string g_EventTypeDied;           // "died"
std::string g_EventTypeCompletedQuest; // "completed quest"
std::string g_EventTypeLearnedSpell;   // "learned spell"
std::string g_EventTypeRequestedDuel;  // "requested to duel"
std::string g_EventTypeStartedDueling; // "started dueling"
std::string g_EventTypeWonDuel;        // "won duel against"
std::string g_EventTypeLeveledUp;      // "leveled up"
std::string g_EventTypeAchievement;    // "earned achievement"
std::string g_EventTypeUsedObject;     // "used object"

// Chance variables for normal events
int g_EventTypeDefeated_Chance = 0;
int g_EventTypeDefeatedPlayer_Chance = 0;
int g_EventTypePetDefeated_Chance = 0;
int g_EventTypeGotItem_Chance = 0;
int g_EventTypeDied_Chance = 0;
int g_EventTypeCompletedQuest_Chance = 0;
int g_EventTypeLearnedSpell_Chance = 0;
int g_EventTypeRequestedDuel_Chance = 0;
int g_EventTypeStartedDueling_Chance = 0;
int g_EventTypeWonDuel_Chance = 0;
int g_EventTypeLeveledUp_Chance = 0;
int g_EventTypeAchievement_Chance = 0;
int g_EventTypeUsedObject_Chance = 0;

// Chance variables for guild events
int g_GuildEventTypeEpicGear_Chance = 0;
int g_GuildEventTypeRareGear_Chance = 0;
int g_GuildEventTypeGuildJoin_Chance = 0;
int g_GuildEventTypeGuildLogin_Chance = 0;
int g_GuildEventTypeGuildLeave_Chance = 0;
int g_GuildEventTypeGuildPromotion_Chance = 0;
int g_GuildEventTypeGuildDemotion_Chance = 0;
int g_GuildEventTypeGuildAchievement_Chance = 0;
int g_GuildEventTypeLevelUp_Chance = 0;
int g_GuildEventTypeDungeonComplete_Chance = 0;

// Event Cooldown
uint32_t g_EventCooldownTime = 10;

// --------------------------------------------
// Channel Disable Settings
// --------------------------------------------
bool g_DisableForCustomChannels = false;
bool g_DisableForSayYell = false;
bool g_DisableForGuild = false;
bool g_DisableForParty = false;

// --------------------------------------------
// Typing Simulation Settings
// --------------------------------------------
bool g_EnableTypingSimulation = false;
uint32_t g_TypingSimulationBaseDelay = 1000; // 1000ms base delay
uint32_t g_TypingSimulationDelayPerChar =
    250; // 250ms per character (4 chars/sec)

static std::vector<std::string> SplitString(const std::string &str,
                                            char delim) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delim)) {
    // Trim whitespace from token
    size_t start = token.find_first_not_of(" \t");
    size_t end = token.find_last_not_of(" \t");
    if (start != std::string::npos && end != std::string::npos)
      tokens.push_back(token.substr(start, end - start + 1));
  }
  return tokens;
}

// Load Bot Personalities from Database
void LoadBotPersonalityList() {
  // Let's make sure our user has sourced the required sql file to add the new
  // table
  QueryResult tableExists = CharacterDatabase.Query(
      "SELECT * FROM information_schema.tables WHERE table_schema = "
      "'acore_characters' AND table_name = 'mod_ollama_chat_personality' LIMIT "
      "1");
  if (!tableExists) {
    LOG_ERROR("server.loading",
              "[Ollama Chat] Please source the required database table first");
    return;
  }

  QueryResult result = CharacterDatabase.Query(
      "SELECT guid,personality FROM mod_ollama_chat_personality");

  if (!result) {
    return;
  }
  if (result->GetRowCount() == 0) {
    return;
  }

  if (g_DebugEnabled) {
    LOG_INFO("server.loading",
             "[Ollama Chat] Fetching Bot Personality List into array");
  }

  do {
    uint64_t personalityBotGUID = result->Fetch()[0].Get<uint64_t>();
    std::string personalityKey = result->Fetch()[1].Get<std::string>();
    g_BotPersonalityList[personalityBotGUID] = personalityKey;
  } while (result->NextRow());
}

std::string GetMultiLineConfigValue(const std::string &configFilePath,
                                    const std::string &key) {
  std::ifstream infile(configFilePath);
  if (!infile)
    return "";

  std::string line;
  std::string value;
  bool foundKey = false;
  while (std::getline(infile, line)) {
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    if (trimmed.empty() || trimmed[0] == '#')
      continue;
    size_t pos = trimmed.find('=');
    if (!foundKey && pos != std::string::npos) {
      std::string possibleKey = trimmed.substr(0, pos);
      possibleKey.erase(possibleKey.find_last_not_of(" \t\r\n") + 1);
      if (possibleKey == key) {
        foundKey = true;
        std::string afterEq = trimmed.substr(pos + 1);
        afterEq.erase(0, afterEq.find_first_not_of(" \t\r\n"));
        value += afterEq;
        continue;
      }
    } else if (foundKey) {
      // New config key or section
      if (trimmed.find('=') != std::string::npos &&
          trimmed.find('[') == std::string::npos)
        break;
      if (!value.empty())
        value += "\n";
      value += trimmed;
    }
  }

  return value;
}

void LoadOllamaChatConfig() {
  g_SayDistance = sConfigMgr->GetOption<float>("OllamaChat.SayDistance", 30.0f);
  g_YellDistance =
      sConfigMgr->GetOption<float>("OllamaChat.YellDistance", 100.0f);

  // Load per-channel-type reply chances
  g_PlayerReplyChance_Say =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Say", 90);
  g_BotReplyChance_Say =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Say", 10);
  g_PlayerReplyChance_Channel = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.PlayerReplyChance.Channel", 50);
  g_BotReplyChance_Channel =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Channel", 5);
  g_PlayerReplyChance_Party =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Party", 90);
  g_BotReplyChance_Party =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Party", 10);
  g_PlayerReplyChance_Guild =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.PlayerReplyChance.Guild", 70);
  g_BotReplyChance_Guild =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.BotReplyChance.Guild", 5);

  g_MaxBotsToPick =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxBotsToPick", 2);
  g_OllamaUrl = sConfigMgr->GetOption<std::string>(
      "OllamaChat.Url", "http://localhost:11434/api/generate");
  g_OllamaModel =
      sConfigMgr->GetOption<std::string>("OllamaChat.Model", "llama3.2:1b");
  g_OllamaNumPredict =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.NumPredict", 40);
  g_OllamaTemperature =
      sConfigMgr->GetOption<float>("OllamaChat.Temperature", 0.8f);
  g_OllamaTopP = sConfigMgr->GetOption<float>("OllamaChat.TopP", 0.95f);
  g_OllamaRepeatPenalty =
      sConfigMgr->GetOption<float>("OllamaChat.RepeatPenalty", 1.1f);
  g_OllamaNumCtx = sConfigMgr->GetOption<uint32_t>("OllamaChat.NumCtx", 0);
  g_OllamaNumThreads =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.NumThreads", 0);
  g_OllamaStop = sConfigMgr->GetOption<std::string>("OllamaChat.Stop", "");
  g_OllamaSystemPrompt =
      sConfigMgr->GetOption<std::string>("OllamaChat.SystemPrompt", "");
  g_OllamaSeed = sConfigMgr->GetOption<std::string>("OllamaChat.Seed", "");

  g_UseOpenRouter =
      sConfigMgr->GetOption<bool>("OllamaChat.UseOpenRouter", false);
  g_OpenRouterApiKey =
      sConfigMgr->GetOption<std::string>("OllamaChat.OpenRouterApiKey", "");
  g_OpenRouterUrl = sConfigMgr->GetOption<std::string>(
      "OllamaChat.OpenRouterUrl",
      "https://openrouter.ai/api/v1/chat/completions");
  g_OpenRouterMaxCallsPerPeriod =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.OpenRouterMaxCallsPerPeriod", 0);
  g_OpenRouterPeriodLengthSeconds =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.OpenRouterPeriodLengthSeconds", 60);

  g_MaxConcurrentQueries =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxConcurrentQueries", 0);

  g_Enable = sConfigMgr->GetOption<bool>("OllamaChat.Enable", true);
  g_DisableRepliesInCombat =
      sConfigMgr->GetOption<bool>("OllamaChat.DisableRepliesInCombat", true);
  g_EnableRandomChatter =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableRandomChatter", true);
  g_EnableEventChatter =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableEventChatter", true);
  g_EnableWhisperReplies =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableWhisperReplies", false);

  g_DebugEnabled =
      sConfigMgr->GetOption<bool>("OllamaChat.DebugEnabled", false);
  g_DebugShowFullPrompt =
      sConfigMgr->GetOption<bool>("OllamaChat.DebugShowFullPrompt", false);

  g_MinRandomInterval =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.MinRandomInterval", 45);
  g_MaxRandomInterval =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxRandomInterval", 180);
  g_RandomChatterRealPlayerDistance = sConfigMgr->GetOption<float>(
      "OllamaChat.RandomChatterRealPlayerDistance", 40.0f);
  g_RandomChatterBotCommentChance = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.RandomChatterBotCommentChance", 25);
  g_RandomChatterMaxBotsPerPlayer = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.RandomChatterMaxBotsPerPlayer", 2);

  g_EnableGuildRandomAmbientChatter = sConfigMgr->GetOption<bool>(
      "OllamaChat.EnableGuildRandomAmbientChatter", true);
  g_GuildRandomChatterChance = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.GuildRandomChatterChance", 10);

  g_EventChatterRealPlayerDistance = sConfigMgr->GetOption<float>(
      "OllamaChat.EventChatterRealPlayerDistance", 40.0f);
  g_EventChatterBotCommentChance = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.EventChatterBotCommentChance", 15);
  g_EventChatterBotSelfCommentChance = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.EventChatterBotSelfCommentChance", 5);
  g_EventChatterMaxBotsPerPlayer = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.EventChatterMaxBotsPerPlayer", 2);

  g_EnableRPPersonalities =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableRPPersonalities", false);

  g_RandomChatterPromptTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.RandomChatterPromptTemplate", "");

  // Load random chatter prompt variations
  std::string variationsStr = sConfigMgr->GetOption<std::string>(
      "OllamaChat.RandomChatterPromptVariations", "");
  g_RandomChatterPromptVariations.clear();
  if (!variationsStr.empty()) {
    std::stringstream ss(variationsStr);
    std::string variation;
    while (std::getline(ss, variation, '|')) {
      if (!variation.empty()) {
        g_RandomChatterPromptVariations.push_back(variation);
      }
    }
  }

  // Load random chatter question variations
  std::string questionsStr = sConfigMgr->GetOption<std::string>(
      "OllamaChat.RandomChatterQuestionVariations", "");
  g_RandomChatterQuestionVariations.clear();
  if (!questionsStr.empty()) {
    std::stringstream ss(questionsStr);
    std::string question;
    while (std::getline(ss, question, '|')) {
      if (!question.empty()) {
        g_RandomChatterQuestionVariations.push_back(question);
      }
    }
  }

  g_EventChatterPromptTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.EventChatterPromptTemplate", "");

  g_ChatPromptTemplate =
      sConfigMgr->GetOption<std::string>("OllamaChat.ChatPromptTemplate", "");

  g_ChatExtraInfoTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.ChatExtraInfoTemplate", "");

  g_DefaultPersonalityPrompt = sConfigMgr->GetOption<std::string>(
      "OllamaChat.DefaultPersonalityPrompt", "");

  g_MaxConversationHistory =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.MaxConversationHistory", 5);
  g_ConversationHistorySaveInterval = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.ConversationHistorySaveInterval", 10);

  g_ChatHistoryHeaderTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.ChatHistoryHeaderTemplate", "");
  g_ChatHistoryLineTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.ChatHistoryLineTemplate", "");
  g_ChatHistoryFooterTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.ChatHistoryFooterTemplate", "");

  g_EnableChatBotSnapshotTemplate = sConfigMgr->GetOption<bool>(
      "OllamaChat.EnableChatBotSnapshotTemplate", false);
  g_ChatBotSnapshotTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.ChatBotSnapshotTemplate", "");

  g_EnableChatHistory =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableChatHistory", true);

  // Bot-Player Sentiment Tracking
  g_EnableSentimentTracking =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableSentimentTracking", true);
  g_SentimentDefaultValue =
      sConfigMgr->GetOption<float>("OllamaChat.SentimentDefaultValue", 0.5f);
  g_SentimentAdjustmentStrength = sConfigMgr->GetOption<float>(
      "OllamaChat.SentimentAdjustmentStrength", 0.1f);
  g_SentimentSaveInterval =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.SentimentSaveInterval", 10);
  g_SentimentAnalysisPrompt = sConfigMgr->GetOption<std::string>(
      "OllamaChat.SentimentAnalysisPrompt",
      "Analyze the sentiment of this message: \"{message}\". Respond only "
      "with: POSITIVE, NEGATIVE, or NEUTRAL.");
  g_SentimentPromptTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.SentimentPromptTemplate",
      "Your relationship sentiment with {player_name} is {sentiment_value} "
      "(0.0=hostile, 0.5=neutral, 1.0=friendly). Use this to guide your tone "
      "and response.");

  // RAG (Retrieval-Augmented Generation) System
  g_EnableRAG = sConfigMgr->GetOption<bool>("OllamaChat.EnableRAG", false);
  g_RAGDataPath =
      sConfigMgr->GetOption<std::string>("OllamaChat.RAGDataPath", "rag/");
  g_RAGMaxRetrievedItems =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGMaxRetrievedItems", 3);
  g_RAGSimilarityThreshold =
      sConfigMgr->GetOption<float>("OllamaChat.RAGSimilarityThreshold", 0.3f);
  g_RAGPromptTemplate = sConfigMgr->GetOption<std::string>(
      "OllamaChat.RAGPromptTemplate",
      "RELEVANT INFORMATION:\n{rag_info}\nUse this information to provide "
      "accurate and detailed responses when applicable.");
  g_RAGMaxVocabularySize =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.RAGMaxVocabularySize", 10000);

  g_ThinkModeEnableForModule =
      sConfigMgr->GetOption<bool>("OllamaChat.ThinkModeEnableForModule", false);

  // Typing Simulation
  g_EnableTypingSimulation =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableTypingSimulation", false);
  g_TypingSimulationBaseDelay = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.TypingSimulationBaseDelay", 1000);
  g_TypingSimulationDelayPerChar = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.TypingSimulationDelayPerChar", 250);

  g_EventTypeDefeated =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeDefeated", "");
  g_EventTypeDefeatedPlayer = sConfigMgr->GetOption<std::string>(
      "OllamaChat.EventTypeDefeatedPlayer", "");
  g_EventTypePetDefeated =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypePetDefeated", "");
  g_EventTypeGotItem =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeGotItem", "");
  g_EventTypeDied =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeDied", "");
  g_EventTypeCompletedQuest = sConfigMgr->GetOption<std::string>(
      "OllamaChat.EventTypeCompletedQuest", "");
  g_EventTypeLearnedSpell = sConfigMgr->GetOption<std::string>(
      "OllamaChat.EventTypeLearnedSpell", "");
  g_EventTypeRequestedDuel = sConfigMgr->GetOption<std::string>(
      "OllamaChat.EventTypeRequestedDuel", "");
  g_EventTypeStartedDueling = sConfigMgr->GetOption<std::string>(
      "OllamaChat.EventTypeStartedDueling", "");
  g_EventTypeWonDuel =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeWonDuel", "");
  g_EventTypeLeveledUp =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeLeveledUp", "");
  g_EventTypeAchievement =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeAchievement", "");
  g_EventTypeUsedObject =
      sConfigMgr->GetOption<std::string>("OllamaChat.EventTypeUsedObject", "");

  // Load extra blacklist commands from config (comma-separated list)
  std::string extraBlacklist =
      sConfigMgr->GetOption<std::string>("OllamaChat.BlacklistCommands", "");
  if (!extraBlacklist.empty()) {
    std::vector<std::string> extraList = SplitString(extraBlacklist, ',');
    for (const auto &cmd : extraList) {
      g_BlacklistCommands.push_back(cmd);
    }
  }

  LoadPersonalityTemplatesFromDB();

  g_queryManager.setMaxConcurrentQueries(g_MaxConcurrentQueries);

  // Loads the environment random chatter message templates for each type.
  // Each config option is a pipe-separated list of string templates,
  // using {} as a placeholder for named substitutions.
  // Helper to load a multi-line config option into a std::vector<std::string>
  auto LoadEnvCommentVector = [](const char *key,
                                 const std::vector<std::string> &defaults = {})
      -> std::vector<std::string> {
    std::string val = sConfigMgr->GetOption<std::string>(key, "");
    std::vector<std::string> result;
    std::istringstream iss(val);
    std::string token;
    while (std::getline(iss, token, '|')) { // Split by '|'
      // Trim whitespace from token
      size_t start = token.find_first_not_of(" \t\r\n");
      size_t end = token.find_last_not_of(" \t\r\n");
      if (start != std::string::npos && end != std::string::npos)
        result.push_back(token.substr(start, end - start + 1));
    }
    if (result.empty() && !defaults.empty())
      return defaults;
    return result;
  };

  g_EnvCommentCreature =
      LoadEnvCommentVector("OllamaChat.EnvCommentCreature", {""});
  g_EnvCommentGameObject =
      LoadEnvCommentVector("OllamaChat.EnvCommentGameObject", {""});
  g_EnvCommentEquippedItem =
      LoadEnvCommentVector("OllamaChat.EnvCommentEquippedItem", {""});
  g_EnvCommentBagItem =
      LoadEnvCommentVector("OllamaChat.EnvCommentBagItem", {""});
  g_EnvCommentBagItemSell =
      LoadEnvCommentVector("OllamaChat.EnvCommentBagItemSell", {""});
  g_EnvCommentSpell = LoadEnvCommentVector("OllamaChat.EnvCommentSpell", {""});
  g_EnvCommentQuestArea =
      LoadEnvCommentVector("OllamaChat.EnvCommentQuestArea", {""});
  g_EnvCommentVendor =
      LoadEnvCommentVector("OllamaChat.EnvCommentVendor", {""});
  g_EnvCommentQuestgiver =
      LoadEnvCommentVector("OllamaChat.EnvCommentQuestgiver", {""});
  g_EnvCommentBagSlots =
      LoadEnvCommentVector("OllamaChat.EnvCommentBagSlots", {""});
  g_EnvCommentDungeon =
      LoadEnvCommentVector("OllamaChat.EnvCommentDungeon", {""});
  g_EnvCommentUnfinishedQuest =
      LoadEnvCommentVector("OllamaChat.EnvCommentUnfinishedQuest", {""});

  // Guild-specific random chatter templates
  g_GuildEnvCommentGuildMember =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildMember", {""});
  g_GuildEnvCommentGuildRank =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildRank", {""});
  g_GuildEnvCommentGuildBank =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildBank", {""});
  g_GuildEnvCommentGuildMOTD =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildMOTD", {""});
  g_GuildEnvCommentGuildInfo =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildInfo", {""});
  g_GuildEnvCommentGuildOnlineMembers = LoadEnvCommentVector(
      "OllamaChat.GuildEnvCommentGuildOnlineMembers", {""});
  g_GuildEnvCommentGuildRaid =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildRaid", {""});
  g_GuildEnvCommentGuildEndgame =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildEndgame", {""});
  g_GuildEnvCommentGuildStrategy =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildStrategy", {""});
  g_GuildEnvCommentGuildGroup =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildGroup", {""});
  g_GuildEnvCommentGuildPvP =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildPvP", {""});
  g_GuildEnvCommentGuildCommunity =
      LoadEnvCommentVector("OllamaChat.GuildEnvCommentGuildCommunity", {""});

  // Guild-specific configuration
  g_EnableGuildEventChatter =
      sConfigMgr->GetOption<bool>("OllamaChat.EnableGuildEventChatter", true);
  g_GuildChatterBotCommentChance = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.GuildChatterBotCommentChance", 25);
  g_GuildChatterMaxBotsPerEvent = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.GuildChatterMaxBotsPerEvent", 2);

  // Guild-specific event templates
  g_GuildEventTypeLevelUp = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeLevelUp", "");
  g_GuildEventTypeDungeonComplete = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeDungeonComplete", "");
  g_GuildEventTypeEpicGear = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeEpicGear", "");
  g_GuildEventTypeRareGear = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeRareGear", "");
  g_GuildEventTypeGuildJoin = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeGuildJoin", "");
  g_GuildEventTypeGuildLogin = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeGuildLogin", "");
  g_GuildEventTypeGuildLeave = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeGuildLeave", "");
  g_GuildEventTypeGuildPromotion = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeGuildPromotion", "");
  g_GuildEventTypeGuildDemotion = sConfigMgr->GetOption<std::string>(
      "OllamaChat.GuildEventTypeGuildDemotion", "");

  // Load chance variables for normal events
  g_EventTypeDefeated_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeDefeated_Chance", 0);
  g_EventTypeDefeatedPlayer_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.EventTypeDefeatedPlayer_Chance", 0);
  g_EventTypePetDefeated_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypePetDefeated_Chance", 0);
  g_EventTypeGotItem_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeGotItem_Chance", 0);
  g_EventTypeDied_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeDied_Chance", 0);
  g_EventTypeCompletedQuest_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.EventTypeCompletedQuest_Chance", 0);
  g_EventTypeLearnedSpell_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeLearnedSpell_Chance", 0);
  g_EventTypeRequestedDuel_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeRequestedDuel_Chance", 0);
  g_EventTypeStartedDueling_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.EventTypeStartedDueling_Chance", 0);
  g_EventTypeWonDuel_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeWonDuel_Chance", 0);
  g_EventTypeLeveledUp_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeLeveledUp_Chance", 0);
  g_EventTypeAchievement_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeAchievement_Chance", 0);
  g_EventTypeUsedObject_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.EventTypeUsedObject_Chance", 0);

  // Load chance variables for guild events
  g_GuildEventTypeEpicGear_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeEpicGear_Chance", 0);
  g_GuildEventTypeRareGear_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeRareGear_Chance", 0);
  g_GuildEventTypeGuildJoin_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeGuildJoin_Chance", 0);
  g_GuildEventTypeGuildLogin_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeGuildLogin_Chance", 0);
  g_GuildEventTypeGuildLeave_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeGuildLeave_Chance", 0);
  g_GuildEventTypeGuildPromotion_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeGuildPromotion_Chance", 0);
  g_GuildEventTypeGuildDemotion_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeGuildDemotion_Chance", 0);
  g_GuildEventTypeGuildAchievement_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeGuildAchievement_Chance", 0);
  g_GuildEventTypeLevelUp_Chance =
      sConfigMgr->GetOption<int>("OllamaChat.GuildEventTypeLevelUp_Chance", 0);
  g_GuildEventTypeDungeonComplete_Chance = sConfigMgr->GetOption<int>(
      "OllamaChat.GuildEventTypeDungeonComplete_Chance", 0);

  // Cooldown time for events
  g_EventCooldownTime =
      sConfigMgr->GetOption<uint32_t>("OllamaChat.EventCooldownTime", 10);

  // Channel disable settings
  g_DisableForCustomChannels =
      sConfigMgr->GetOption<bool>("OllamaChat.DisableForCustomChannels", false);
  g_DisableForSayYell =
      sConfigMgr->GetOption<bool>("OllamaChat.DisableForSayYell", false);
  g_DisableForGuild =
      sConfigMgr->GetOption<bool>("OllamaChat.DisableForGuild", false);
  g_DisableForParty =
      sConfigMgr->GetOption<bool>("OllamaChat.DisableForParty", false);

  LOG_INFO("server.loading",
           "[Ollama Chat] Config loaded: Enabled = {}, SayDistance = {}, "
           "YellDistance = {}, "
           "Reply Chances - Say: P{}%/B{}%, Channel: P{}%/B{}%, Party: "
           "P{}%/B{}%, Guild: P{}%/B{}%, MaxBotsToPick = {}, "
           "Url = {}, Model = {}, MaxConcurrentQueries = {}, "
           "EnableRandomChatter = {}, MinRandInt = {}, MaxRandInt = {}, "
           "RandomChatterRealPlayerDistance = {}, "
           "RandomChatterBotCommentChance = {}. MaxConcurrentQueries = {}. "
           "Extra blacklist commands: {}",
           g_Enable, g_SayDistance, g_YellDistance, g_PlayerReplyChance_Say,
           g_BotReplyChance_Say, g_PlayerReplyChance_Channel,
           g_BotReplyChance_Channel, g_PlayerReplyChance_Party,
           g_BotReplyChance_Party, g_PlayerReplyChance_Guild,
           g_BotReplyChance_Guild, g_MaxBotsToPick, g_OllamaUrl, g_OllamaModel,
           g_MaxConcurrentQueries, g_EnableRandomChatter, g_MinRandomInterval,
           g_MaxRandomInterval, g_RandomChatterRealPlayerDistance,
           g_RandomChatterBotCommentChance, g_MaxConcurrentQueries,
           extraBlacklist);

  // Important Bot Backgrounds
  g_EnableImportantBotBackgrounds = sConfigMgr->GetOption<bool>(
      "OllamaChat.EnableImportantBotBackgrounds", true);
  g_ImportantBotTraitWordCount = sConfigMgr->GetOption<uint32_t>(
      "OllamaChat.ImportantBotTraitWordCount", 3);
  g_ImportantBotBackgroundGenerationPrompt =
      sConfigMgr->GetOption<std::string>(
          "OllamaChat.ImportantBotBackgroundGenerationPrompt",
          "Generate a 1-2 paragraph personality profile for a World of Warcraft character. "
          "This is a Wrath of the Lich King era player character, not an NPC. "
          "Write it as a description of how this person behaves, talks, and thinks - their quirks, habits, and social tendencies. "
          "Do NOT use the character's name in the description. Do NOT write in first person. "
          "Write in third person as a character study.\n\n"
          "Character details:\n"
          "- Name: {bot_name}\n- Class: {bot_class}\n- Race: {bot_race}\n- Gender: {bot_gender}\n"
          "- Base personality archetype: {personality_name} ({personality_prompt})\n"
          "- MBTI type: {mbti}\n- Key traits: {trait_words}\n\n"
          "Write a vivid, specific personality description that makes this character feel like a real person playing WoW. "
          "Include subtle contradictions and human touches. Keep it to 1-2 short paragraphs, no bullet points.");

  {
    std::string traitWordsStr = sConfigMgr->GetOption<std::string>(
        "OllamaChat.ImportantBotTraitWords",
        "stubborn|curious|impulsive|loyal|sarcastic|cautious|reckless|witty|blunt|nostalgic|"
        "cheerful|brooding|hot-headed|meticulous|laid-back|competitive|superstitious|forgetful|"
        "dramatic|stoic|perfectionist|mischievous|absent-minded|generous|stingy|paranoid|"
        "optimistic|pessimistic|philosophical|restless|patient|territorial|humble|boastful|"
        "resourceful|clumsy|perceptive|oblivious|eccentric|deadpan");
    g_ImportantBotTraitWords.clear();
    std::istringstream iss(traitWordsStr);
    std::string token;
    while (std::getline(iss, token, '|')) {
      size_t start = token.find_first_not_of(" \t\r\n\"");
      size_t end = token.find_last_not_of(" \t\r\n\"");
      if (start != std::string::npos && end != std::string::npos)
        g_ImportantBotTraitWords.push_back(token.substr(start, end - start + 1));
    }
  }

  LOG_INFO("server.loading", "[Ollama Chat] Backend: {}",
           g_UseOpenRouter ? "OpenRouter" : "Ollama (local)");

  if (g_UseOpenRouter && g_OpenRouterApiKey.empty()) {
    LOG_ERROR("server.loading",
              "[Ollama Chat] UseOpenRouter is enabled but OpenRouterApiKey is "
              "empty! API calls will fail.");
  }
}

void LoadPersonalityTemplatesFromDB() {
  g_PersonalityPrompts.clear();
  g_PersonalityKeys.clear();
  g_PersonalityKeysRandomOnly.clear();

  QueryResult result =
      CharacterDatabase.Query("SELECT `key`, `prompt`, `manual_only` FROM "
                              "`mod_ollama_chat_personality_templates`");
  if (!result) {
    LOG_ERROR("server.loading",
              "[Ollama Chat] No personality templates found in the database!");
    return;
  }

  do {
    std::string key = (*result)[0].Get<std::string>();
    std::string prompt = (*result)[1].Get<std::string>();
    bool manualOnly = (*result)[2].Get<bool>();

    g_PersonalityPrompts[key] = prompt;
    g_PersonalityKeys.push_back(key);

    // Only add to random pool if not manual_only
    if (!manualOnly) {
      g_PersonalityKeysRandomOnly.push_back(key);
    }
  } while (result->NextRow());

  LOG_INFO("server.loading",
           "[Ollama Chat] Cached {} personalities ({} available for random "
           "assignment).",
           g_PersonalityKeys.size(), g_PersonalityKeysRandomOnly.size());
}

void LoadBotConversationHistoryFromDB() {
  QueryResult result = CharacterDatabase.Query(
      "SELECT bot_guid, player_guid, player_message, bot_reply FROM "
      "mod_ollama_chat_history ORDER BY timestamp ASC");
  if (!result)
    return;

  std::lock_guard<std::mutex> lock(g_ConversationHistoryMutex);
  g_BotConversationHistory.clear();

  do {
    uint64_t botGuid = (*result)[0].Get<uint64_t>();
    uint64_t playerGuid = (*result)[1].Get<uint64_t>();
    std::string playerMsg = (*result)[2].Get<std::string>();
    std::string botReply = (*result)[3].Get<std::string>();

    auto &playerHistory = g_BotConversationHistory[botGuid][playerGuid];
    playerHistory.push_back({playerMsg, botReply});
    while (playerHistory.size() > g_MaxConversationHistory) {
      playerHistory.pop_front();
    }

  } while (result->NextRow());
}

void LoadImportantBotBackgroundsFromDB() {
  std::lock_guard<std::mutex> lock(g_ImportantBotBackgroundsMutex);
  g_ImportantBotBackgrounds.clear();

  QueryResult result = CharacterDatabase.Query(
      "SELECT bot_guid, background FROM mod_ollama_chat_important_bot_backgrounds");
  if (!result)
    return;

  do {
    uint64_t botGuid = (*result)[0].Get<uint64_t>();
    std::string background = (*result)[1].Get<std::string>();
    g_ImportantBotBackgrounds[botGuid] = background;
  } while (result->NextRow());

  LOG_INFO("server.loading",
           "[Ollama Chat] Loaded {} important bot backgrounds from database",
           g_ImportantBotBackgrounds.size());
}

void IdentifyAndGenerateImportantBotBackgrounds() {
  if (!g_EnableImportantBotBackgrounds) {
    LOG_INFO("server.loading",
             "[Ollama Chat] Important bot backgrounds disabled, skipping");
    return;
  }

  if (g_BotPersonalityList.empty()) {
    LOG_INFO("server.loading",
             "[Ollama Chat] No bot personalities loaded, skipping important bot background generation");
    return;
  }

  std::set<uint64_t> importantBotGuids;

  // Friends: find bots (GUIDs in personality table) that appear on any friend list
  QueryResult friendResult = CharacterDatabase.Query(
      "SELECT DISTINCT cs.friend FROM character_social cs "
      "INNER JOIN mod_ollama_chat_personality p ON cs.friend = p.guid "
      "WHERE cs.flags & 1");
  if (friendResult) {
    do {
      importantBotGuids.insert((*friendResult)[0].Get<uint64_t>());
    } while (friendResult->NextRow());
  }

  // Guild: find bots that share a guild with non-bot players
  // A "non-bot player" is a guild member whose GUID is NOT in the personality table
  QueryResult guildResult = CharacterDatabase.Query(
      "SELECT DISTINCT gm_bot.guid FROM guild_member gm_bot "
      "INNER JOIN mod_ollama_chat_personality p ON gm_bot.guid = p.guid "
      "WHERE gm_bot.guildid IN ("
      "  SELECT gm_real.guildid FROM guild_member gm_real "
      "  LEFT JOIN mod_ollama_chat_personality p2 ON gm_real.guid = p2.guid "
      "  WHERE p2.guid IS NULL"
      ")");
  if (guildResult) {
    do {
      importantBotGuids.insert((*guildResult)[0].Get<uint64_t>());
    } while (guildResult->NextRow());
  }

  // Filter out bots that already have backgrounds
  std::vector<uint64_t> botsNeedingBackgrounds;
  {
    std::lock_guard<std::mutex> lock(g_ImportantBotBackgroundsMutex);
    for (uint64_t guid : importantBotGuids) {
      if (g_ImportantBotBackgrounds.find(guid) == g_ImportantBotBackgrounds.end()) {
        botsNeedingBackgrounds.push_back(guid);
      }
    }
  }

  if (botsNeedingBackgrounds.empty()) {
    LOG_INFO("server.loading",
             "[Ollama Chat] All {} important bots already have backgrounds",
             importantBotGuids.size());
    return;
  }

  LOG_INFO("server.loading",
           "[Ollama Chat] Found {} important bots, {} need background generation",
           importantBotGuids.size(), botsNeedingBackgrounds.size());

  // Generate backgrounds in a detached background thread to avoid blocking startup
  std::thread([botsNeedingBackgrounds]() {
    // Small delay to let the server finish starting up and the LLM API become available
    std::this_thread::sleep_for(std::chrono::seconds(10));

    for (size_t i = 0; i < botsNeedingBackgrounds.size(); ++i) {
      if (g_isShuttingDown) return;

      uint64_t botGuid = botsNeedingBackgrounds[i];
      LOG_INFO("server.loading",
               "[Ollama Chat] Generating background for important bot {} ({}/{})",
               botGuid, i + 1, botsNeedingBackgrounds.size());

      GenerateImportantBotBackground(botGuid);

      // Small delay between generations to avoid flooding the LLM
      if (i + 1 < botsNeedingBackgrounds.size()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }

    LOG_INFO("server.loading",
             "[Ollama Chat] Finished generating backgrounds for {} important bots",
             botsNeedingBackgrounds.size());
  }).detach();
}

// Definition of the configuration WorldScript.
OllamaChatConfigWorldScript::OllamaChatConfigWorldScript()
    : WorldScript("OllamaChatConfigWorldScript") {}

void OllamaChatConfigWorldScript::OnStartup() {
  LoadOllamaChatConfig();
  LoadBotPersonalityList();
  LoadBotConversationHistoryFromDB();
  InitializeSentimentTracking();

  // Initialize RAG system if enabled
  if (g_EnableRAG) {
    if (g_RAGSystem) {
      delete g_RAGSystem;
    }
    g_RAGSystem = new OllamaRAGSystem();
    if (!g_RAGSystem->Initialize()) {
      LOG_ERROR("server.loading",
                "[Ollama Chat] Failed to initialize RAG system");
      delete g_RAGSystem;
      g_RAGSystem = nullptr;
    } else {
      LOG_INFO("server.loading",
               "[Ollama Chat] RAG system initialized successfully");
    }
  }

  // Load existing important bot backgrounds and generate missing ones
  LoadImportantBotBackgroundsFromDB();
  IdentifyAndGenerateImportantBotBackgrounds();
}

void OllamaChatConfigWorldScript::OnShutdown() {
  // Clean up RAG system
  if (g_RAGSystem) {
    delete g_RAGSystem;
    g_RAGSystem = nullptr;
    LOG_INFO("server.loading", "[Ollama Chat] RAG system cleaned up");
  }
}
