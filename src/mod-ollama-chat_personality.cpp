#include "mod-ollama-chat_personality.h"
#include "Player.h"
#include "PlayerbotMgr.h"
#include "Log.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat-utilities.h"
#include "DatabaseEnv.h"
#include <random>
#include <vector>
#include <future>

static const char* MBTI_TYPES[] = {
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
};
static const size_t MBTI_TYPE_COUNT = 16;

// Internal personality map
std::string GetBotPersonality(Player* bot)
{
    uint64_t botGuid = bot->GetGUID().GetRawValue();

    // If personality already assigned, return it (but only if RP personalities are enabled)
    auto it = g_BotPersonalityList.find(botGuid);
    if (it != g_BotPersonalityList.end())
    {
        // If RP personalities are disabled, reset to default
        if (!g_EnableRPPersonalities)
        {
            g_BotPersonalityList[botGuid] = "default";
            return "default";
        }
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Using existing personality '{}' for bot {}", it->second, bot->GetName());
        }
        return it->second;
    }

    // RP personalities disabled or config not loaded
    if (!g_EnableRPPersonalities || g_PersonalityKeysRandomOnly.empty())
    {
        g_BotPersonalityList[botGuid] = "default";
        return "default";
    }

    // Try to load from database if you have persistence
    if (g_BotPersonalityList.find(botGuid) != g_BotPersonalityList.end())
    {
        // DB stores string keys now
        std::string dbPersonality = g_BotPersonalityList[botGuid];

        if (dbPersonality.empty())
        {
            dbPersonality = "default";
        }

        g_BotPersonalityList[botGuid] = dbPersonality;

        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[Ollama Chat] Using database personality '{}' for bot {}", dbPersonality, bot->GetName());
        }
        return dbPersonality;
    }

    // Otherwise, assign randomly from config (only from non-manual personalities)
    uint32 newIdx = urand(0, g_PersonalityKeysRandomOnly.size() - 1);
    std::string chosenPersonality = g_PersonalityKeysRandomOnly[newIdx];
    g_BotPersonalityList[botGuid] = chosenPersonality;

    // Save to database if schema supports string (recommend TEXT or VARCHAR column for personality)
    QueryResult tableExists = CharacterDatabase.Query(
        "SELECT * FROM information_schema.tables WHERE table_schema = 'acore_characters' AND table_name = 'mod_ollama_chat_personality' LIMIT 1;");
    if (!tableExists)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Please source the required database table first");
    }
    else
    {
        CharacterDatabase.Execute("INSERT INTO mod_ollama_chat_personality (guid, personality) VALUES ({}, '{}')", botGuid, chosenPersonality);
    }

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Assigned new personality '{}' to bot {}", chosenPersonality, bot->GetName());
    }
    return chosenPersonality;
}


std::string GetPersonalityPromptAddition(const std::string& personality)
{
    auto it = g_PersonalityPrompts.find(personality);
    if (it != g_PersonalityPrompts.end())
        return it->second;
    return g_DefaultPersonalityPrompt;
}

bool SetBotPersonality(Player* bot, const std::string& personality)
{
    if (!bot)
        return false;
    
    uint64_t botGuid = bot->GetGUID().GetRawValue();
    
    // Check if personality exists
    if (g_PersonalityPrompts.find(personality) == g_PersonalityPrompts.end() && personality != "default")
    {
        return false;
    }
    
    // Update in memory
    g_BotPersonalityList[botGuid] = personality;
    
    // Update in database
    CharacterDatabase.Execute("REPLACE INTO mod_ollama_chat_personality (guid, personality) VALUES ({}, '{}')", 
                             botGuid, personality);
    
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Set personality '{}' for bot {}", personality, bot->GetName());
    }
    
    return true;
}

std::vector<std::string> GetAllPersonalityKeys()
{
    return g_PersonalityKeys;
}

bool PersonalityExists(const std::string& personality)
{
    if (personality == "default")
        return true;
    return g_PersonalityPrompts.find(personality) != g_PersonalityPrompts.end();
}

void ClearAllBotPersonalities()
{
    g_BotPersonalityList.clear();
    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[Ollama Chat] Cleared all bot personality assignments due to RP personalities being disabled");
    }
}

static std::string GetClassNameFromId(uint8_t classId)
{
    switch (classId) {
        case 1:  return "Warrior";
        case 2:  return "Paladin";
        case 3:  return "Hunter";
        case 4:  return "Rogue";
        case 5:  return "Priest";
        case 6:  return "Death Knight";
        case 7:  return "Shaman";
        case 8:  return "Mage";
        case 9:  return "Warlock";
        case 11: return "Druid";
        default: return "Adventurer";
    }
}

static std::string GetRaceNameFromId(uint8_t raceId)
{
    switch (raceId) {
        case 1:  return "Human";
        case 2:  return "Orc";
        case 3:  return "Dwarf";
        case 4:  return "Night Elf";
        case 5:  return "Undead";
        case 6:  return "Tauren";
        case 7:  return "Gnome";
        case 8:  return "Troll";
        case 10: return "Blood Elf";
        case 11: return "Draenei";
        default: return "Unknown";
    }
}

void GenerateImportantBotBackground(uint64_t botGuid)
{
    // Check if already generated (thread-safe)
    {
        std::lock_guard<std::mutex> lock(g_ImportantBotBackgroundsMutex);
        if (g_ImportantBotBackgrounds.find(botGuid) != g_ImportantBotBackgrounds.end())
            return;
    }

    // Look up personality key for this bot
    std::string personalityKey = "default";
    auto it = g_BotPersonalityList.find(botGuid);
    if (it != g_BotPersonalityList.end())
        personalityKey = it->second;

    std::string personalityPrompt = g_DefaultPersonalityPrompt;
    auto pit = g_PersonalityPrompts.find(personalityKey);
    if (pit != g_PersonalityPrompts.end())
        personalityPrompt = pit->second;

    // Query character details from DB (name, race, class, gender)
    QueryResult charResult = CharacterDatabase.Query(
        "SELECT name, race, class, gender FROM characters WHERE guid = {}",
        botGuid);
    if (!charResult) {
        LOG_ERROR("server.loading",
                  "[Ollama Chat] Could not find character data for bot GUID {} during background generation",
                  botGuid);
        return;
    }

    std::string botName = (*charResult)[0].Get<std::string>();
    uint8_t raceId = (*charResult)[1].Get<uint8_t>();
    uint8_t classId = (*charResult)[2].Get<uint8_t>();
    uint8_t genderByte = (*charResult)[3].Get<uint8_t>();

    std::string botRace = GetRaceNameFromId(raceId);
    std::string botClass = GetClassNameFromId(classId);
    std::string botGender = (genderByte == 0 ? "Male" : "Female");

    // Pick random MBTI
    std::string mbti = MBTI_TYPES[urand(0, MBTI_TYPE_COUNT - 1)];

    // Pick random trait words
    std::string traitWords;
    if (!g_ImportantBotTraitWords.empty()) {
        uint32_t count = std::min(g_ImportantBotTraitWordCount, (uint32_t)g_ImportantBotTraitWords.size());
        std::vector<size_t> indices(g_ImportantBotTraitWords.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

        // Fisher-Yates partial shuffle
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t j = urand(i, indices.size() - 1);
            std::swap(indices[i], indices[j]);
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (i > 0) traitWords += ", ";
            traitWords += g_ImportantBotTraitWords[indices[i]];
        }
    }

    // Build the generation prompt
    std::string prompt = SafeFormat(
        g_ImportantBotBackgroundGenerationPrompt,
        fmt::arg("bot_name", botName),
        fmt::arg("bot_class", botClass),
        fmt::arg("bot_race", botRace),
        fmt::arg("bot_gender", botGender),
        fmt::arg("personality_name", personalityKey),
        fmt::arg("personality_prompt", personalityPrompt),
        fmt::arg("mbti", mbti),
        fmt::arg("trait_words", traitWords)
    );

    if (prompt.empty() || prompt == "[Format Error]") {
        LOG_ERROR("server.loading",
                  "[Ollama Chat] Failed to build background generation prompt for bot {} (GUID {})",
                  botName, botGuid);
        return;
    }

    try {
        auto future = SubmitQuery(prompt, true);
        if (!future.valid()) {
            LOG_ERROR("server.loading",
                      "[Ollama Chat] Failed to submit background generation query for bot {} (GUID {})",
                      botName, botGuid);
            return;
        }

        std::string background = future.get();
        if (background.empty()) {
            LOG_ERROR("server.loading",
                      "[Ollama Chat] Empty background generated for bot {} (GUID {})",
                      botName, botGuid);
            return;
        }

        // Store in memory
        {
            std::lock_guard<std::mutex> lock(g_ImportantBotBackgroundsMutex);
            g_ImportantBotBackgrounds[botGuid] = background;
        }

        // Store in database
        std::string escBackground = background;
        CharacterDatabase.EscapeString(escBackground);
        std::string escPersonalityKey = personalityKey;
        CharacterDatabase.EscapeString(escPersonalityKey);

        CharacterDatabase.Execute(
            "REPLACE INTO mod_ollama_chat_important_bot_backgrounds "
            "(bot_guid, personality_key, mbti, background, generated_at) "
            "VALUES ({}, '{}', '{}', '{}', NOW())",
            botGuid, escPersonalityKey, mbti, escBackground);

        LOG_INFO("server.loading",
                 "[Ollama Chat] Generated background for important bot {} (GUID {}, {}, {})",
                 botName, botGuid, personalityKey, mbti);
    }
    catch (const std::exception& ex) {
        LOG_ERROR("server.loading",
                  "[Ollama Chat] Exception during background generation for bot {} (GUID {}): {}",
                  botName, botGuid, ex.what());
    }
}

std::string GetImportantBotBackground(uint64_t botGuid)
{
    std::lock_guard<std::mutex> lock(g_ImportantBotBackgroundsMutex);
    auto it = g_ImportantBotBackgrounds.find(botGuid);
    if (it != g_ImportantBotBackgrounds.end())
        return it->second;
    return "";
}
