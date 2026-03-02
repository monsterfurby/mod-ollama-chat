#include "mod-ollama-chat_rag.h"
#include "mod-ollama-chat_config.h"
#include "Log.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;

OllamaRAGSystem::OllamaRAGSystem() : m_initialized(false) {}

OllamaRAGSystem::~OllamaRAGSystem() {}

bool OllamaRAGSystem::Initialize()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized) {
        return true;
    }

    m_ragEntries.clear();
    m_vocabulary.clear();
    m_vocabSet.clear();

    // Use the configured RAG data path directly
    std::string fullPath = g_RAGDataPath;

    if (!LoadRAGDataFromDirectory(fullPath)) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Failed to load RAG data from directory: {}", fullPath);
        return false;
    }

    // Build vocabulary from all entries
    for (const auto& entry : m_ragEntries) {
        auto tokens = TokenizeText(PreprocessText(entry.title + " " + entry.content));
        for (const auto& token : tokens) {
            m_vocabSet.insert(token);
        }
        for (const auto& keyword : entry.keywords) {
            auto keywordTokens = TokenizeText(PreprocessText(keyword));
            for (const auto& token : keywordTokens) {
                m_vocabSet.insert(token);
            }
        }
    }
    m_vocabulary.assign(m_vocabSet.begin(), m_vocabSet.end());

    m_initialized = true;
    LOG_INFO("server.loading", "[Ollama Chat RAG] Initialized with {} entries and {} vocabulary terms",
             m_ragEntries.size(), m_vocabulary.size());

    return true;
}

bool OllamaRAGSystem::LoadRAGDataFromDirectory(const std::string& directoryPath)
{
    try {
        if (!fs::exists(directoryPath)) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Directory does not exist: {}", directoryPath);
            return false;
        }

        if (!fs::is_directory(directoryPath)) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Path is not a directory: {}", directoryPath);
            return false;
        }

        uint32_t loadedFiles = 0;
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                if (LoadRAGDataFromFile(entry.path().string())) {
                    loadedFiles++;
                }
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} JSON files from {}", loadedFiles, directoryPath);
        return loadedFiles > 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading directory {}: {}", directoryPath, e.what());
        return false;
    }
}

bool OllamaRAGSystem::LoadRAGDataFromFile(const std::string& filePath)
{
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Cannot open file: {}", filePath);
            return false;
        }

        nlohmann::json jsonData;
        file >> jsonData;

        if (!jsonData.is_array()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] JSON file must contain an array of entries: {}", filePath);
            return false;
        }

        uint32_t entriesLoaded = 0;
        for (const auto& item : jsonData) {
            try {
                RAGEntry entry;
                entry.id = item.value("id", "");
                entry.title = item.value("title", "");
                entry.content = item.value("content", "");

                if (entry.id.empty() || entry.content.empty()) {
                    LOG_ERROR("server.loading", "[Ollama Chat RAG] Entry missing required 'id' or 'content' field in file: {}", filePath);
                    continue;
                }

                // Load keywords array
                if (item.contains("keywords") && item["keywords"].is_array()) {
                    for (const auto& keyword : item["keywords"]) {
                        entry.keywords.push_back(keyword.get<std::string>());
                    }
                }

                // Load tags array
                if (item.contains("tags") && item["tags"].is_array()) {
                    for (const auto& tag : item["tags"]) {
                        entry.tags.push_back(tag.get<std::string>());
                    }
                }

                m_ragEntries.push_back(entry);
                entriesLoaded++;
            }
            catch (const std::exception& e) {
                LOG_ERROR("server.loading", "[Ollama Chat RAG] Error parsing entry in {}: {}", filePath, e.what());
            }
        }

        LOG_INFO("server.loading", "[Ollama Chat RAG] Loaded {} entries from {}", entriesLoaded, filePath);
        return entriesLoaded > 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error loading file {}: {}", filePath, e.what());
        return false;
    }
}

bool OllamaRAGSystem::SaveNewRAGEntry(const std::string& id, const std::string& title, const std::string& content, const std::vector<std::string>& keywords, const std::vector<std::string>& tags)
{
    try {
        std::string filename = std::string(g_RAGDataPath) + "/" + id + ".json";
        
        nlohmann::json entryJson;
        entryJson["id"] = id;
        entryJson["title"] = title;
        entryJson["content"] = content;
        entryJson["keywords"] = keywords;
        entryJson["tags"] = tags;
        
        nlohmann::json arrayJson = nlohmann::json::array();
        arrayJson.push_back(entryJson);
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("server.loading", "[Ollama Chat RAG] Cannot open file for writing: {}", filename);
            return false;
        }
        
        file << arrayJson.dump(4);
        file.close();
        
        // Update in-memory structures if already initialized
        if (m_initialized) {
            std::lock_guard<std::mutex> lock(m_mutex);
            RAGEntry newEntry;
            newEntry.id = id;
            newEntry.title = title;
            newEntry.content = content;
            newEntry.keywords = keywords;
            newEntry.tags = tags;
            
            // Remove existing entry with same ID if it exists
            m_ragEntries.erase(std::remove_if(m_ragEntries.begin(), m_ragEntries.end(),
                [&id](const RAGEntry& e) { return e.id == id; }), m_ragEntries.end());
                
            m_ragEntries.push_back(newEntry);
            
            // Rebuild vocabulary (simple append - O(1) with unordered_set)
            auto tokens = TokenizeText(PreprocessText(title + " " + content));
            for (const auto& token : tokens) {
                if (m_vocabSet.insert(token).second) {
                    m_vocabulary.push_back(token);
                }
            }
            for (const auto& keyword : keywords) {
                auto keywordTokens = TokenizeText(PreprocessText(keyword));
                for (const auto& token : keywordTokens) {
                    if (m_vocabSet.insert(token).second) {
                        m_vocabulary.push_back(token);
                    }
                }
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("server.loading", "[Ollama Chat RAG] Error saving RAG entry: {}", e.what());
        return false;
    }
}

std::vector<RAGResult> OllamaRAGSystem::RetrieveRelevantInfo(const std::string& query, uint32_t maxResults, float similarityThreshold)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<RAGResult> results;

    if (!m_initialized || query.empty()) {
        return results;
    }

    for (const auto& entry : m_ragEntries) {
        float similarity = CalculateSimilarity(query, entry);
        if (similarity >= similarityThreshold) {
            results.push_back({&entry, similarity});
        }
    }

    // Sort by similarity (highest first)
    std::sort(results.begin(), results.end(),
              [](const RAGResult& a, const RAGResult& b) {
                  return a.similarity > b.similarity;
              });

    // Limit results
    if (results.size() > maxResults) {
        results.resize(maxResults);
    }

    return results;
}

std::string OllamaRAGSystem::GetFormattedRAGInfo(const std::vector<RAGResult>& results)
{
    if (results.empty()) {
        return "";
    }

    std::stringstream ss;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        ss << "- " << result.entry->title << ": " << result.entry->content;
        if (i < results.size() - 1) {
            ss << "\n";
        }
    }

    return ss.str();
}

float OllamaRAGSystem::CalculateSimilarity(const std::string& query, const RAGEntry& entry)
{
    // Combine entry content with keywords for better matching
    std::string entryText = entry.title + " " + entry.content;
    for (const auto& keyword : entry.keywords) {
        entryText += " " + keyword;
    }

    // Simple TF-IDF like similarity using term frequency vectors
    auto queryVector = TextToTFVector(PreprocessText(query), m_vocabulary);
    auto entryVector = TextToTFVector(PreprocessText(entryText), m_vocabulary);

    return CalculateCosineSimilarity(queryVector, entryVector);
}

std::string OllamaRAGSystem::PreprocessText(const std::string& text) const
{
    std::string result = text;
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);

    // Remove punctuation (simple approach)
    result.erase(std::remove_if(result.begin(), result.end(),
                                [](char c) { return std::ispunct(c); }), result.end());

    return result;
}

std::vector<std::string> OllamaRAGSystem::TokenizeText(const std::string& text) const
{
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    while (ss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

float OllamaRAGSystem::CalculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) const
{
    if (vec1.size() != vec2.size()) {
        return 0.0f;
    }

    float dotProduct = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f;
    }

    return dotProduct / (norm1 * norm2);
}

std::vector<float> OllamaRAGSystem::TextToTFVector(const std::string& text, const std::vector<std::string>& vocabulary) const
{
    auto tokens = TokenizeText(text);
    std::unordered_map<std::string, int> termFreq;

    // Count term frequencies
    for (const auto& token : tokens) {
        termFreq[token]++;
    }

    // Create TF vector
    std::vector<float> vector(vocabulary.size(), 0.0f);
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        auto it = termFreq.find(vocabulary[i]);
        if (it != termFreq.end()) {
            vector[i] = static_cast<float>(it->second);
        }
    }

    return vector;
}