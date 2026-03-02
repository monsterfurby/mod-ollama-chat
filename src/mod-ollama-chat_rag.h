#ifndef MOD_OLLAMA_CHAT_RAG_H
#define MOD_OLLAMA_CHAT_RAG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <nlohmann/json.hpp>

struct RAGEntry {
    std::string id;
    std::string title;
    std::string content;
    std::vector<std::string> keywords;
    std::vector<std::string> tags;
};

struct RAGResult {
    const RAGEntry* entry;
    float similarity;
};

class OllamaRAGSystem {
public:
    OllamaRAGSystem();
    ~OllamaRAGSystem();

    // Initialize the RAG system by loading JSON data files
    bool Initialize();

    // Retrieve relevant information based on a query
    std::vector<RAGResult> RetrieveRelevantInfo(const std::string& query, uint32_t maxResults = 3, float similarityThreshold = 0.3f);

    // Get formatted RAG information for prompt inclusion
    std::string GetFormattedRAGInfo(const std::vector<RAGResult>& results);

    // Save a new RAG entry to a file and update the index
    bool SaveNewRAGEntry(const std::string& id, const std::string& title, const std::string& content, const std::vector<std::string>& keywords = {}, const std::vector<std::string>& tags = {});

private:
    // Load RAG data from JSON files in the specified directory
    bool LoadRAGDataFromDirectory(const std::string& directoryPath);

    // Load a single JSON file
    bool LoadRAGDataFromFile(const std::string& filePath);

    // Calculate similarity between query and entry
    float CalculateSimilarity(const std::string& query, const RAGEntry& entry);

    // Simple text preprocessing (lowercase, remove punctuation)
    std::string PreprocessText(const std::string& text) const;

    // Split text into words
    std::vector<std::string> TokenizeText(const std::string& text) const;

    // Calculate cosine similarity between two vectors
    float CalculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) const;

    // Convert text to simple TF vector (term frequency)
    std::vector<float> TextToTFVector(const std::string& text, const std::vector<std::string>& vocabulary) const;

private:
    std::vector<RAGEntry> m_ragEntries;
    std::vector<std::string> m_vocabulary;
    std::unordered_set<std::string> m_vocabSet; // For O(1) lookups
    bool m_initialized;
    mutable std::mutex m_mutex;
};

#endif // MOD_OLLAMA_CHAT_RAG_H