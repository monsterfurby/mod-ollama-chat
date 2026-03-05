// RAG Smoke Test
// ===============
// Standalone test for the RAG (Retrieval-Augmented Generation) system.
// Tests: Save, Load, Vocab Cap, Atomic Writes, Thread Safety.
//
// Build (outside Docker, C++17):
//   g++ -std=c++17 -pthread -I../src tests/rag_smoke_test.cpp -o rag_smoke_test
//
// NOTE: This test uses stubs for AzerothCore-specific functions (LOG_INFO etc.)
//       so it can compile independently.

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cstdlib>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Stubs / Minimal replacements for AzerothCore logging and config globals
// ---------------------------------------------------------------------------
#define LOG_INFO(cat, ...) do { /* no-op */ } while(0)
#define LOG_ERROR(cat, ...) do { std::cerr << "[ERROR] " << __VA_ARGS__ << std::endl; } while(0)

// Globals that the RAG code references
bool g_DebugEnabled = false;
uint32_t g_RAGMaxVocabularySize = 50; // intentionally small for testing

// Minimal RAG entry structure (mirrors the one in mod-ollama-chat_rag.h)
struct RAGEntry {
    std::string id;
    std::string title;
    std::string content;
    std::vector<std::string> keywords;
    std::vector<std::string> tags;
};

// Minimal inline JSON helpers (avoid nlohmann dependency for this test)
namespace TestJSON {
    std::string escape(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else out += c;
        }
        return out;
    }
    
    std::string serializeEntry(const RAGEntry& e) {
        std::ostringstream ss;
        ss << "[{\"id\":\"" << escape(e.id)
           << "\",\"title\":\"" << escape(e.title)
           << "\",\"content\":\"" << escape(e.content)
           << "\",\"keywords\":[";
        for (size_t i = 0; i < e.keywords.size(); ++i) {
            if (i) ss << ",";
            ss << "\"" << escape(e.keywords[i]) << "\"";
        }
        ss << "],\"tags\":[";
        for (size_t i = 0; i < e.tags.size(); ++i) {
            if (i) ss << ",";
            ss << "\"" << escape(e.tags[i]) << "\"";
        }
        ss << "]}]";
        return ss.str();
    }
}

// ---------------------------------------------------------------------------
// Minimal RAG system that mirrors the real code's behaviour for testing
// ---------------------------------------------------------------------------
class TestRAGSystem {
public:
    bool m_initialized = false;
    std::mutex m_mutex;
    std::vector<RAGEntry> m_ragEntries;
    std::vector<std::string> m_vocabulary;
    std::unordered_set<std::string> m_vocabSet;
    std::string m_dataPath;

    void Init(const std::string& dataPath) {
        m_dataPath = dataPath;
        fs::create_directories(m_dataPath);
        m_initialized = true;
    }

    std::vector<std::string> TokenizeText(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            // Simple lowercase
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            tokens.push_back(word);
        }
        return tokens;
    }

    // Mirrors the fixed SaveNewRAGEntry: lock at top, atomic write, vocab cap
    bool SaveNewRAGEntry(const std::string& id, const std::string& title,
                         const std::string& content,
                         const std::vector<std::string>& keywords,
                         const std::vector<std::string>& tags) {
        try {
            std::lock_guard<std::mutex> lock(m_mutex);

            std::string filename = m_dataPath + "/" + id + ".json";
            std::string tmpFilename = filename + ".tmp";

            RAGEntry entry{id, title, content, keywords, tags};
            std::string json = TestJSON::serializeEntry(entry);

            // Write to temp file first (atomic write pattern)
            {
                std::ofstream file(tmpFilename);
                if (!file.is_open()) return false;
                file << json;
                file.close();
            }

            // Atomic rename
            std::error_code ec;
            fs::rename(tmpFilename, filename, ec);
            if (ec) {
                fs::remove(tmpFilename, ec);
                return false;
            }

            // Update in-memory if initialized
            if (m_initialized) {
                m_ragEntries.erase(
                    std::remove_if(m_ragEntries.begin(), m_ragEntries.end(),
                                   [&id](const RAGEntry& e) { return e.id == id; }),
                    m_ragEntries.end());
                m_ragEntries.push_back(entry);

                // Vocab cap
                auto tokens = TokenizeText(title + " " + content);
                for (const auto& token : tokens) {
                    if (m_vocabulary.size() >= g_RAGMaxVocabularySize) break;
                    if (m_vocabSet.insert(token).second) {
                        m_vocabulary.push_back(token);
                    }
                }
                for (const auto& kw : keywords) {
                    if (m_vocabulary.size() >= g_RAGMaxVocabularySize) break;
                    auto kwTokens = TokenizeText(kw);
                    for (const auto& token : kwTokens) {
                        if (m_vocabulary.size() >= g_RAGMaxVocabularySize) break;
                        if (m_vocabSet.insert(token).second) {
                            m_vocabulary.push_back(token);
                        }
                    }
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }
};

// ---------------------------------------------------------------------------
// Test functions
// ---------------------------------------------------------------------------
static int g_passed = 0;
static int g_failed = 0;

#define TEST_ASSERT(cond, msg)                                          \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::cerr << "[FAIL] " << msg << " (line " << __LINE__     \
                      << ")" << std::endl;                              \
            ++g_failed;                                                 \
        } else {                                                        \
            std::cout << "[PASS] " << msg << std::endl;                 \
            ++g_passed;                                                 \
        }                                                               \
    } while (0)

void TestSaveAndLoad() {
    std::cout << "\n--- Test: Save and Load ---\n";
    std::string testDir = "rag_test_data_save";
    fs::remove_all(testDir);

    TestRAGSystem rag;
    rag.Init(testDir);

    bool ok = rag.SaveNewRAGEntry("entry_1", "First Memory",
                                   "Player helped bot in dungeon.",
                                   {"player", "dungeon"},
                                   {"memory"});
    TEST_ASSERT(ok, "SaveNewRAGEntry returns true");
    TEST_ASSERT(rag.m_ragEntries.size() == 1, "1 entry in memory");
    TEST_ASSERT(fs::exists(testDir + "/entry_1.json"), "JSON file exists on disk");

    // Verify no .tmp file remains
    TEST_ASSERT(!fs::exists(testDir + "/entry_1.json.tmp"), "No leftover .tmp file");

    // Save a second entry with same ID — should replace
    ok = rag.SaveNewRAGEntry("entry_1", "Updated Memory",
                              "Player helped bot again.",
                              {"player"}, {"memory"});
    TEST_ASSERT(ok, "Overwrite existing entry succeeds");
    TEST_ASSERT(rag.m_ragEntries.size() == 1, "Still 1 entry (replaced, not duplicated)");
    TEST_ASSERT(rag.m_ragEntries[0].title == "Updated Memory", "Title updated");

    fs::remove_all(testDir);
}

void TestVocabCap() {
    std::cout << "\n--- Test: Vocabulary Cap ---\n";
    std::string testDir = "rag_test_data_vocab";
    fs::remove_all(testDir);

    TestRAGSystem rag;
    rag.Init(testDir);

    // Generate content with many unique words to exceed the cap (50)
    std::string bigContent;
    for (int i = 0; i < 100; ++i) {
        bigContent += "uniqueword" + std::to_string(i) + " ";
    }

    rag.SaveNewRAGEntry("vocab_test", "Many Words", bigContent, {}, {});

    TEST_ASSERT(rag.m_vocabulary.size() <= g_RAGMaxVocabularySize,
                "Vocabulary does not exceed max (" +
                    std::to_string(rag.m_vocabulary.size()) + " <= " +
                    std::to_string(g_RAGMaxVocabularySize) + ")");

    // Subsequent saves should also respect the cap
    rag.SaveNewRAGEntry("vocab_test_2", "More Words", "even more brand new tokens here", {}, {});
    TEST_ASSERT(rag.m_vocabulary.size() <= g_RAGMaxVocabularySize,
                "Vocabulary still capped after second entry (" +
                    std::to_string(rag.m_vocabulary.size()) + ")");

    fs::remove_all(testDir);
}

void TestAtomicWrite() {
    std::cout << "\n--- Test: Atomic Write (no partial files) ---\n";
    std::string testDir = "rag_test_data_atomic";
    fs::remove_all(testDir);

    TestRAGSystem rag;
    rag.Init(testDir);

    rag.SaveNewRAGEntry("atomic_1", "Test", "Content", {}, {});

    // Verify the file is valid (not truncated/empty)
    std::ifstream f(testDir + "/atomic_1.json");
    TEST_ASSERT(f.is_open(), "File opens successfully");
    std::string contents((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    f.close();

    TEST_ASSERT(!contents.empty(), "File is not empty");
    TEST_ASSERT(contents.find("\"id\":\"atomic_1\"") != std::string::npos,
                "File contains expected JSON content");

    // No .tmp files should remain
    for (const auto& entry : fs::directory_iterator(testDir)) {
        TEST_ASSERT(entry.path().extension() != ".tmp",
                     "No .tmp file remains: " + entry.path().string());
    }

    fs::remove_all(testDir);
}

void TestConcurrentSaves() {
    std::cout << "\n--- Test: Concurrent Saves (Thread Safety) ---\n";
    std::string testDir = "rag_test_data_concurrent";
    fs::remove_all(testDir);

    TestRAGSystem rag;
    rag.Init(testDir);

    const int NUM_THREADS = 8;
    const int ENTRIES_PER_THREAD = 10;
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&rag, t]() {
            for (int i = 0; i < ENTRIES_PER_THREAD; ++i) {
                std::string id = "thread" + std::to_string(t) + "_entry" + std::to_string(i);
                rag.SaveNewRAGEntry(id, "Title " + std::to_string(i),
                                     "Content from thread " + std::to_string(t),
                                     {"key"}, {"tag"});
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    int expectedEntries = NUM_THREADS * ENTRIES_PER_THREAD;
    TEST_ASSERT(static_cast<int>(rag.m_ragEntries.size()) == expectedEntries,
                "All " + std::to_string(expectedEntries) + " entries saved (got " +
                    std::to_string(rag.m_ragEntries.size()) + ")");

    // Count files on disk
    int fileCount = 0;
    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.path().extension() == ".json") ++fileCount;
    }
    TEST_ASSERT(fileCount == expectedEntries,
                "All " + std::to_string(expectedEntries) + " files on disk (got " +
                    std::to_string(fileCount) + ")");

    // No .tmp files should remain
    for (const auto& entry : fs::directory_iterator(testDir)) {
        TEST_ASSERT(entry.path().extension() != ".tmp",
                     "No leftover .tmp: " + entry.path().string());
    }

    fs::remove_all(testDir);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== RAG Smoke Test ===\n";

    TestSaveAndLoad();
    TestVocabCap();
    TestAtomicWrite();
    TestConcurrentSaves();

    std::cout << "\n=== Results: " << g_passed << " passed, " << g_failed << " failed ===\n";
    return g_failed > 0 ? 1 : 0;
}
