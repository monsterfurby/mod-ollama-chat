#include "mod-ollama-chat_querymanager.h"
#include "mod-ollama-chat_config.h"  // For g_MaxConcurrentQueries and OpenRouter config
#include "Log.h"
#include <thread>

// Constructor: initialize with the configuration value.
QueryManager::QueryManager()
    : maxConcurrentQueries(g_MaxConcurrentQueries), currentQueries(0)
{
}

// Set maximum concurrent queries (0 means no limit).
void QueryManager::setMaxConcurrentQueries(int maxQueries) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxConcurrentQueries = maxQueries;
}

// Submit a query and return a future for the result.
std::future<std::string> QueryManager::submitQuery(const std::string& prompt) {
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();

    bool shouldRunNow = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Handle OpenRouter Rate Limiting
        if (g_UseOpenRouter && g_OpenRouterMaxCallsPerPeriod > 0) {
            auto now = std::chrono::steady_clock::now();
            auto period = std::chrono::seconds(g_OpenRouterPeriodLengthSeconds);

            // Clean up old timestamps outside the tracking window
            while (!openRouterCallTimestamps.empty() && (now - openRouterCallTimestamps.front()) > period) {
                openRouterCallTimestamps.pop_front();
            }

            // Check if we hit the limit
            if (openRouterCallTimestamps.size() >= g_OpenRouterMaxCallsPerPeriod) {
                if (g_DebugEnabled) {
                    LOG_INFO("server.loading", "[Ollama Chat] OpenRouter rate limit exceeded ({} calls per {} seconds). Dropping query.", 
                             openRouterCallTimestamps.size(), g_OpenRouterPeriodLengthSeconds);
                }
                // Rate limit hit: resolve immediately with an empty string rather than querying
                promise.set_value("");
                return future;
            } else {
                // We're under the limit, record this call
                openRouterCallTimestamps.push_back(now);
            }
        }

        if (maxConcurrentQueries == 0 || currentQueries < maxConcurrentQueries) {
            ++currentQueries;
            shouldRunNow = true;
        } else {
            taskQueue.push({ prompt, std::move(promise) });
        }
    }

    if (shouldRunNow) {
        std::thread(&QueryManager::processQuery, this, prompt, std::move(promise)).detach();
    }

    return future;
}

// Process the query by calling the API and then handling any queued tasks.
void QueryManager::processQuery(const std::string& prompt, std::promise<std::string> promise) {
    std::string result = QueryOllamaAPI(prompt);
    promise.set_value(result);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        --currentQueries;
        if (!taskQueue.empty() && (maxConcurrentQueries == 0 || currentQueries < maxConcurrentQueries)) {
            QueryTask task = std::move(taskQueue.front());
            taskQueue.pop();
            ++currentQueries;
            std::thread(&QueryManager::processQuery, this, task.prompt, std::move(task.promise)).detach();
        }
    }
}
