#include "mod-ollama-chat_api.h"
#include "Log.h"
#include "mod-ollama-chat-utilities.h"
#include "mod-ollama-chat_config.h"
#include "mod-ollama-chat_httpclient.h"
#include <fmt/core.h>
#include <future>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>
#include <thread>


std::string ExtractTextBetweenDoubleQuotes(const std::string &response) {
  size_t first = response.find('"');
  size_t second = response.find('"', first + 1);
  if (first != std::string::npos && second != std::string::npos) {
    return response.substr(first + 1, second - first - 1);
  }
  return response;
}

// Function to perform the API call.
std::string QueryOllamaAPI(const std::string &prompt) {
  // Initialize our custom HTTP client
  static OllamaHttpClient httpClient;

  if (!httpClient.IsAvailable()) {
    LOG_ERROR("server.loading",
              "[OllamaChat] ERROR: HTTP client not available.");
    return "";
  }

  // Sanitize the prompt to ensure it's valid UTF-8 before creating JSON
  std::string sanitizedPrompt = SanitizeUTF8(prompt);

  // ---------------------------------------------------------------
  // OpenRouter path: OpenAI-compatible chat completions API
  // ---------------------------------------------------------------
  if (g_UseOpenRouter) {
    if (g_OpenRouterApiKey.empty()) {
      LOG_ERROR(
          "server.loading",
          "[OllamaChat] OpenRouter API key is empty. Cannot make request.");
      return "";
    }

    nlohmann::json messages = nlohmann::json::array();

    // System prompt injected as the system role message
    if (!g_OllamaSystemPrompt.empty()) {
      messages.push_back({{"role", "system"},
                          {"content", SanitizeUTF8(g_OllamaSystemPrompt)}});
    }

    messages.push_back({{"role", "user"}, {"content", sanitizedPrompt}});

    nlohmann::json requestData = {{"model", g_OllamaModel},
                                  {"messages", messages}};

    // Map parameters to OpenAI equivalents (all at root level, not nested in
    // "options")
    if (g_OllamaNumPredict > 0)
      requestData["max_tokens"] = g_OllamaNumPredict;

    if (g_OllamaTemperature != 0.8f)
      requestData["temperature"] = g_OllamaTemperature;

    if (g_OllamaTopP != 0.95f)
      requestData["top_p"] = g_OllamaTopP;

    // repeat_penalty in Ollama maps to frequency_penalty in OpenAI
    // Note: OpenAI uses range 0.0-2.0, Ollama uses range around 1.0
    // Remap: Ollama 1.0 == neutral -> OpenAI 0.0; scale (penalty - 1.0) to
    // OpenAI range
    if (g_OllamaRepeatPenalty != 1.1f) {
      float openaiFreqPenalty =
          std::max(0.0f, std::min(2.0f, g_OllamaRepeatPenalty - 1.0f));
      requestData["frequency_penalty"] = openaiFreqPenalty;
    }

    // Stop sequences: same field name, same array format
    if (!g_OllamaStop.empty()) {
      std::vector<std::string> stopSeqs;
      std::stringstream ss(g_OllamaStop);
      std::string item;
      while (std::getline(ss, item, ',')) {
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos)
          stopSeqs.push_back(item.substr(start, end - start + 1));
      }
      if (!stopSeqs.empty())
        requestData["stop"] = stopSeqs;
    }

    // NumCtx, NumThreads, Seed, and Think mode are Ollama-specific — not
    // forwarded
    if (g_DebugEnabled) {
      LOG_INFO("server.loading",
               "[OllamaChat] OpenRouter request to: {} model: {}",
               g_OpenRouterUrl, g_OllamaModel);
      if (g_OllamaNumCtx > 0 || g_OllamaNumThreads > 0 ||
          !g_OllamaSeed.empty() || g_ThinkModeEnableForModule) {
        LOG_INFO("server.loading",
                 "[OllamaChat] Note: NumCtx, NumThreads, Seed, and ThinkMode "
                 "are Ollama-specific and are ignored for OpenRouter.");
      }
    }

    std::string requestDataStr = requestData.dump();
    std::string authHeader = "Bearer " + g_OpenRouterApiKey;

    std::string responseBuffer =
        httpClient.Post(g_OpenRouterUrl, requestDataStr, authHeader);

    if (responseBuffer.empty()) {
      LOG_ERROR("server.loading",
                "[OllamaChat] ERROR: Failed to reach OpenRouter API at {}. "
                "Check URL, API key, and network connectivity.",
                g_OpenRouterUrl);
      return "";
    }

    try {
      nlohmann::json jsonResponse = nlohmann::json::parse(responseBuffer);

      // Surface OpenRouter/OpenAI error responses
      if (jsonResponse.contains("error")) {
        std::string errMsg =
            jsonResponse["error"].contains("message")
                ? jsonResponse["error"]["message"].get<std::string>()
                : jsonResponse["error"].dump();
        LOG_ERROR("server.loading", "[OllamaChat] OpenRouter API error: {}",
                  errMsg);
        return "";
      }

      if (!jsonResponse.contains("choices") ||
          jsonResponse["choices"].empty()) {
        LOG_ERROR("server.loading",
                  "[OllamaChat] OpenRouter: No choices in response.");
        if (g_DebugEnabled)
          LOG_INFO("server.loading", "[OllamaChat] Raw response: {}",
                   responseBuffer);
        return "";
      }

      std::string botReply =
          jsonResponse["choices"][0]["message"]["content"].get<std::string>();

      if (botReply.empty()) {
        LOG_ERROR("server.loading",
                  "[OllamaChat] OpenRouter returned an empty message content.");
        return "";
      }

      if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] OpenRouter response: {}",
                 botReply);

      return botReply;
    } catch (const std::exception &e) {
      LOG_ERROR("server.loading",
                "[OllamaChat] OpenRouter JSON parsing failed: {}", e.what());
      if (g_DebugEnabled)
        LOG_INFO("server.loading", "[OllamaChat] Raw response: {}",
                 responseBuffer);
      return "";
    }
  }

  // ---------------------------------------------------------------
  // Ollama path (original logic, unchanged)
  // ---------------------------------------------------------------
  std::string url = g_OllamaUrl;
  std::string model = g_OllamaModel;

  nlohmann::json requestData = {
      {"model", model}, {"prompt", sanitizedPrompt}, {"stream", false}};

  // Create options object for model parameters
  nlohmann::json options;
  bool hasOptions = false;

  // Only include if set (do not send defaults if user did not set them)
  if (g_OllamaNumPredict > 0) {
    options["num_predict"] = g_OllamaNumPredict;
    hasOptions = true;
  }
  if (g_OllamaTemperature != 0.8f) {
    options["temperature"] = g_OllamaTemperature;
    hasOptions = true;
  }
  if (g_OllamaTopP != 0.95f) {
    options["top_p"] = g_OllamaTopP;
    hasOptions = true;
  }
  if (g_OllamaRepeatPenalty != 1.1f) {
    options["repeat_penalty"] = g_OllamaRepeatPenalty;
    hasOptions = true;
  }
  if (g_OllamaNumCtx > 0) {
    options["num_ctx"] = g_OllamaNumCtx;
    hasOptions = true;
  }
  if (g_OllamaNumThreads > 0) {
    options["num_thread"] = g_OllamaNumThreads;
    hasOptions = true;
  }
  if (!g_OllamaSeed.empty()) {
    try {
      int seedValue = std::stoi(g_OllamaSeed);
      options["seed"] = seedValue;
      hasOptions = true;
    } catch (const std::exception &e) {
      if (g_DebugEnabled) {
        LOG_INFO("server.loading", "[Ollama Chat] Invalid seed value: {}",
                 g_OllamaSeed);
      }
    }
  }

  // Add options object if any options were set
  if (hasOptions) {
    requestData["options"] = options;
  }

  // Root-level parameters (these stay at root level)
  if (!g_OllamaStop.empty()) {
    // If comma-separated, convert to array
    std::vector<std::string> stopSeqs;
    std::stringstream ss(g_OllamaStop);
    std::string item;
    while (std::getline(ss, item, ',')) {
      // trim whitespace
      size_t start = item.find_first_not_of(" \t");
      size_t end = item.find_last_not_of(" \t");
      if (start != std::string::npos && end != std::string::npos)
        stopSeqs.push_back(item.substr(start, end - start + 1));
    }
    if (!stopSeqs.empty())
      requestData["stop"] = stopSeqs;
  }
  if (!g_OllamaSystemPrompt.empty()) {
    // Sanitize system prompt as well
    requestData["system"] = SanitizeUTF8(g_OllamaSystemPrompt);
  }

  if (g_ThinkModeEnableForModule) {
    if (g_DebugEnabled) {
      LOG_INFO("server.loading", "[Ollama Chat] LLM set to Think mode.");
    }
    requestData["think"] = true;
    requestData["hidethinking"] = true;
  }

  std::string requestDataStr = requestData.dump();

  // Make HTTP POST request using our custom client (no auth header for local
  // Ollama)
  std::string responseBuffer = httpClient.Post(url, requestDataStr);

  if (responseBuffer.empty()) {
    LOG_ERROR("server.loading",
              "[OllamaChat] ERROR: Failed to reach Ollama API at {}. Check URL "
              "configuration and network connectivity.",
              url);
    if (g_DebugEnabled) {
      LOG_INFO("server.loading",
               "[OllamaChat] Debug: Empty response buffer from HTTP client. "
               "Model: {}",
               model);
    }
    return "";
  }

  std::stringstream ss(responseBuffer);
  std::string line;
  std::ostringstream extractedResponse;

  try {
    while (std::getline(ss, line)) {
      if (line.empty() || std::all_of(line.begin(), line.end(), isspace))
        continue;

      nlohmann::json jsonResponse = nlohmann::json::parse(line);

      if (jsonResponse.contains("response") &&
          !jsonResponse["response"].get<std::string>().empty()) {
        extractedResponse << jsonResponse["response"].get<std::string>();
      }
    }
  } catch (const std::exception &e) {
    LOG_ERROR("server.loading",
              "[OllamaChat] ERROR: JSON parsing failed. Exception: {}",
              e.what());
    if (g_DebugEnabled) {
      LOG_INFO("server.loading",
               "[OllamaChat] Debug: Response buffer content: {}",
               responseBuffer);
    }
    return "";
  }

  std::string botReply = extractedResponse.str();

  botReply = ExtractTextBetweenDoubleQuotes(botReply);

  // Check for unclosed think tags
  if (botReply.find("<think>") != std::string::npos ||
      botReply.find("</think>") != std::string::npos) {
    LOG_ERROR("server.loading",
              "[OllamaChat] ERROR: Unclosed <think> tags detected in response. "
              "This usually means the model's output was truncated.");
    LOG_ERROR("server.loading",
              "[OllamaChat] SOLUTION: Set 'OllamaChat.ThinkModeEnableForModule "
              "= 1' in mod_ollama_chat.conf");
    LOG_ERROR("server.loading",
              "[OllamaChat] SOLUTION: Set 'OllamaChat.NumPredict = 0' "
              "(unlimited tokens) in mod_ollama_chat.conf");
    LOG_ERROR("server.loading",
              "[OllamaChat] SOLUTION: Set 'OllamaChat.NumCtx = 0' (model "
              "default context) in mod_ollama_chat.conf");
    if (g_DebugEnabled) {
      LOG_INFO("server.loading",
               "[OllamaChat] Debug: Partial response with think tags: {}",
               botReply);
    }
    return "";
  }

  if (botReply.empty()) {
    LOG_ERROR("server.loading",
              "[OllamaChat] ERROR: Empty response extracted from API. Model "
              "may not have generated any output.");
    if (g_DebugEnabled) {
      LOG_INFO("server.loading",
               "[OllamaChat] Debug: Raw extracted response was empty.");
    }
    return "";
  }

  if (g_DebugEnabled) {
    LOG_INFO("server.loading", "[Ollama Chat] Parsed bot response: {}",
             botReply);

    if (g_ThinkModeEnableForModule) {
      if (g_DebugEnabled) {
        LOG_INFO("server.loading", "[Ollama Chat] Bot used think.");
      }
    }
  }

  return botReply;
}

// Helper function to check if a response is valid (not empty and not an error)
bool IsValidAPIResponse(const std::string &response) {
  if (response.empty()) {
    return false;
  }
  // Response is valid if it's not empty
  return true;
}

QueryManager g_queryManager;

// Interface function to submit a query.
std::future<std::string> SubmitQuery(const std::string &prompt, bool bypassOpenRouterThrottle) {
  return g_queryManager.submitQuery(prompt, bypassOpenRouterThrottle);
}