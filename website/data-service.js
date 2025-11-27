/**
 * Scarlet Sails Dashboard Data Service
 * 
 * Provides read-only access to repository data via GitHub raw content URLs.
 * All data is fetched from the main repository without authentication.
 * If data is missing or malformed, graceful fallbacks are provided.
 * 
 * Key principles:
 * - NO secrets, API keys, or authentication tokens
 * - NO data fabrication; missing data shows as "-- (No data)"
 * - All fetches use public GitHub raw content URLs
 * - Implements simple caching with 5-minute TTL
 * - All errors are caught and converted to fallback values
 */

const DataService = (() => {
  const CONFIG = {
    REPO_OWNER: 'Antihrist-star',
    REPO_NAME: 'ScArlet-Sails',
    BRANCH: 'main',
    RAW_URL: 'https://raw.githubusercontent.com',
    CACHE_TTL: 5 * 60 * 1000, // 5 minutes in ms
  };

  // Simple in-memory cache
  const cache = new Map();

  /**
   * Internal: Get cache key for a path
   */
  function getCacheKey(path) {
    return `${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}/${CONFIG.BRANCH}${path}`;
  }

  /**
   * Internal: Check if cache is still valid
   */
  function isCacheValid(key) {
    const entry = cache.get(key);
    if (!entry) return false;
    const now = Date.now();
    return (now - entry.timestamp) < CONFIG.CACHE_TTL;
  }

  /**
   * Internal: Get or fetch from cache
   */
  async function getCachedOrFetch(path, fallback, parser) {
    const key = getCacheKey(path);
    
    // Return cached value if valid
    if (isCacheValid(key)) {
      return cache.get(key).value;
    }

    // Fetch fresh data
    try {
      const url = `${CONFIG.RAW_URL}/${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}/${CONFIG.BRANCH}${path}`;
      const response = await fetch(url, { cache: 'no-store' });
      
      if (!response.ok) {
        console.warn(`Data fetch failed for ${path}: ${response.status}`);
        return fallback;
      }

      const data = await response.text();
      let result = data;

      // Apply parser if provided
      if (parser) {
        result = parser(data);
      }

      // Cache the result
      cache.set(key, {
        value: result,
        timestamp: Date.now(),
      });

      return result;
    } catch (error) {
      console.error(`Data service error for ${path}:`, error.message);
      return fallback;
    }
  }

  /**
   * Fetch raw text file from repository
   */
  async function fetchText(path, fallback = '-- (No data available)') {
    return getCachedOrFetch(path, fallback);
  }

  /**
   * Fetch and parse JSON file from repository
   */
  async function fetchJSON(path, fallback = null) {
    const parser = (text) => {
      try {
        return JSON.parse(text);
      } catch (e) {
        console.error(`JSON parse error for ${path}:`, e.message);
        return fallback || {};
      }
    };
    return getCachedOrFetch(path, fallback || {}, parser);
  }

  /**
   * Parse README.md for a specific field
   * Looks for "Field: value" patterns
   */
  async function parseReadmeField(fieldName, fallback = '-- (No data)') {
    try {
      const readme = await fetchText('/README.md');
      const regex = new RegExp(`\\b${fieldName}\\s*:\\s*([^\\n]+)`, 'i');
      const match = readme.match(regex);
      return match ? match[1].trim() : fallback;
    } catch (error) {
      console.error(`Failed to parse README field ${fieldName}:`, error.message);
      return fallback;
    }
  }

  /**
   * Get latest result file from results directory
   * Assumes results are JSON files with timestamps in name or content
   */
  async function getLatestResult(fallback = null) {
    try {
      // Try to fetch a standard results file
      // This would need actual endpoint discovery in production
      // For now, attempt common patterns
      const possiblePaths = [
        '/results/latest_result.json',
        '/results/backtest_results.json',
      ];

      for (const path of possiblePaths) {
        try {
          const data = await fetchJSON(path);
          if (data && Object.keys(data).length > 0) {
            return data;
          }
        } catch (e) {
          // Continue to next path
        }
      }

      return fallback || {};
    } catch (error) {
      console.error('Failed to get latest result:', error.message);
      return fallback || {};
    }
  }

  /**
   * Extract metric from result data
   * Looks for common metric field names
   */
  function extractMetric(resultData, metricName, fallback = '-- (No data)') {
    if (!resultData || typeof resultData !== 'object') {
      return fallback;
    }

    // Try direct field access
    if (metricName in resultData) {
      const value = resultData[metricName];
      return value !== null && value !== undefined ? value : fallback;
    }

    // Try snake_case version
    const snakeCase = metricName.replace(/([A-Z])/g, '_$1').toLowerCase();
    if (snakeCase in resultData) {
      const value = resultData[snakeCase];
      return value !== null && value !== undefined ? value : fallback;
    }

    return fallback;
  }

  /**
   * Get Sharpe Ratio from latest result
   */
  async function getSharpeRatio() {
    const result = await getLatestResult();
    return extractMetric(result, 'sharpe_ratio', '-- (No data)');
  }

  /**
   * Get Win Rate from latest result
   */
  async function getWinRate() {
    const result = await getLatestResult();
    return extractMetric(result, 'win_rate', '-- (No data)');
  }

  /**
   * Get Total Trades from latest result
   */
  async function getTotalTrades() {
    const result = await getLatestResult();
    const value = extractMetric(result, 'total_trades', null);
    return value !== null ? value : '-- (No data)';
  }

  /**
   * Get list of model files from models directory
   * Returns array of {name, path}
   */
  async function listModels() {
    try {
      // Since GitHub API for directory listing requires authentication,
      // we'll try fetching known model files
      const knownModels = [
        { name: 'P_rb', path: '/models/P_rb.py' },
        { name: 'P_mi', path: '/models/P_mi.py' },
        { name: 'P_hyb', path: '/models/P_hyb.py' },
      ];

      const availableModels = [];

      for (const model of knownModels) {
        try {
          const exists = await fetch(
            `${CONFIG.RAW_URL}/${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}/${CONFIG.BRANCH}${model.path}`,
            { method: 'HEAD' }
          );
          if (exists.ok) {
            availableModels.push(model);
          }
        } catch (e) {
          // Model doesn't exist, skip
        }
      }

      return availableModels.length > 0 ? availableModels : [];
    } catch (error) {
      console.error('Failed to list models:', error.message);
      return [];
    }
  }

  /**
   * Get model source code
   */
  async function getModelCode(modelName) {
    const path = `/models/${modelName}.py`;
    return fetchText(path, `-- (Model ${modelName} not found)`);
  }

  /**
   * Get README content
   */
  async function getReadme() {
    return fetchText('/README.md', '-- (README not found)');
  }

  /**
   * Parse model description from README
   * Looks for sections like "## Model: P_rb" or similar
   */
  async function getModelDescription(modelName) {
    try {
      const readme = await getReadme();
      const regex = new RegExp(
        `## .*?${modelName}.*?\\n\\n([\\s\\S]*?)(?=\\n##|$)`,
        'i'
      );
      const match = readme.match(regex);
      return match ? match[1].trim() : `-- (No description for ${modelName})`;
    } catch (error) {
      console.error(`Failed to get model description for ${modelName}:`, error);
      return `-- (No description for ${modelName})`;
    }
  }

  /**
   * Clear cache (useful for testing or manual refresh)
   */
  function clearCache() {
    cache.clear();
  }

  /**
   * Get cache statistics (for debugging)
   */
  function getCacheStats() {
    return {
      size: cache.size,
      keys: Array.from(cache.keys()),
    };
  }

  // Public API
  return {
    // Configuration
    config: () => CONFIG,

    // Raw data fetching
    fetchText,
    fetchJSON,

    // README parsing
    parseReadmeField,
    getReadme,

    // Results and metrics
    getLatestResult,
    extractMetric,
    getSharpeRatio,
    getWinRate,
    getTotalTrades,

    // Models
    listModels,
    getModelCode,
    getModelDescription,

    // Cache management
    clearCache,
    getCacheStats,
  };
})();

// Export for use in HTML via <script> tag
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DataService;
}
