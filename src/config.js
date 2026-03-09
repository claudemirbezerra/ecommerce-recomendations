/**
 * App configuration.
 * API_URL: Vector storage backend (default: same host, port 3001).
 */
export const config = {
  apiUrl: window.API_URL || `${window.location.protocol}//${window.location.hostname}:3001`,
};
