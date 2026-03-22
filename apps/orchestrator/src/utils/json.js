function parseJson(value, fallback = null) {
  if (value == null || value === '') return fallback;
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function toJson(value, fallback = {}) {
  return JSON.stringify(value ?? fallback);
}

module.exports = {
  parseJson,
  toJson,
};
