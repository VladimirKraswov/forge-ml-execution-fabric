const crypto = require('crypto');

function newId(prefix) {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, '')}`;
}

module.exports = { newId };
