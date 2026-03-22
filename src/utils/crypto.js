const crypto = require('crypto');

function hashToken(rawToken) {
  return crypto.createHash('sha256').update(String(rawToken || '')).digest('hex');
}

function issueOpaqueToken(prefix = 'tok') {
  return `${prefix}_${crypto.randomBytes(24).toString('hex')}`;
}

module.exports = {
  hashToken,
  issueOpaqueToken,
};
