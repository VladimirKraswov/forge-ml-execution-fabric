const { verifyJwt } = require('../../infrastructure/auth/auth-service');

function authRequired(req, res, next) {
  const header = String(req.headers.authorization || '');
  if (!header.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const token = header.slice('Bearer '.length).trim();
  const claims = verifyJwt(token);

  if (!claims) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  req.user = claims;
  next();
}

module.exports = { authRequired };
