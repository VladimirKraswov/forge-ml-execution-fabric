const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { db } = require('../db');
const { CONFIG } = require('../config');
const { newId } = require('../utils/ids');
const { nowIso } = require('../utils/time');

async function seedAdminUser() {
  const existing = await db('users').where({ username: CONFIG.adminUsername }).first();
  if (existing) return existing;

  const passwordHash = await bcrypt.hash(CONFIG.adminPassword, 10);
  const now = nowIso();
  const user = {
    id: newId('usr'),
    username: CONFIG.adminUsername,
    password_hash: passwordHash,
    role: 'admin',
    created_at: now,
    updated_at: now,
  };
  await db('users').insert(user);
  return user;
}

async function login(username, password) {
  const user = await db('users').where({ username }).first();
  if (!user) return null;

  const valid = await bcrypt.compare(String(password || ''), user.password_hash);
  if (!valid) return null;

  const token = jwt.sign(
    {
      sub: user.id,
      username: user.username,
      role: user.role,
    },
    CONFIG.jwtSecret,
    { expiresIn: '7d' }
  );

  return {
    user: {
      id: user.id,
      username: user.username,
      role: user.role,
    },
    token,
  };
}

function verifyJwt(token) {
  try {
    return jwt.verify(token, CONFIG.jwtSecret);
  } catch {
    return null;
  }
}

module.exports = {
  seedAdminUser,
  login,
  verifyJwt,
};
