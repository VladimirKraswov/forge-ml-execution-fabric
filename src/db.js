const fs = require('fs');
const path = require('path');
const knex = require('knex');
const { CONFIG } = require('./config');

fs.mkdirSync(path.dirname(CONFIG.dbFile), { recursive: true });

const db = knex({
  client: 'sqlite3',
  connection: {
    filename: CONFIG.dbFile,
  },
  useNullAsDefault: true,
  pool: {
    afterCreate(connection, done) {
      connection.run('PRAGMA journal_mode = WAL;', () => {
        connection.run('PRAGMA synchronous = NORMAL;', () => {
          connection.run('PRAGMA foreign_keys = ON;', () => {
            connection.run('PRAGMA busy_timeout = 5000;', done);
          });
        });
      });
    },
  },
  migrations: {
    directory: path.join(__dirname, 'db', 'migrations'),
  },
});

async function initDb() {
  await db.migrate.latest();
}

module.exports = { db, initDb };
