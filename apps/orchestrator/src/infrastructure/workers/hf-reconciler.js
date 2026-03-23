'use strict';

const https = require('https');
const { db } = require('../db/db');
const { nowIso, addMinutes } = require('../../utils/time');
const { parseJson, toJson } = require('../../utils/json');
const { newId } = require('../../utils/ids');

function hfRequest(path, token) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'huggingface.co',
      port: 443,
      path: `/api/${path}`,
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(parseJson(data, {}));
        } else {
          reject(new Error(`HF API returned ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', (e) => { reject(e); });
    req.end();
  });
}

async function reconcileHfSync(syncState) {
  const now = nowIso();
  const hfToken = process.env.HF_TOKEN;
  if (!hfToken) {
    throw new Error('HF_TOKEN environment variable is missing');
  }

  const repoPath = `${syncState.repo_type}s/${syncState.repo_id}`;
  const hfData = await hfRequest(repoPath, hfToken);

  const manifest = {
    ...parseJson(syncState.manifest_json, {}),
    hfData,
    syncedAt: now,
  };

  await db.transaction(async (trx) => {
    await trx('huggingface_sync_states').where({ id: syncState.id }).update({
      status: 'synced',
      last_synced_at: now,
      last_seen_revision: hfData.sha || null,
      manifest_json: toJson(manifest),
      updated_at: now,
    });

    if (Array.isArray(hfData.siblings)) {
      for (const file of hfData.siblings) {
        await trx('job_artifacts').insert({
          id: newId('art'),
          job_id: syncState.job_id,
          attempt_id: syncState.attempt_id,
          artifact_type: 'hf_file',
          role: 'hf_repo_file',
          backend: 'huggingface',
          uri: `https://huggingface.co/${syncState.repo_id}/blob/main/${file.rfilename}`,
          storage_key: file.rfilename,
          metadata_json: toJson({ repoId: syncState.repo_id, rfilename: file.rfilename }),
          sync_status: 'synced',
          created_at: now,
        });
      }
    }

    await trx('job_events').insert({
      id: newId('evt'),
      job_id: syncState.job_id,
      attempt_id: syncState.attempt_id,
      event_type: 'hf_sync.completed',
      severity: 'info',
      delivery_id: newId('delivery'),
      event_time: now,
      received_at: now,
      payload_json: toJson({ syncStateId: syncState.id, repoId: syncState.repo_id }),
    });
  });
}

async function processHfSyncs() {
  const pending = await db('huggingface_sync_states')
    .whereIn('status', ['pending', 'retry'])
    .andWhere(function() {
      this.whereNull('next_retry_at').orWhere('next_retry_at', '<=', nowIso());
    });

  for (const syncState of pending) {
    try {
      await reconcileHfSync(syncState);
      console.log(`Synced HF repo ${syncState.repo_id} for job ${syncState.job_id}`);
    } catch (err) {
      console.error(`Error syncing HF repo ${syncState.repo_id}:`, err);
      const retryCount = (syncState.retry_count || 0) + 1;
      const nextRetry = addMinutes(new Date(), Math.pow(2, retryCount)).toISOString();

      await db('huggingface_sync_states').where({ id: syncState.id }).update({
        status: retryCount > 5 ? 'failed' : 'retry',
        retry_count: retryCount,
        next_retry_at: nextRetry,
        last_error: err.message,
        updated_at: nowIso(),
      });
    }
  }
}

function startHfReconciler(intervalMs = 60000) {
  setInterval(() => {
    processHfSyncs().catch(err => console.error('Error in HF reconciler:', err));
  }, intervalMs);
}

module.exports = { startHfReconciler, processHfSyncs };
