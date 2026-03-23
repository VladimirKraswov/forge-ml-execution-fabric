'use strict';

const { db } = require('../db/db');
const { nowIso, addMinutes } = require('../../utils/time');
const { newId } = require('../../utils/ids');
const { toJson } = require('../../utils/json');

async function detectLostJobs(thresholdMinutes = 10) {
  const threshold = addMinutes(new Date(), -thresholdMinutes);

  const lostAttempts = await db('job_attempts')
    .whereIn('status', ['queued', 'started', 'running', 'finalizing'])
    .andWhere('last_seen_at', '<', threshold);

  for (const attempt of lostAttempts) {
    const now = nowIso();
    await db.transaction(async (trx) => {
      await trx('job_attempts').where({ id: attempt.id }).update({
        status: 'lost',
        updated_at: now,
      });

      const job = await trx('jobs').where({ id: attempt.job_id }).first();
      if (job && !['finished', 'failed', 'cancelled'].includes(job.status)) {
        await trx('jobs').where({ id: job.id }).update({
          status: 'lost',
          headline: 'Job marked as lost due to inactivity',
          updated_at: now,
        });
      }

      await trx('job_events').insert({
        id: newId('evt'),
        job_id: attempt.job_id,
        attempt_id: attempt.id,
        event_type: 'job.lost',
        severity: 'error',
        delivery_id: newId('delivery'),
        event_time: now,
        received_at: now,
        payload_json: toJson({
          lastSeenAt: attempt.last_seen_at,
          thresholdMinutes
        }),
      });
    });
    console.log(`Marked job ${attempt.job_id} (attempt ${attempt.id}) as lost`);
  }
}

function startLostJobDetector(intervalMs = 60000) {
  setInterval(() => {
    detectLostJobs().catch(err => console.error('Error in lost job detector:', err));
  }, intervalMs);
}

module.exports = { startLostJobDetector, detectLostJobs };
