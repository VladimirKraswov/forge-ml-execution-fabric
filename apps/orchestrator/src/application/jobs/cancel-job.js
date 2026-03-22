const { db } = require('../../infrastructure/db/db');
const { nowIso } = require('../../utils/time');
const { toJson } = require('../../utils/json');
const { newId } = require('../../utils/ids');
const { getJobById } = require('../../infrastructure/repositories/jobs-repo');
const { getJobView } = require('./get-job-view');

async function cancelJob(jobId, actor) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');

  const now = nowIso();
  await db('jobs').where({ id: jobId }).update({
    desired_state: 'cancel_requested',
    headline: 'Cancellation requested',
    updated_at: now,
  });

  await db('job_events').insert({
    id: newId('evt'),
    job_id: jobId,
    attempt_id: job.latestAttemptId || null,
    step_key: null,
    event_type: 'job.cancel_requested',
    severity: 'warn',
    sequence_no: 0,
    delivery_id: newId('delivery'),
    event_time: now,
    received_at: now,
    payload_json: toJson({
      requestedBy: actor?.sub || null,
    }),
  });

  return getJobView(jobId);
}

module.exports = { cancelJob };
