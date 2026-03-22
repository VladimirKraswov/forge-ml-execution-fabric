exports.up = async function up(knex) {
  await knex.schema.createTable('users', (table) => {
    table.text('id').primary();
    table.text('username').notNullable().unique();
    table.text('password_hash').notNullable();
    table.text('role').notNullable();
    table.text('created_at').notNullable();
    table.text('updated_at').notNullable();
  });

  await knex.schema.createTable('runtime_profiles', (table) => {
    table.text('id').primary();
    table.text('profile_key').notNullable();
    table.integer('version').notNullable();
    table.text('status').notNullable();

    table.text('title').notNullable();
    table.text('description');

    table.text('runtime_image').notNullable();
    table.text('base_model_family').notNullable();

    table.text('capabilities_json').notNullable();
    table.text('supported_step_kinds_json').notNullable();
    table.text('resource_hints_json').notNullable();
    table.text('required_env_json').notNullable();
    table.text('launch_hints_json').notNullable();
    table.text('supported_config_versions_json').notNullable();

    table.text('created_at').notNullable();
    table.text('updated_at').notNullable();

    table.unique(['profile_key', 'version']);
  });
  await knex.schema.raw('create index idx_runtime_profiles_status on runtime_profiles(status)');
  await knex.schema.raw('create index idx_runtime_profiles_family on runtime_profiles(base_model_family)');

  await knex.schema.createTable('jobs', (table) => {
    table.text('id').primary();
    table.text('workspace_id');
    table.text('project_id');
    table.text('created_by_user_id');

    table.text('name').notNullable();
    table.text('job_kind').notNullable();

    table.text('status').notNullable();
    table.text('stage').notNullable();
    table.text('desired_state').notNullable();

    table.text('runtime_profile_id').notNullable();
    table.text('current_config_snapshot_id');
    table.text('latest_attempt_id');

    table.text('current_step_key');
    table.text('labels_json').notNullable();

    table.text('headline');
    table.text('terminal_reason');

    table.real('progress_percent').notNullable().defaultTo(0);
    table.text('created_at').notNullable();
    table.text('started_at');
    table.text('finished_at');
    table.text('updated_at').notNullable();

    table.foreign('runtime_profile_id').references('runtime_profiles.id');
  });
  await knex.schema.raw('create index idx_jobs_status_created on jobs(status, created_at desc)');
  await knex.schema.raw('create index idx_jobs_workspace_created on jobs(workspace_id, created_at desc)');
  await knex.schema.raw('create index idx_jobs_project_created on jobs(project_id, created_at desc)');

  await knex.schema.createTable('job_attempts', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.integer('attempt_no').notNullable();

    table.text('status').notNullable();
    table.text('stage').notNullable();

    table.text('runtime_image').notNullable();
    table.text('executor_version');

    table.text('host_info_json').notNullable();
    table.text('runtime_info_json').notNullable();

    table.text('first_seen_at');
    table.text('config_fetched_at');
    table.text('started_at');
    table.text('last_seen_at');
    table.text('finished_at');

    table.integer('exit_code');
    table.text('failure_reason');
    table.text('final_payload_received_at');
    table.integer('last_sequence_no');

    table.text('created_at').notNullable();
    table.text('updated_at').notNullable();

    table.unique(['job_id', 'attempt_no']);
    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_job_attempts_job_started on job_attempts(job_id, started_at desc)');
  await knex.schema.raw('create index idx_job_attempts_status_last_seen on job_attempts(status, last_seen_at)');

  await knex.schema.createTable('job_config_snapshots', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.integer('version_no').notNullable();
    table.text('config_version').notNullable();
    table.text('digest_sha256').notNullable();

    table.text('compiled_from_profile_id').notNullable();
    table.integer('compiled_from_profile_version').notNullable();

    table.text('snapshot_json').notNullable();

    table.text('created_by_user_id');
    table.text('created_at').notNullable();

    table.unique(['job_id', 'version_no']);
    table.unique(['digest_sha256']);
    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_job_config_snapshots_job on job_config_snapshots(job_id, version_no desc)');

  await knex.schema.createTable('job_pipeline_steps', (table) => {
    table.text('id').primary();
    table.text('config_snapshot_id').notNullable();

    table.text('step_key').notNullable();
    table.text('display_name').notNullable();
    table.text('step_kind').notNullable();

    table.integer('enabled').notNullable();
    table.text('depends_on_json').notNullable();
    table.text('run_if').notNullable();
    table.integer('order_index').notNullable();
    table.real('weight');
    table.text('params_json').notNullable();

    table.unique(['config_snapshot_id', 'step_key']);
    table.foreign('config_snapshot_id').references('job_config_snapshots.id').onDelete('CASCADE');
  });
  await knex.schema.raw(
    'create index idx_job_pipeline_steps_snapshot_order on job_pipeline_steps(config_snapshot_id, order_index)'
  );

  await knex.schema.createTable('job_step_runs', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id').notNullable();

    table.text('step_key').notNullable();
    table.text('step_kind').notNullable();
    table.text('status').notNullable();

    table.real('progress_current');
    table.real('progress_total');
    table.text('progress_unit');
    table.real('progress_percent');

    table.text('message');
    table.text('metrics_json');

    table.text('started_at');
    table.text('finished_at');
    table.integer('last_sequence_no');
    table.text('error_summary');

    table.unique(['attempt_id', 'step_key']);
    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_job_step_runs_job_attempt on job_step_runs(job_id, attempt_id)');
  await knex.schema.raw('create index idx_job_step_runs_status on job_step_runs(status)');

  await knex.schema.createTable('job_events', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id');
    table.text('step_key');

    table.text('event_type').notNullable();
    table.text('severity');
    table.integer('sequence_no');
    table.text('delivery_id').notNullable();

    table.text('event_time').notNullable();
    table.text('received_at').notNullable();

    table.text('payload_json').notNullable();

    table.unique(['attempt_id', 'delivery_id']);
    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_job_events_job_received on job_events(job_id, received_at asc)');
  await knex.schema.raw('create index idx_job_events_attempt_sequence on job_events(attempt_id, sequence_no asc)');
  await knex.schema.raw('create index idx_job_events_type on job_events(event_type)');

  await knex.schema.createTable('job_log_streams', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id').notNullable();
    table.text('step_key');
    table.text('stream_name').notNullable();
    table.text('created_at').notNullable();

    table.unique(['attempt_id', 'step_key', 'stream_name']);
    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });

  await knex.schema.createTable('job_log_chunks', (table) => {
    table.text('id').primary();
    table.text('stream_id').notNullable();
    table.integer('chunk_seq').notNullable();
    table.integer('offset_bytes').notNullable();
    table.integer('size_bytes').notNullable();

    table.text('encoding').notNullable();
    table.text('compression');

    table.text('text_payload');
    table.text('blob_key');

    table.text('emitted_at');
    table.text('received_at').notNullable();

    table.unique(['stream_id', 'chunk_seq']);
    table.foreign('stream_id').references('job_log_streams.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_job_log_chunks_stream_offset on job_log_chunks(stream_id, offset_bytes)');

  await knex.schema.createTable('job_artifacts', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id');
    table.text('step_key');

    table.text('artifact_type').notNullable();
    table.text('role');
    table.text('backend').notNullable();

    table.text('uri');
    table.text('storage_key');

    table.text('content_type');
    table.text('format');
    table.integer('size_bytes');
    table.text('checksum_sha256');

    table.text('metadata_json').notNullable();
    table.integer('is_primary').notNullable().defaultTo(0);
    table.integer('previewable').notNullable().defaultTo(0);
    table.text('sync_status').notNullable();

    table.text('created_at').notNullable();

    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_job_artifacts_job_primary on job_artifacts(job_id, is_primary, created_at desc)');
  await knex.schema.raw('create index idx_job_artifacts_attempt_step on job_artifacts(attempt_id, step_key)');
  await knex.schema.raw('create index idx_job_artifacts_type on job_artifacts(artifact_type)');

  await knex.schema.createTable('job_result_summaries', (table) => {
    table.text('job_id').primary();
    table.text('attempt_id').notNullable();

    table.text('outcome').notNullable();
    table.text('headline');
    table.text('primary_metrics_json').notNullable();
    table.text('summary_json').notNullable();

    table.text('created_at').notNullable();
    table.text('updated_at').notNullable();

    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });

  await knex.schema.createTable('huggingface_sync_states', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id');

    table.text('repo_id').notNullable();
    table.text('repo_type').notNullable();

    table.text('requested_revision');
    table.text('last_seen_revision');

    table.text('status').notNullable();
    table.text('manifest_json').notNullable();
    table.text('last_error');

    table.text('last_synced_at');
    table.text('next_retry_at');
    table.integer('retry_count').notNullable().defaultTo(0);

    table.text('created_at').notNullable();
    table.text('updated_at').notNullable();

    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_hf_sync_job on huggingface_sync_states(job_id)');
  await knex.schema.raw('create index idx_hf_sync_status_retry on huggingface_sync_states(status, next_retry_at)');

  await knex.schema.createTable('runtime_callback_credentials', (table) => {
    table.text('id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id');

    table.text('credential_type').notNullable();
    table.text('token_hash').notNullable();
    table.text('scope_json').notNullable();

    table.text('expires_at');
    table.text('revoked_at');
    table.text('first_used_at');
    table.text('bound_attempt_id');

    table.text('created_at').notNullable();

    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_runtime_credentials_job_type on runtime_callback_credentials(job_id, credential_type)');
  await knex.schema.raw('create index idx_runtime_credentials_expires on runtime_callback_credentials(expires_at)');

  await knex.schema.createTable('ingest_receipts', (table) => {
    table.text('delivery_id').primary();
    table.text('job_id').notNullable();
    table.text('attempt_id');
    table.text('endpoint_kind').notNullable();
    table.integer('sequence_no');
    table.text('payload_hash').notNullable();
    table.text('received_at').notNullable();

    table.foreign('job_id').references('jobs.id').onDelete('CASCADE');
    table.foreign('attempt_id').references('job_attempts.id').onDelete('CASCADE');
  });
  await knex.schema.raw('create index idx_ingest_receipts_attempt_seq on ingest_receipts(attempt_id, sequence_no)');
};

exports.down = async function down(knex) {
  await knex.schema.dropTableIfExists('ingest_receipts');
  await knex.schema.dropTableIfExists('runtime_callback_credentials');
  await knex.schema.dropTableIfExists('huggingface_sync_states');
  await knex.schema.dropTableIfExists('job_result_summaries');
  await knex.schema.dropTableIfExists('job_artifacts');
  await knex.schema.dropTableIfExists('job_log_chunks');
  await knex.schema.dropTableIfExists('job_log_streams');
  await knex.schema.dropTableIfExists('job_events');
  await knex.schema.dropTableIfExists('job_step_runs');
  await knex.schema.dropTableIfExists('job_pipeline_steps');
  await knex.schema.dropTableIfExists('job_config_snapshots');
  await knex.schema.dropTableIfExists('job_attempts');
  await knex.schema.dropTableIfExists('jobs');
  await knex.schema.dropTableIfExists('runtime_profiles');
  await knex.schema.dropTableIfExists('users');
};
