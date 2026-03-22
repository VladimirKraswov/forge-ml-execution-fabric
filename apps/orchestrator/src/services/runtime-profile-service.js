const { db } = require('../infrastructure/db/db');
const { newId } = require('../utils/ids');
const { nowIso } = require('../utils/time');
const { parseJson, toJson } = require('../utils/json');
const { CONFIG } = require('../config');

function mapProfile(row) {
  if (!row) return null;
  return {
    id: row.id,
    profileKey: row.profile_key,
    version: row.version,
    status: row.status,
    title: row.title,
    description: row.description,
    runtimeImage: row.runtime_image,
    baseModelFamily: row.base_model_family,
    capabilities: parseJson(row.capabilities_json, {}),
    supportedStepKinds: parseJson(row.supported_step_kinds_json, []),
    resourceHints: parseJson(row.resource_hints_json, {}),
    requiredEnv: parseJson(row.required_env_json, []),
    launchHints: parseJson(row.launch_hints_json, {}),
    supportedConfigVersions: parseJson(row.supported_config_versions_json, []),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

async function seedDefaultRuntimeProfiles() {
  const countRow = await db('runtime_profiles').count({ count: '*' }).first();
  const count = Number(countRow?.count || 0);
  if (count > 0) return;

  const now = nowIso();
  const defaults = [
    {
      id: newId('rp'),
      profile_key: 'trainer-service-qwen7b',
      version: 1,
      status: 'active',
      title: 'Trainer Service Qwen 7B',
      description: 'Local Docker runtime for the provided trainer-service image and callback contract',
      runtime_image: CONFIG.defaultRuntimeImage,
      base_model_family: 'qwen',
      capabilities_json: toJson({
        trainerService: true,
        train: true,
        merge: true,
        evaluate: true,
        publishHf: true,
        uploadArtifacts: true,
      }),
      supported_step_kinds_json: toJson([
        'bootstrap',
        'prepare_assets',
        'training',
        'merge_model',
        'evaluation',
        'publish_hf',
        'upload_artifacts',
      ]),
      resource_hints_json: toJson({
        gpus: 1,
        shmSize: '16g',
      }),
      required_env_json: toJson(['HF_TOKEN']),
      launch_hints_json: toJson({
        gpus: 'all',
        shmSize: '16g',
      }),
      supported_config_versions_json: toJson(['1.0', 'trainer-service/v1']),
      created_at: now,
      updated_at: now,
    },
  ];
  await db('runtime_profiles').insert(defaults);
}

async function listRuntimeProfiles() {
  const rows = await db('runtime_profiles').orderBy([
    { column: 'status', order: 'asc' },
    { column: 'profile_key', order: 'asc' },
    { column: 'version', order: 'desc' },
  ]);
  return rows.map(mapProfile);
}

async function getRuntimeProfileById(id) {
  const row = await db('runtime_profiles').where({ id }).first();
  return mapProfile(row);
}

async function createRuntimeProfile(input) {
  const now = nowIso();
  const row = {
    id: newId('rp'),
    profile_key: String(input.profileKey || '').trim(),
    version: Number(input.version || 1),
    status: String(input.status || 'active'),
    title: String(input.title || '').trim(),
    description: input.description == null ? null : String(input.description),
    runtime_image: String(input.runtimeImage || '').trim(),
    base_model_family: String(input.baseModelFamily || '').trim(),
    capabilities_json: toJson(input.capabilities || {}),
    supported_step_kinds_json: toJson(input.supportedStepKinds || []),
    resource_hints_json: toJson(input.resourceHints || {}),
    required_env_json: toJson(input.requiredEnv || []),
    launch_hints_json: toJson(input.launchHints || {}),
    supported_config_versions_json: toJson(input.supportedConfigVersions || ['1.0']),
    created_at: now,
    updated_at: now,
  };

  if (!row.profile_key || !row.title || !row.runtime_image || !row.base_model_family) {
    throw new Error('profileKey, title, runtimeImage and baseModelFamily are required');
  }

  await db('runtime_profiles').insert(row);
  return getRuntimeProfileById(row.id);
}

async function patchRuntimeProfile(id, patch) {
  const existing = await db('runtime_profiles').where({ id }).first();
  if (!existing) throw new Error('Runtime profile not found');

  const next = {
    updated_at: nowIso(),
  };

  if (patch.status !== undefined) next.status = String(patch.status);
  if (patch.title !== undefined) next.title = String(patch.title || '').trim();
  if (patch.description !== undefined) next.description = patch.description == null ? null : String(patch.description);
  if (patch.runtimeImage !== undefined) next.runtime_image = String(patch.runtimeImage || '').trim();
  if (patch.baseModelFamily !== undefined) next.base_model_family = String(patch.baseModelFamily || '').trim();
  if (patch.capabilities !== undefined) next.capabilities_json = toJson(patch.capabilities || {});
  if (patch.supportedStepKinds !== undefined) next.supported_step_kinds_json = toJson(patch.supportedStepKinds || []);
  if (patch.resourceHints !== undefined) next.resource_hints_json = toJson(patch.resourceHints || {});
  if (patch.requiredEnv !== undefined) next.required_env_json = toJson(patch.requiredEnv || []);
  if (patch.launchHints !== undefined) next.launch_hints_json = toJson(patch.launchHints || {});
  if (patch.supportedConfigVersions !== undefined) next.supported_config_versions_json = toJson(patch.supportedConfigVersions || []);

  await db('runtime_profiles').where({ id }).update(next);
  return getRuntimeProfileById(id);
}

module.exports = {
  seedDefaultRuntimeProfiles,
  listRuntimeProfiles,
  getRuntimeProfileById,
  createRuntimeProfile,
  patchRuntimeProfile,
};
