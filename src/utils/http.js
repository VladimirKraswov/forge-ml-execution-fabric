function buildPublicBaseUrl(req, configuredBaseUrl = '') {
  const configured = String(configuredBaseUrl || '').trim();
  if (configured) {
    return configured.replace(/\/+$/, '');
  }

  const proto = String(
    req.headers['x-forwarded-proto'] || req.protocol || 'http'
  ).split(',')[0].trim();

  const host = String(
    req.headers['x-forwarded-host'] || req.get('host') || ''
  ).split(',')[0].trim();

  if (!host) {
    return '';
  }

  return `${proto}://${host}`.replace(/\/+$/, '');
}

module.exports = { buildPublicBaseUrl };
