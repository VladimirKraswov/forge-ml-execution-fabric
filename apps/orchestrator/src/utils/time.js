function nowIso() {
  return new Date().toISOString();
}

function addMinutes(date, minutes) {
  return new Date(date.getTime() + minutes * 60 * 1000).toISOString();
}

function addHours(date, hours) {
  return new Date(date.getTime() + hours * 60 * 60 * 1000).toISOString();
}

module.exports = {
  nowIso,
  addMinutes,
  addHours,
};
