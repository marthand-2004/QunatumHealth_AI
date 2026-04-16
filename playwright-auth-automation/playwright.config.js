// @ts-check
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',

  // Global timeout per test — increased for JS-rendered app
  timeout: 60000,

  use: {
    baseURL: 'https://owlet-campus.com/',
    headless: true,
    // Wait for network to be idle after navigation
    navigationTimeout: 30000,
    actionTimeout: 15000,
  },

  // HTML reporter — never auto-open so it works in CI
  reporter: [['html', { open: 'never' }]],
});
