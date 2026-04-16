// @ts-check
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  // Directory where Playwright will look for test files
  testDir: './tests',

  // Global timeout per test — increased to 60 s for a JS-rendered app
  // that may have slow network or heavy client-side rendering
  timeout: 60000,

  use: {
    // Base URL prepended to all page.goto() calls that use a relative path
    baseURL: 'https://owlet-campus.com/',

    // Run tests in headless mode (no visible browser window)
    // Set to false locally when debugging to watch the browser
    headless: true,

    // Capture a screenshot automatically when a test fails
    // Attached to the HTML report for easy post-mortem inspection
    screenshot: 'only-on-failure',

    // Maximum time to wait for a navigation (page load / redirect) to complete
    navigationTimeout: 30000,

    // Maximum time to wait for a single action (click, fill, etc.) to complete
    actionTimeout: 15000,
  },

  // HTML reporter — never auto-open so the suite works safely in CI pipelines
  // Run `npx playwright show-report` locally to view the generated report
  reporter: [['html', { open: 'never' }]],
});
